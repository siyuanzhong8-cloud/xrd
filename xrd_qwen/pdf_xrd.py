#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从文献 PDF 自动找出 XRD 图并裁剪，再调用 xrd_vl 做四项识图。

流程：
1) 从 PDF 定位并裁出 XRD 图；模型同时读出 x 轴刻度位置、曲线数/颜色/图例、靶材线索（替代 Windows OCR）。
2) 用刻度把裁图数字化为 .xy，重绘并生成原图叠加校验图，可选让模型复核；再用 xy_peaks 寻峰，
   把峰表+裁图交给 xrd_vl 做物相/合成判断。
3) 可选：按候选物相下载 CIF，用 DARA+BGMN 拟合（需单独环境，见 phase_fit.py）。

API Key 只从环境变量 DASHSCOPE_API_KEY 读取，绝不写入文件或打印。
不依赖 Windows OCR，Linux 与 Windows 均可运行。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

# 保证与本脚本同目录的 xrd_vl.py 可被导入（从其他工作目录启动时也行）
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import xrd_vl
from xrd_vl import (
    ALL_TASKS,
    DEFAULT_MAX_SIDE,
    DEFAULT_MODEL,
    call_model,
    call_model_hedged,
    encode_image,
    extract_json,
    format_usage,
    get_api_key,
    parse_tasks,
    process_one,
    redact_secret,
    write_peaks_csv,
    write_report_md,
    write_summary_csv,
)

try:
    import pymupdf
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "未安装 PyMuPDF。请先执行：pip install -r requirements.txt\n"
        f"（原始错误：{exc}）"
    ) from exc


DEFAULT_DPI = 200
DEFAULT_LOCATE_MODEL = "deepseek-v4.1-flash-1017"
DEFAULT_PHASE_MODEL = "qwen3.5-omni-flash"
LOCATE_COMPARE_MODELS = ("deepseek-v4.1-flash-1017", "qwen3.6-flash")
DEFAULT_LOCATE_MAX_SIDE = 1280
DEFAULT_CALL_TIMEOUT = 25.0
DEFAULT_HEDGE_AFTER = 12.0
DEFAULT_LOCATE_MAX_TOKENS = 1536
DEFAULT_PAD = 0.03  # 裁框外扩比例，避免切掉坐标轴或刻度
REFINE_PAD = 0.15  # 二次校正时先多留一些边，再让模型收紧
MIN_BOX_FRAC = 0.008  # 相对整页面积的最小框，过滤误检小碎片
PDF_SUFFIXES = {".pdf"}

# 筛候选页用的正向关键词（文本已做 ligature / 大小写归一）
POS_KEYWORDS = (
    "xrd",
    "pxrd",
    "gxrd",
    "x-ray diffract",
    "xray diffract",
    "x-ray powder",
    "xray powder",
    "powder diffract",
    "diffractogram",
    "diffractogram",
    "diffraction pattern",
    "diffraction patterns",
    "2theta",
    "2-theta",
    "2 theta",
    "2θ",
    "2 θ",
    "θ-2θ",
    "theta-2theta",
    "粉末衍射",
    "x射线衍射",
    "x 射线衍射",
    "射线衍射",
    "衍射图谱",
    "衍射图",
    "衍射峰",
    "衍射花样",
    "rietveld",
    "精修",
)

# 仅出现这些、且没有明确 XRD 字样时，不把页面当作候选（降低 SAED / 电子衍射误召）
ELECTRON_DIFF_HINTS = (
    "electron diffraction",
    "selected area electron",
    "saed",
    "电子衍射",
    "选区电子衍射",
    "ted pattern",
)

# 图注里出现这些且没有 XRD 正向词，基本可判断不是 XRD 图（仍可能因正文命中而送去模型）
NON_XRD_CAPTION = (
    "raman",
    "xps",
    "photoelectron",
    "tem image",
    "tem images",
    "sem image",
    "sem images",
    "hrtem",
    "stem image",
    "afm",
    "tga",
    "thermogravimetric",
    "dsc ",
    "uv-vis",
    "uv–vis",
    "ftir",
    "ft-ir",
    "infrared",
    "nmr",
    "impedance",
    "cyclic voltammetr",
    "hysteresis",
    "magnetization",
    "eds",
    "edx",
    "energy-dispersive",
    "mapping",
    "n2 adsorption",
    "isotherm",
    "bet ",
    "photoluminescence",
    "拉曼",
    "光电子",
    "热重",
    "磁滞",
    "扫描电镜",
    "透射电镜",
)

CAP_RE = re.compile(
    r"^\s*(?:fig(?:ure)?|图)\s*\.?\s*(\d+[a-z]?)\s*([\.:：].+)$",
    re.I | re.S,
)

# 官方价目粗算（元 / 百万 token），以控制台为准
PRICE_PER_MILLION = {
    "qwen3.8-flash": (0.8, 2.7),
    "qwen3.6-flash": (1.2, 7.2),
    "qwen3-vl-plus": (1.0, 10.0),
    "qwen3.5-flash": (0.2, 2.0),  # 若开通；仅作估算
    "qwen3.7-flash": (0.3, 1.2),
    "qwen3.5-omni-flash": (2.2, 13.3),  # 文本/图片输入 + 文本输出；以控制台为准
    # deepseek-v4.1-flash-1017：闲时/忙时在 estimate_cost_cny 里按北京时间选择
    "deepseek-v4.1-flash-1017": (1.0, 4.0),
}


def configure_stdio() -> None:
    """Windows 控制台下尽量用 UTF-8，避免中文与 2θ 乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def normalize_text(text: str) -> str:
    """还原 PDF 常见连字、破折号，并压缩空白。"""
    if not text:
        return ""
    repl = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00ad": "",
        "\u00a0": " ",
    }
    for a, b in repl.items():
        text = text.replace(a, b)
    return " ".join(text.split())


def norm_key(text: str) -> str:
    return normalize_text(text).lower()


def parse_pages(spec: str | None, page_count: int) -> list[int]:
    """解析 --pages，返回 1-based 且落在 [1, page_count] 内的页码。"""
    if not spec or not str(spec).strip():
        return list(range(1, page_count + 1))
    pages: set[int] = set()
    for raw in re.split(r"[,，\s]+", str(spec).strip()):
        if not raw:
            continue
        m = re.match(r"^(\d+)$", raw)
        if m:
            pages.add(int(m.group(1)))
            continue
        m = re.match(r"^(\d+)\s*[-–~至到]\s*(\d+)$", raw)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            pages.update(range(lo, hi + 1))
            continue
        raise ValueError(f"无法解析页码 “{raw}”。示例：5  或  4-6,8")
    chosen = [p for p in sorted(pages) if 1 <= p <= page_count]
    if not chosen:
        raise ValueError(f"指定的页码与该 PDF 的 1–{page_count} 页没有交集。")
    return chosen


def collect_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in PDF_SUFFIXES:
            raise ValueError(f"不是 PDF 文件：{input_path}")
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in PDF_SUFFIXES
        )
        if not files:
            raise ValueError(f"文件夹里没有 PDF：{input_path}")
        return files
    raise FileNotFoundError(f"找不到输入路径：{input_path}")


def safe_stem(name: str) -> str:
    """去掉 Windows 文件名非法字符，保留论文常见的连字符。"""
    cleaned = re.sub(r'[<>:"/\\\\|?*]', "_", name)
    cleaned = cleaned.strip(" .")
    return cleaned or "pdf"


def keyword_hits(text: str) -> list[str]:
    key = norm_key(text)
    # 2θ 在 lower() 后仍是 2θ；同时也搜原文字
    hits = [kw for kw in POS_KEYWORDS if kw in key or kw in text]
    return hits


def is_electron_diffraction_only(text: str, hits: list[str]) -> bool:
    """正文只有电子衍射相关词、没有 XRD / 2θ 等时，不当作候选页。"""
    key = norm_key(text)
    if not any(h in key for h in ELECTRON_DIFF_HINTS):
        return False
    strong = {"xrd", "pxrd", "gxrd", "x-ray diffract", "xray diffract", "x-ray powder",
              "粉末衍射", "x射线衍射", "x 射线衍射", "2theta", "2-theta", "2θ", "rietveld"}
    return not any(h in strong or "x-ray diffract" in h or "射线衍射" in h for h in hits)


def extract_captions(page: "pymupdf.Page") -> list[dict[str, Any]]:
    """从文本层取出形如 Figure 1. / 图 1： 的图注块。"""
    caps: list[dict[str, Any]] = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, raw, *_rest = block
        text = normalize_text(str(raw))
        m = CAP_RE.match(text)
        if not m:
            continue
        caps.append(
            {
                "num": m.group(1),
                "text": text,
                "rect": (float(x0), float(y0), float(x1), float(y1)),
            }
        )
    return caps


def page_has_visuals(page: "pymupdf.Page") -> bool:
    """粗略判断页面是否可能含图（嵌入图或较丰富的矢量绘制）。"""
    images = page.get_images(full=True)
    if images:
        return True
    drawings = page.get_drawings()
    # 坐标轴、曲线常以矢量形式嵌入；数量很少多半是装饰线
    return len(drawings) >= 8


def is_candidate_page(
    page: "pymupdf.Page",
    text: str,
    captions: list[dict[str, Any]],
    *,
    textless_fallback: bool = False,
) -> tuple[bool, str, str]:
    """用关键词 + 是否有图，便宜地决定要不要送给视觉模型。

    返回 (是否候选, 原因, 类别)。类别为 caption / body / textless / skip。
    """
    blob = text + " " + " ".join(c["text"] for c in captions)
    hits = keyword_hits(blob)
    if not hits:
        # 「打印成 PDF」一类文件往往没有文本层，关键词筛页会把整篇都跳过。
        # 仅在全文几乎无字时启用：有嵌入图的页送给视觉模型，参考文献页通常无图可跳过。
        if textless_fallback:
            if page.get_images(full=True):
                return True, "PDF 无文本层，本页含嵌入图，送视觉模型筛查", "textless"
            return False, "PDF 无文本层，本页无嵌入图（多半是参考文献）", "skip"
        return False, "无 XRD 关键词", "skip"
    if is_electron_diffraction_only(blob, hits):
        return False, "仅电子衍射相关词，已跳过", "skip"
    cap_hits = [c for c in captions if keyword_hits(c["text"])]
    if cap_hits:
        return True, "图注含 XRD 关键词：" + ",".join(keyword_hits(" ".join(c["text"] for c in cap_hits))), "caption"
    if page_has_visuals(page):
        return True, "正文含关键词且页面有图：" + ",".join(hits[:6]), "body"
    return False, "有关键词但本页几乎无图（多为方法/讨论文字）", "skip"


def rect_to_box1000(rect: tuple[float, float, float, float], page_rect: "pymupdf.Rect") -> list[int]:
    w = max(page_rect.width, 1e-6)
    h = max(page_rect.height, 1e-6)
    x0 = (rect[0] - page_rect.x0) / w * 1000
    y0 = (rect[1] - page_rect.y0) / h * 1000
    x1 = (rect[2] - page_rect.x0) / w * 1000
    y1 = (rect[3] - page_rect.y0) / h * 1000
    return [int(round(v)) for v in (x0, y0, x1, y1)]


def clamp_box(box: list[float], *, lo: float = 0, hi: float = 1000) -> list[float] | None:
    if len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0 = min(max(x0, lo), hi)
    x1 = min(max(x1, lo), hi)
    y0 = min(max(y0, lo), hi)
    y1 = min(max(y1, lo), hi)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return [x0, y0, x1, y1]


def pad_box(box: list[float], pad: float, *, left: float | None = None, bottom: float | None = None) -> list[float]:
    """外扩裁框。默认给左边、底边更大余量，避免切掉轴标题和刻度数字。"""
    x0, y0, x1, y1 = box
    pw = (x1 - x0) * pad
    ph = (y1 - y0) * pad
    # 底边、左边需要容纳刻度数字和轴标题，比上/右多留一些
    pad_left = pw * 1.6 if left is None else (x1 - x0) * left
    pad_bottom = ph * 2.0 if bottom is None else (y1 - y0) * bottom
    return [
        max(0.0, x0 - pad_left),
        max(0.0, y0 - ph),
        min(1000.0, x1 + pw),
        min(1000.0, y1 + pad_bottom),
    ]


def extend_box_to_caption(box: list[float], cap: dict[str, Any] | None, page_rect: "pymupdf.Rect") -> list[float]:
    """若已匹配到图注，把框的底边伸到图注上沿之前，把横轴刻度数字包进来。"""
    if not cap:
        return box
    cap_box = rect_to_box1000(cap["rect"], page_rect)
    # 图注上沿之上留 4‰ 页高的缝，避免把 "Figure 1." 裁进图
    target = max(box[3], cap_box[1] - 4)
    if target > box[3]:
        box = list(box)
        box[3] = min(1000.0, target)
    return box


def box_area_frac(box: list[float]) -> float:
    return max(0.0, (box[2] - box[0]) / 1000.0) * max(0.0, (box[3] - box[1]) / 1000.0)


def iou(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def dedupe_boxes(figures: list[dict[str, Any]], iou_thr: float = 0.7) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for fig in sorted(figures, key=lambda z: -float(z.get("confidence") or 0)):
        if any(iou(fig["box_1000"], k["box_1000"]) >= iou_thr for k in kept):
            continue
        kept.append(fig)
    # 阅读顺序：从上到下、从左到右
    kept.sort(key=lambda z: (z["box_1000"][1], z["box_1000"][0]))
    return kept


def map_box_from_crop(crop_box: list[float], refined: list[float]) -> list[float]:
    """把二次校正图（0-1000）上的框映射回整页 0-1000。"""
    cx0, cy0, cx1, cy1 = crop_box
    rw, rh = cx1 - cx0, cy1 - cy0
    return [
        cx0 + refined[0] / 1000.0 * rw,
        cy0 + refined[1] / 1000.0 * rh,
        cx0 + refined[2] / 1000.0 * rw,
        cy0 + refined[3] / 1000.0 * rh,
    ]


_USAGE_LOCK = threading.Lock()


def add_usage(total: dict[str, int], usage: dict | None) -> None:
    if not usage:
        return
    with _USAGE_LOCK:
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v = usage.get(k)
            if isinstance(v, int):
                total[k] = total.get(k, 0) + v


def _deepseek_unit_prices() -> tuple[float, float]:
    """华北 2 刊例：闲时（UTC+8 22:00–08:00）输入 1 / 输出 4，忙时 2 / 8（元/百万 token）。"""
    from datetime import datetime, timezone, timedelta
    hour = datetime.now(timezone(timedelta(hours=8))).hour
    off = hour >= 22 or hour < 8
    return (1.0, 4.0) if off else (2.0, 8.0)


def estimate_cost_cny(model: str, usage: dict[str, int] | None) -> str:
    if not usage:
        return "未知"
    if "deepseek-v4.1-flash" in (model or ""):
        pin, pout = _deepseek_unit_prices()
        hour_note = "闲时" if pin == 1.0 else "忙时"
        cost = (usage.get("prompt_tokens", 0) / 1e6) * pin + (
            usage.get("completion_tokens", 0) / 1e6
        ) * pout
        return f"约 {cost:.4f} 元（按 {model} {hour_note}刊例估算，以控制台为准）"
    prices = PRICE_PER_MILLION.get(model)
    if not prices:
        return f"未内置 {model} 单价，请到控制台查看"
    pin, pout = prices
    cost = (usage.get("prompt_tokens", 0) / 1e6) * pin + (
        usage.get("completion_tokens", 0) / 1e6
    ) * pout
    return f"约 {cost:.4f} 元（按 {model} 公开价估算，未扣免费额度，以控制台为准）"


def render_page_png(
    page: "pymupdf.Page",
    dest: Path,
    dpi: int | None = None,
    max_side: int | None = None,
) -> Path:
    """渲染整页。定位阶段用 max_side 限制长边；裁图仍用较高 DPI。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if max_side and max_side > 0:
        rect = page.rect
        zoom = max_side / max(float(max(rect.width, rect.height)), 1e-6)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    else:
        pix = page.get_pixmap(dpi=int(dpi or DEFAULT_DPI), alpha=False)
    pix.save(str(dest))
    return dest


def crop_page_region(
    page: "pymupdf.Page",
    box_1000: list[float],
    dest: Path,
    dpi: int,
) -> Path:
    """按整页 0-1000 框，直接从 PDF 矢量/位图渲染裁区，避免二次取样。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = page.rect
    clip = pymupdf.Rect(
        r.x0 + box_1000[0] / 1000.0 * r.width,
        r.y0 + box_1000[1] / 1000.0 * r.height,
        r.x0 + box_1000[2] / 1000.0 * r.width,
        r.y0 + box_1000[3] / 1000.0 * r.height,
    ) & r
    if clip.is_empty or clip.width < 2 or clip.height < 2:
        raise ValueError("裁剪区域无效或过小")
    pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    pix.save(str(dest))
    return dest


def caption_for_box(
    box_1000: list[float],
    captions: list[dict[str, Any]],
    page_rect: "pymupdf.Rect",
) -> dict[str, Any] | None:
    """取位于框下方、横向有重叠、且间隙最近的图注。"""
    best: tuple[float, dict[str, Any]] | None = None
    for cap in captions:
        cap_box = rect_to_box1000(cap["rect"], page_rect)
        gap = cap_box[1] - box_1000[3]
        # 允许少量重叠（图注偶尔被框进一点）以及图下方约 15% 页高的距离
        if gap < -40 or gap > 180:
            continue
        overlap = min(box_1000[2], cap_box[2]) - max(box_1000[0], cap_box[0])
        width = max(1.0, min(box_1000[2] - box_1000[0], cap_box[2] - cap_box[0]))
        if overlap < 0.25 * width:
            continue
        if best is None or abs(gap) < best[0]:
            best = (abs(gap), cap)
    return None if best is None else best[1]


def extract_radiation_hint(doc: "pymupdf.Document") -> str:
    """通读全文，抽出辐射源/靶材句子（常在实验方法页，而不在插图页）。"""
    blobs = []
    for page in doc:
        blobs.append(nearby_context(page.get_text("text"), limit=400))
    text = " ".join(b for b in blobs if b)
    return text[:800]


def nearby_context(text: str, limit: int = 600) -> str:
    """从页面正文里抽出与辐射源、靶材有关的句子，帮助识图。"""
    lines = []
    for raw in re.split(r"(?<=[。．.!?])\s+", normalize_text(text)):
        key = norm_key(raw)
        if any(
            s in key
            for s in (
                "cu k",
                "co k",
                "mo k",
                "cu kα",
                "co kα",
                "radiation",
                "incident",
                "靶",
                "辐射",
                "wavelength",
                "λ =",
                "lambda",
                "jcpds",
                "icdd",
                "pdf #",
                "pdf no",
            )
        ):
            lines.append(raw.strip())
    blob = " ".join(lines)
    return blob[:limit]


def build_locate_prompt(page_no: int, captions: list[dict[str, Any]], page_rect: "pymupdf.Rect") -> str:
    hints = []
    for cap in captions:
        box = rect_to_box1000(cap["rect"], page_rect)
        hints.append(
            f"- 图注「{cap['text'][:240]}」文本框约在归一化坐标 {box}（0-1000，原点左上）。"
        )
    hint_block = "\n".join(hints) if hints else "（文本层未解析到标准 Figure/图 图注）"
    schema = {
        "page_has_xrd": True,
        "figures": [
            {
                "figure_label": "Figure 1",
                "description": "简短中文描述，例如堆叠的粉末 XRD 图谱",
                "box_1000": [120, 80, 880, 420],
                "confidence": 0.85,
                "notes": "从图上读到的样品名、辐射源、卡片号等，没有则 null",
            }
        ],
        "rejected": [
            {"figure_label": "Figure 3", "reason": "这是 TEM，不是 XRD"}
        ],
    }
    return "\n".join(
        [
            "请判断这张论文页面里是否含有 X 射线粉末/薄膜衍射图（XRD / PXRD / θ-2θ 强度图，含 Rietveld 精修图）。",
            f"当前是第 {page_no} 页。",
            "",
            "只把真正的实验室 X 射线粉末/薄膜衍射图框出来（横坐标为 2θ 或 2theta 的强度图，含实验室 XRD 的 Rietveld 图）。",
            "以下都不是目标图，必须排除：",
            "TEM / HRTEM / SEM / STEM 照片、选区电子衍射（SAED）、电子衍射、XPS、Raman、TGA/DSC、",
            "UV-Vis、FTIR、NMR、EDS/EDX、磁滞回线、N2 吸脱附、示意图、流程图、结构式、表格、",
            "中子衍射图、横坐标为晶面间距 d / Å、散射矢量 Q 或飞行时间 TOF 的图（即使做了 Rietveld）。",
            "",
            "坐标约定：",
            "- box_1000 为相对于整页图像的归一化框 [x0, y0, x1, y1]，范围 0 到 1000，原点在左上角。",
            "- 框必须完整包含：坐标轴、轴标题、刻度数字、图例、面板标签（如 0 K、800 °C）、图内 hkl / 符号标注。",
            "- 不要包含图下方的图注文字（Figure / 图 x. ...），图注我会从 PDF 文本层另行提取。",
            "- 也不要包含页眉页脚、栏外正文。",
            "- 同一幅 Figure 里若全部是 XRD 子图（例如按样品堆叠的多条谱），请用一个框包住整幅图。",
            "- 若一页有多幅独立的 XRD 图（如 Figure 1 和 Figure 2），请分别给出各自的框。",
            "- 若一幅拼图里只有部分子图是 XRD，只框 XRD 子图。",
            "",
            "本页文本层图注及大致位置（供参考，仍须以图像为准）：",
            hint_block,
            "",
            "请只输出一个 JSON 对象，不要 Markdown 围栏，不要其它文字。结构如下：",
            json.dumps(schema, ensure_ascii=False, indent=2),
            "若没有 XRD 图，page_has_xrd 为 false，figures 为空数组，并在 rejected 里简要说明页上其它图是什么。",
        ]
    )


def build_refine_prompt() -> str:
    schema = {
        "is_xrd": True,
        "box_1000": [30, 40, 970, 920],
        "cut_off": [],
        "reason": "如判定不是 XRD，说明原因；是 XRD 则 null",
    }
    return "\n".join(
        [
            "这是从论文页面中裁出的一块候选区域（已经略微外扩）。",
            "请判断它是否是 X 射线衍射图（含 Rietveld 精修图）。",
            "如果是，请给出更精确的框，必须完整包含坐标轴、轴标题、刻度数字、图例、面板标签和全部衍射曲线。",
            "若这张裁图里还拼了相邻子图（Tafel/CV、柱状图、XPS、示意图），只框 XRD 这一块（2θ–强度图），不要把它们包进来。",
            "不要切掉左轴或底轴的文字；不要包含图注（Figure / 图 x.）以及大段正文。",
            "box_1000 相对于【本张裁图】，[x0,y0,x1,y1]，0 到 1000，原点左上。",
            "若原框已经合适，可基本沿用（例如 [20,20,980,980] 这类略留边的框）。",
            "若不是实验室 2θ XRD（例如 TEM、XPS、Raman、SAED、中子衍射、d–强度图、Q 或 TOF 图），is_xrd 设为 false，box_1000 可为空数组。",
            "只输出 JSON，不要其它文字：",
            json.dumps(schema, ensure_ascii=False, indent=2),
        ]
    )


def parse_locate_result(parsed: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        return []
    if parsed.get("page_has_xrd") is False:
        figs = parsed.get("figures") or []
        if not figs:
            return []
    figures: list[dict[str, Any]] = []
    for item in parsed.get("figures") or []:
        if not isinstance(item, dict):
            continue
        box = clamp_box(item.get("box_1000") or item.get("bbox") or [])
        if not box:
            continue
        if box_area_frac(box) < MIN_BOX_FRAC:
            continue
        conf = item.get("confidence")
        try:
            conf_v = float(conf) if conf is not None else 0.7
        except (TypeError, ValueError):
            conf_v = 0.7
        figures.append(
            {
                "figure_label": str(item.get("figure_label") or "").strip(),
                "description": str(item.get("description") or "").strip(),
                "box_1000": box,
                "confidence": conf_v,
                "notes": item.get("notes"),
            }
        )
    return dedupe_boxes(figures)


def draw_overlay(page_png: Path, figures: list[dict[str, Any]], dest: Path) -> None:
    """在页面缩略图上画框，便于人工抽查裁切是否完整。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(page_png) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        w, h = im.size
        colors = [(220, 30, 30), (30, 90, 200), (20, 150, 80), (180, 100, 0)]
        for i, fig in enumerate(figures):
            x0, y0, x1, y1 = fig["box_1000"]
            rect = [x0 / 1000 * w, y0 / 1000 * h, x1 / 1000 * w, y1 / 1000 * h]
            color = colors[i % len(colors)]
            draw.rectangle(rect, outline=color, width=max(2, w // 400))
            label = fig.get("figure_label") or f"#{i + 1}"
            ty = max(0, rect[1] - 16)
            draw.text((rect[0] + 2, ty), label, fill=color)
        im.save(dest, format="PNG")



def _score_locate_box(page_png: Path, box_1000: list[float]) -> float:
    """用程序化坐标框检测给定位框打分：框过宽（包进邻图）会扣分。"""
    try:
        from trace_xrd import choose_frame, load_rgb
        rgb = load_rgb(page_png)
        H, W = rgb.shape[:2]
        x0 = int(max(0, box_1000[0] / 1000.0 * W))
        y0 = int(max(0, box_1000[1] / 1000.0 * H))
        x1 = int(min(W, box_1000[2] / 1000.0 * W))
        y1 = int(min(H, box_1000[3] / 1000.0 * H))
        if x1 - x0 < 20 or y1 - y0 < 20:
            return -9.0
        crop = rgb[y0:y1, x0:x1]
        fr = choose_frame(crop)
        if not fr:
            return -2.0
        score = float(fr.get("score") or 0)
        n_ticks = int(fr.get("n_ticks") or 0)
        # 完整坐标轴（8 个 10° 主刻度）远好于被裁切的半幅图
        score += 1.6 * min(n_ticks, 12)
        cw, ch = crop.shape[1], crop.shape[0]
        fx0, fy0, fx1, fy1 = fr["box"]
        # 数据框顶到裁图边缘，多半把轴裁断了
        if fx1 >= cw - 6 and n_ticks < 7:
            score -= 6.0
        if fx0 <= 5 and n_ticks < 7:
            score -= 3.0
        if fy1 >= ch - 4 and n_ticks < 7:
            score -= 3.0
        return score
    except Exception:
        return 0.0


def _iou1000(a: list[float], b: list[float]) -> float:
    ox = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    oy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ox * oy
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def merge_locate_results(
    page_png: Path,
    results: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """合并两个定位模型的框，按程序化打分挑更好的。"""
    merged_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    all_figs: list[dict[str, Any]] = []
    raw_status = "ok"
    errors = []
    picked_models = []
    for model, loc in results:
        if loc.get("usage"):
            add_usage(merged_usage, loc["usage"])
        if loc.get("status") != "ok":
            errors.append(f"{model}: {loc.get('error') or loc.get('status')}")
            if not loc.get("figures"):
                continue
        for fig in loc.get("figures") or []:
            rec = dict(fig)
            rec["_locate_model"] = model
            rec["_score"] = _score_locate_box(page_png, rec.get("box_1000") or [0, 0, 0, 0])
            all_figs.append(rec)
    chosen: list[dict[str, Any]] = []
    for fig in sorted(all_figs, key=lambda z: -float(z.get("_score") or 0)):
        box = fig.get("box_1000") or [0, 0, 0, 0]
        rival = None
        for c in chosen:
            same_label = (
                (fig.get("figure_label") or "")
                and (fig.get("figure_label") == (c.get("figure_label") or ""))
            )
            if _iou1000(box, c.get("box_1000") or [0, 0, 0, 0]) >= 0.2 or same_label:
                rival = c
                break
        if rival is None:
            chosen.append(fig)
            picked_models.append(fig.get("_locate_model"))
            continue
        # 已有重叠框，保留分数更高者
        if float(fig.get("_score") or 0) > float(rival.get("_score") or 0) + 0.3:
            chosen.remove(rival)
            chosen.append(fig)
    out = {
        "figures": [{k: v for k, v in f.items() if not k.startswith("_")} | {"locate_model_used": f.get("_locate_model"), "locate_score": f.get("_score")} for f in chosen],
        "usage": merged_usage,
        "status": "ok" if chosen or raw_status == "ok" else "failed",
        "error": "; ".join(errors),
        "locate_compare": {m: (loc.get("status"), loc.get("elapsed_s"), len(loc.get("figures") or [])) for m, loc in results},
        "elapsed_s": max((loc.get("elapsed_s") or 0) for _, loc in results) if results else None,
    }
    # carry send size from first success
    for _, loc in results:
        if loc.get("send_size"):
            out["send_size"] = loc["send_size"]
            out["nbytes"] = loc.get("nbytes")
            out["mime"] = loc.get("mime")
            break
    return out


def locate_page_auto(
    page_png: Path,
    prompt: str,
    *,
    api_key: str,
    max_side: int,
    max_retries: int,
    thinking: bool,
    dry_run: bool,
    timeout: float,
    max_tokens: int,
    models: tuple[str, ...] = LOCATE_COMPARE_MODELS,
) -> dict[str, Any]:
    """两个定位模型并行，再按坐标框质量挑选。不做 hedge，避免变成四路请求。"""
    if dry_run:
        return locate_page(
            page_png, prompt, model=models[0], api_key=api_key, max_side=max_side,
            max_retries=max_retries, thinking=thinking, dry_run=True, timeout=timeout,
            max_tokens=max_tokens, hedge_after=0, prefer_jpeg=True,
        )
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results: list[tuple[str, dict[str, Any]]] = []
    def _one(m: str):
        try:
            return m, locate_page(
                page_png, prompt, model=m, api_key=api_key, max_side=max_side,
                max_retries=max_retries, thinking=False, dry_run=False, timeout=timeout,
                max_tokens=max_tokens, hedge_after=0, prefer_jpeg=True,
            )
        except Exception as exc:
            return m, {"status": "failed", "error": redact_secret(str(exc), api_key), "figures": [], "usage": None, "elapsed_s": None}
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futs = [pool.submit(_one, m) for m in models]
        for fut in as_completed(futs):
            results.append(fut.result())
    # stable order
    order = {m: i for i, m in enumerate(models)}
    results.sort(key=lambda z: order.get(z[0], 99))
    merged = merge_locate_results(page_png, results)
    return merged



CALIB_MODEL_WHEN_AUTO = "qwen3.6-flash"


def resolve_vision_model(name: str | None, fallback: str = CALIB_MODEL_WHEN_AUTO) -> str:
    """auto/compare 只用于定位并行，读刻度/二次校正仍要用一个真实视觉模型。"""
    n = (name or "").strip()
    if not n or n.lower() in {"auto", "compare"}:
        return fallback
    return n


def locate_page(
    page_png: Path,
    prompt: str,
    *,
    model: str,
    api_key: str,
    max_side: int,
    max_retries: int,
    thinking: bool,
    dry_run: bool,
    timeout: float = DEFAULT_CALL_TIMEOUT,
    max_tokens: int = DEFAULT_LOCATE_MAX_TOKENS,
    hedge_after: float | None = DEFAULT_HEDGE_AFTER,
    prefer_jpeg: bool = True,
) -> dict[str, Any]:
    encoded = encode_image(page_png, max_side=max_side, prefer_jpeg=prefer_jpeg)
    result: dict[str, Any] = {
        "figures": [],
        "raw": None,
        "usage": None,
        "elapsed_s": None,
        "error": "",
        "status": "",
        "send_size": encoded["send_size"],
        "nbytes": encoded.get("nbytes"),
        "mime": encoded.get("mime"),
    }
    if dry_run:
        result["status"] = "dry-run"
        return result
    started = time.perf_counter()
    raw, usage = call_model_hedged(
        encoded["data_url"],
        prompt,
        model=model,
        api_key=api_key,
        max_retries=max(1, min(int(max_retries), 2)),
        thinking=thinking,
        timeout=timeout,
        max_tokens=max_tokens,
        hedge_after=hedge_after,
    )
    result["elapsed_s"] = round(time.perf_counter() - started, 2)
    result["usage"] = usage
    result["raw"] = raw
    parsed, err = extract_json(raw)
    if parsed is None:
        result["status"] = "parse_failed"
        result["error"] = err or "定位结果解析失败"
        return result
    result["figures"] = parse_locate_result(parsed)
    result["rejected"] = parsed.get("rejected") if isinstance(parsed, dict) else None
    result["page_has_xrd"] = parsed.get("page_has_xrd") if isinstance(parsed, dict) else None
    result["status"] = "ok"
    return result


def refine_one(
    page: "pymupdf.Page",
    fig: dict[str, Any],
    tmp_path: Path,
    *,
    model: str,
    api_key: str,
    dpi: int,
    pad: float,
    max_side: int,
    max_retries: int,
    thinking: bool,
    timeout: float = DEFAULT_CALL_TIMEOUT,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """对单个候选框做二次校正：在外扩区域内再要一次精确框；失败则退回外扩后的原框。"""
    loose = pad_box(fig["box_1000"], REFINE_PAD)
    crop_page_region(page, loose, tmp_path, dpi=dpi)
    encoded = encode_image(tmp_path, max_side=max_side, prefer_jpeg=True)
    started = time.perf_counter()
    raw, usage = call_model(
        encoded["data_url"],
        build_refine_prompt(),
        model=model,
        api_key=api_key,
        max_retries=max(1, min(int(max_retries), 2)),
        thinking=thinking,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    elapsed = round(time.perf_counter() - started, 2)
    parsed, err = extract_json(raw)
    info = {"usage": usage, "elapsed_s": elapsed, "error": err or "", "raw_ok": parsed is not None}
    if parsed is None:
        fig["box_1000"] = pad_box(fig["box_1000"], pad)
        fig["refine"] = {**info, "applied": False, "reason": err}
        return fig
    if parsed.get("is_xrd") is False:
        fig["rejected_by_refine"] = True
        fig["refine"] = {**info, "applied": True, "reason": parsed.get("reason") or "二次校正判定不是 XRD"}
        return fig
    refined = clamp_box(parsed.get("box_1000") or [])
    if not refined:
        fig["box_1000"] = pad_box(fig["box_1000"], pad)
        fig["refine"] = {**info, "applied": False, "reason": "二次校正未给出有效框"}
        return fig
    mapped = clamp_box(map_box_from_crop(loose, refined))
    if not mapped or box_area_frac(mapped) < MIN_BOX_FRAC:
        fig["box_1000"] = pad_box(fig["box_1000"], pad)
        fig["refine"] = {**info, "applied": False, "reason": "二次校正框与原框偏差过大或无效，已回退"}
        return fig
    original = list(fig["box_1000"])
    # 校正框若大幅收缩/偏移，或把堆叠谱的上下切掉，回退到定位框。
    too_short = (mapped[3] - mapped[1]) < max(100.0, 0.6 * (original[3] - original[1]))
    if iou(mapped, original) < 0.45 or box_area_frac(mapped) < 0.55 * box_area_frac(original) or too_short:
        fig["box_1000"] = pad_box(original, pad)
        fig["refine"] = {
            **info,
            "applied": False,
            "reason": "二次校正框与定位框偏差过大，已回退",
            "rejected_box_1000": mapped,
        }
        return fig
    fig["box_1000"] = pad_box(mapped, pad)
    fig["refine"] = {**info, "applied": True, "reason": parsed.get("reason")}
    return fig


def detect_path_for(out_dir: Path, pdf_path: Path) -> Path:
    return out_dir / "detect" / f"{safe_stem(pdf_path.stem)}.json"


def build_figure_context(
    fig: dict[str, Any],
    pdf_name: str,
    page_no: int,
    user_context: str | None,
    page_extra: str,
) -> str:
    parts = [
        f"来源文献：{pdf_name} 第 {page_no} 页",
    ]
    if fig.get("figure_label"):
        parts.append(f"图号：{fig['figure_label']}")
    if fig.get("caption"):
        parts.append(f"图注：{fig['caption']}")
    if fig.get("notes"):
        parts.append(f"定位模型看到的信息：{fig['notes']}")
    if fig.get("description"):
        parts.append(f"画面描述：{fig['description']}")
    if page_extra:
        parts.append(f"同页可能相关的实验条件：{page_extra}")
    if user_context and user_context.strip():
        parts.append("用户补充背景：")
        parts.append(user_context.strip())
    return "\n".join(parts)


def process_pdf(
    pdf_path: Path,
    *,
    out_dir: Path,
    pages_spec: str | None,
    model: str,
    api_key: str | None,
    dpi: int,
    pad: float,
    refine: bool,
    max_side: int,
    max_retries: int,
    overwrite: bool,
    dry_run: bool,
    thinking: bool,
    workers: int,
    user_context: str | None,
    locate_model: str | None = None,
    locate_max_side: int = DEFAULT_LOCATE_MAX_SIDE,
    locate_workers: int = 0,
    call_timeout: float = DEFAULT_CALL_TIMEOUT,
    hedge_after: float | None = DEFAULT_HEDGE_AFTER,
    caption_priority: bool = True,
    on_figures: Any = None,
) -> dict[str, Any]:
    """处理一篇 PDF：筛页、定位、裁图。返回检测摘要（含 figures 列表）。

    所有候选页并行定位（不受 --workers=4 限制）。每页裁出图后立即回调 on_figures，
    便于下游边定位边数字化/识图。
    """
    locate_model = locate_model or model
    detect_file = detect_path_for(out_dir, pdf_path)
    crops_dir = out_dir / "crops"
    pages_dir = out_dir / "pages"
    overlays_dir = out_dir / "overlays"
    tmp_dir = out_dir / "_tmp"
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    doc = pymupdf.open(pdf_path)
    try:
        page_count = doc.page_count
        pages = parse_pages(pages_spec, page_count)
        radiation_hint = extract_radiation_hint(doc)
        if detect_file.exists() and not overwrite and not dry_run:
            try:
                cached = json.loads(detect_file.read_text(encoding="utf-8"))
                crops_ok = True
                for fig in cached.get("figures") or []:
                    cp = fig.get("crop_path")
                    if not cp or not Path(cp).exists():
                        crops_ok = False
                        break
                if crops_ok and cached.get("figures") is not None and cached.get("status") == "ok":
                    print(f"[跳过定位] {pdf_path.name}（已有 {detect_file.name}，需要重跑请加 --overwrite）")
                    figs = list(cached.get("figures") or [])
                    for fig in figs:
                        extra = fig.get("page_extra") or ""
                        if radiation_hint and radiation_hint not in extra:
                            extra = (extra + " " + radiation_hint).strip()
                            fig["page_extra"] = extra
                        fig["context"] = build_figure_context(
                            fig, pdf_path.name, fig.get("page") or 0, user_context, extra
                        )
                    if on_figures and figs:
                        on_figures(figs)
                    return cached
            except Exception:
                pass

        scan_rows = []
        candidates: list[int] = []
        text_chars = 0
        for pno in pages:
            text_chars += len((doc[pno - 1].get_text("text") or "").strip())
            if text_chars >= 80:
                break
        textless_fallback = text_chars < 80
        if textless_fallback:
            print(f"  [提示] {pdf_path.name} 几乎没有文本层（「打印成 PDF」常见），改把含嵌入图的页送给视觉模型筛查。")
        for pno in pages:
            page = doc[pno - 1]
            text = page.get_text("text")
            caps = extract_captions(page)
            ok, reason, kind = is_candidate_page(page, text, caps, textless_fallback=textless_fallback)
            scan_rows.append(
                {
                    "page": pno,
                    "candidate": ok,
                    "reason": reason,
                    "kind": kind,
                    "n_captions": len(caps),
                    "n_images": len(page.get_images(full=True)),
                    "caption_preview": [c["text"][:180] for c in caps],
                }
            )
            mark = "候选" if ok else "跳过"
            print(f"  [{mark}] {pdf_path.name} 第 {pno}/{page_count} 页  {reason}")
            if ok:
                candidates.append(pno)

        # 有图注命中时，正文里仅提到 XRD 的页不再送视觉模型（无文本层回退除外）
        if caption_priority and not textless_fallback:
            cap_pages = [r["page"] for r in scan_rows if r.get("kind") == "caption"]
            if cap_pages:
                kept: list[int] = []
                for row in scan_rows:
                    if row["page"] in candidates and row.get("kind") == "body":
                        row["candidate"] = False
                        row["deferred"] = True
                        row["reason"] = (row.get("reason") or "") + "（已有图注命中页，正文提及不再送检）"
                        print(
                            f"  [跳过] {pdf_path.name} 第 {row['page']}/{page_count} 页  "
                            f"已有图注命中（{cap_pages}），正文提及不再送检"
                        )
                    elif row.get("candidate"):
                        kept.append(row["page"])
                candidates = kept

        detect: dict[str, Any] = {
            "pdf": pdf_path.name,
            "pdf_path": str(pdf_path),
            "page_count": page_count,
            "pages_requested": pages,
            "candidate_pages": candidates,
            "scan": scan_rows,
            "figures": [],
            "page_results": [],
            "usage": usage_total,
            "status": "dry-run" if dry_run else "ok",
            "model": locate_model,
            "locate_max_side": locate_max_side,
        }
    finally:
        doc.close()

    def finish_page(pno: int, loc: dict[str, Any]) -> list[dict[str, Any]]:
        """在定位结果上做二次校正、裁图；每页独立打开 PDF，避免共享 Document 的线程问题。"""
        figs = loc.get("figures") or []
        if loc.get("status") != "ok" or not figs or dry_run:
            return []
        local = pymupdf.open(pdf_path)
        try:
            page = local[pno - 1]
            caps = loc.get("captions") or extract_captions(page)
            page_extra = nearby_context(loc.get("page_text") or page.get_text("text"))
            refined_figs: list[dict[str, Any]] = []
            for i, fig in enumerate(figs, start=1):
                fig = dict(fig)
                fig["page"] = pno
                fig["source_pdf"] = pdf_path.name
                if refine and not dry_run and api_key:
                    tmp = tmp_dir / f"{safe_stem(pdf_path.stem)}_p{pno}_{i}_refine.png"
                    try:
                        fig = refine_one(
                            page,
                            fig,
                            tmp,
                            model=resolve_vision_model(locate_model),
                            api_key=api_key,
                            dpi=dpi,
                            pad=pad,
                            max_side=min(int(max_side), int(locate_max_side) or max_side),
                            max_retries=max_retries,
                            thinking=thinking,
                            timeout=call_timeout,
                        )
                        add_usage(usage_total, (fig.get("refine") or {}).get("usage"))
                    except Exception as exc:
                        fig["box_1000"] = pad_box(fig["box_1000"], pad)
                        fig["refine"] = {
                            "applied": False,
                            "reason": redact_secret(f"二次校正失败，已按 {pad:.0%} 外扩：{exc}", api_key),
                        }
                else:
                    fig["box_1000"] = pad_box(fig["box_1000"], pad)

                if fig.get("rejected_by_refine"):
                    reason = (fig.get("refine") or {}).get("reason") or "二次校正判定不是 XRD"
                    print(f"    - 放弃一幅：{reason}")
                    continue

                cap = caption_for_box(fig["box_1000"], caps, page.rect)
                if cap:
                    fig["caption"] = cap["text"]
                    if not fig.get("figure_label"):
                        fig["figure_label"] = f"Figure {cap['num']}"
                    fig["box_1000"] = extend_box_to_caption(fig["box_1000"], cap, page.rect)
                elif not fig.get("caption"):
                    fig["caption"] = ""

                crop_name = f"{safe_stem(pdf_path.stem)}_p{pno}_{i:02d}.png"
                crop_path = crops_dir / crop_name
                try:
                    crop_page_region(page, fig["box_1000"], crop_path, dpi=dpi)
                except Exception as exc:
                    print(f"    [裁剪失败] {crop_name}：{exc}", file=sys.stderr)
                    continue
                fig["crop_path"] = str(crop_path.resolve())
                fig["crop_name"] = crop_name
                extra = page_extra
                if radiation_hint and radiation_hint not in (extra or ""):
                    extra = (extra + " " + radiation_hint).strip()
                fig["page_extra"] = extra
                fig["context"] = build_figure_context(
                    fig, pdf_path.name, pno, user_context, extra
                )
                print(
                    f"    - 保存 {crop_name}  框 {[int(round(v)) for v in fig['box_1000']]}  "
                    f"{fig.get('figure_label') or ''}"
                )
                refined_figs.append(fig)

            if refined_figs and loc.get("page_png"):
                try:
                    draw_overlay(
                        Path(loc["page_png"]),
                        refined_figs,
                        overlays_dir / f"{safe_stem(pdf_path.stem)}_p{pno}_boxes.png",
                    )
                except Exception:
                    pass
            return refined_figs
        finally:
            local.close()

    def handle_one_page(pno: int) -> dict[str, Any]:
        local = pymupdf.open(pdf_path)
        try:
            page = local[pno - 1]
            caps = extract_captions(page)
            page_png = pages_dir / f"{safe_stem(pdf_path.stem)}_p{pno}.png"
            render_page_png(page, page_png, max_side=locate_max_side)
            prompt = build_locate_prompt(pno, caps, page.rect)
            page_text = page.get_text("text")
        finally:
            local.close()
        try:
            if (locate_model or "auto").strip().lower() in {"auto", "compare"}:
                loc = locate_page_auto(
                    page_png,
                    prompt,
                    api_key=api_key or "",
                    max_side=locate_max_side,
                    max_retries=max_retries,
                    thinking=thinking,
                    dry_run=dry_run,
                    timeout=call_timeout,
                    max_tokens=DEFAULT_LOCATE_MAX_TOKENS,
                )
            else:
                loc = locate_page(
                    page_png,
                    prompt,
                    model=locate_model,
                    api_key=api_key or "",
                    max_side=locate_max_side,
                    max_retries=max_retries,
                    thinking=thinking,
                    dry_run=dry_run,
                    timeout=call_timeout,
                    max_tokens=DEFAULT_LOCATE_MAX_TOKENS,
                    hedge_after=hedge_after,
                    prefer_jpeg=True,
                )
        except Exception as exc:
            return {
                "page": pno,
                "status": "failed",
                "error": redact_secret(str(exc), api_key),
                "figures": [],
                "usage": None,
                "page_png": str(page_png),
                "captions": caps,
                "page_text": page_text,
            }
        loc["page"] = pno
        loc["page_png"] = str(page_png)
        loc["captions"] = caps
        loc["page_text"] = page_text
        loc["cropped_figures"] = finish_page(pno, loc)
        return loc

    page_results: list[dict[str, Any]] = []
    all_figs: list[dict[str, Any]] = []
    n_workers = 1
    if candidates:
        n_workers = 1 if dry_run else max(1, locate_workers or len(candidates))
        n_workers = min(n_workers, len(candidates))
        print(f"  [定位] {pdf_path.name} 并行 {len(candidates)} 页（并发 {n_workers}，模型 {locate_model}，长边 {locate_max_side}）")
        if n_workers == 1:
            ordered = [handle_one_page(pno) for pno in candidates]
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futs = {ex.submit(handle_one_page, pno): pno for pno in candidates}
                tmp_map: dict[int, dict[str, Any]] = {}
                for fut in as_completed(futs):
                    pno = futs[fut]
                    try:
                        loc = fut.result()
                    except Exception as exc:
                        loc = {
                            "page": pno,
                            "status": "failed",
                            "error": redact_secret(str(exc), api_key),
                            "figures": [],
                            "cropped_figures": [],
                        }
                    tmp_map[pno] = loc
                    # 先完成的页立刻把裁图交给下游，不必等其它页定位结束
                    cropped = loc.get("cropped_figures") or []
                    if cropped and on_figures:
                        try:
                            on_figures(cropped)
                        except Exception as exc:
                            print(f"    [下游回调失败] 第 {pno} 页：{redact_secret(exc, api_key)}", file=sys.stderr)
                ordered = [tmp_map[p] for p in candidates]
        page_results = ordered

    compact_page_results = []
    for loc in page_results:
        pno = loc["page"]
        add_usage(usage_total, loc.get("usage"))
        compact_page_results.append(
            {
                "page": pno,
                "status": loc.get("status"),
                "error": loc.get("error", ""),
                "elapsed_s": loc.get("elapsed_s"),
                "usage": loc.get("usage"),
                "page_has_xrd": loc.get("page_has_xrd"),
                "n_figures": len(loc.get("cropped_figures") or loc.get("figures") or []),
                "rejected": loc.get("rejected"),
                "page_png": loc.get("page_png"),
                "send_size": loc.get("send_size"),
                "nbytes": loc.get("nbytes"),
            }
        )
        if loc.get("status") == "dry-run":
            print(f"  [演练] {pdf_path.name} 第 {pno} 页已渲染，未调用定位接口")
            continue
        if loc.get("status") != "ok":
            print(
                f"  [定位失败] {pdf_path.name} 第 {pno} 页：{loc.get('error')}",
                file=sys.stderr,
            )
            continue
        cropped = loc.get("cropped_figures") or []
        raw_figs = loc.get("figures") or []
        if not raw_figs:
            print(f"  [无 XRD] {pdf_path.name} 第 {pno} 页（模型认为本页没有 XRD 图）")
            continue
        print(
            f"  [定位] {pdf_path.name} 第 {pno} 页 找到 {len(raw_figs)} 幅 XRD 图  "
            f"{format_usage(loc.get('usage'))}  用时 {loc.get('elapsed_s')} s"
        )
        all_figs.extend(cropped)
        # 单线程路径（或回调尚未在 as_completed 里触发）再补一次
        if cropped and on_figures and (n_workers == 1 or dry_run):
            on_figures(cropped)

    detect["figures"] = all_figs
    detect["page_results"] = compact_page_results
    detect["usage"] = usage_total
    detect_file.parent.mkdir(parents=True, exist_ok=True)
    safe = json.loads(json.dumps(detect, ensure_ascii=False, default=str))
    detect_file.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
    return detect


def attach_pdf_fields(record: dict[str, Any], fig: dict[str, Any]) -> dict[str, Any]:
    record["source_pdf"] = fig.get("source_pdf") or ""
    record["page"] = fig.get("page") or ""
    record["figure_label"] = fig.get("figure_label") or ""
    record["caption"] = fig.get("caption") or ""
    record["crop_path"] = fig.get("crop_path") or ""
    dig = fig.get("digitize") or {}
    record["xy_files"] = dig.get("xy_files") or fig.get("xy_files") or []
    record["replot_path"] = dig.get("replot") or ""
    record["validate_path"] = dig.get("validate") or ""
    record["calib_summary"] = (fig.get("calib") or {}).get("summary") or ""
    review = fig.get("review") or {}
    if review:
        findings = "；".join(review.get("findings") or [])
        record["review_verdict"] = review.get("verdict") or ""
        record["review_findings"] = findings
    if fig.get("fit_summary"):
        record["fit_summary"] = fig["fit_summary"]
    return record


def build_peak_table(xy_files: list[str]) -> str:
    """对数字化得到的 .xy 调用 xy_peaks，拼成交给识图模型的峰表。"""
    try:
        from xy_peaks import find_peaks_xy, format_peak_table, load_xy
    except ImportError as exc:
        return f"（无法导入 xy_peaks.py：{exc}）"
    parts: list[str] = []
    for raw in xy_files:
        path = Path(raw)
        try:
            x, y = load_xy(path)
            peaks = find_peaks_xy(x, y)
            parts.append(format_peak_table(peaks, title=f"数字化曲线：{path.name}"))
        except Exception as exc:
            parts.append(f"（寻峰失败：{path.name}，原因：{exc}）")
    if not parts:
        return "（没有可用的数字化 .xy）"
    return (
        "以下峰位来自对文献插图数字化后的 2θ–强度曲线（不是仪器原始数据，精度受裁图、刻度定标和描线质量限制，"
        "但仍通常优于纯读图估峰）。请优先采用本表的峰位与相对强度，再结合图例、hkl 与卡片竖线解释。\n\n"
        + "\n\n".join(parts)
    )



def recover_cached_digitize(fig: dict[str, Any], out_dir: Path) -> bool:
    """若磁盘上已有定标和 .xy，则填回 fig，避免重复调用模型。"""
    crop = Path(fig.get("crop_path") or "")
    if not crop.name:
        return False
    run_dir = Path(out_dir) / "digitize" / crop.stem
    calib = fig.get("calib") if isinstance(fig.get("calib"), dict) and (fig.get("calib") or {}).get("ticks") else None
    if calib is None and (run_dir / "calib.json").is_file():
        try:
            payload = json.loads((run_dir / "calib.json").read_text(encoding="utf-8"))
            calib = dict(payload.get("calib") or {})
            if payload.get("ticks"):
                calib["ticks"] = payload["ticks"]
            if payload.get("fit"):
                calib["fit"] = payload["fit"]
        except Exception:
            calib = None
    index_path = Path(out_dir) / "figures_index.json"
    if (calib is None or not calib.get("wavelength_angstrom")) and index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for item in index.get("figures") or []:
                if item.get("crop_name") == crop.name or Path(item.get("crop_path") or "").name == crop.name:
                    cached_cal = item.get("calib") if isinstance(item.get("calib"), dict) else None
                    if cached_cal:
                        calib = {**(cached_cal or {}), **(calib or {})}
                    break
        except Exception:
            pass
    xy_files: list[str] = []
    xy_root = Path(out_dir) / "xy"
    if xy_root.is_dir():
        xy_files = [str(p) for p in sorted(xy_root.glob(crop.stem + "_*.xy"))]
    if not xy_files and (run_dir / "traces_xy").is_dir():
        try:
            from digitize_crop import _list_trace_xy
            xy_files = [str(p) for p in _list_trace_xy(run_dir / "traces_xy")]
        except Exception:
            xy_files = [str(p) for p in sorted((run_dir / "traces_xy").glob("*.xy")) if "_grid" not in p.stem]
    if not xy_files:
        return False
    if calib:
        fig["calib"] = calib
    replot = run_dir / "replot.png"
    overlay = run_dir / "overlay_check.png"
    validate = run_dir / "validate.png"
    fig["digitize"] = {
        "status": "cached",
        "method": "cached",
        "run_dir": str(run_dir),
        "xy_files": xy_files,
        "replot": str(replot) if replot.is_file() else "",
        "validate": str(overlay if overlay.is_file() else validate) if (overlay.is_file() or validate.is_file()) else "",
        "n_traces": len(xy_files),
        "error": "",
        "trace_names": [Path(p).stem for p in xy_files],
    }
    return True


def resolve_radiation(fig: dict[str, Any]) -> tuple[float, str]:
    """综合定标、图注和全文辐射线索，尽量不要把钴靶图误判成铜靶。"""
    calib = fig.get("calib") or {}
    text = " ".join(
        [
            str(calib.get("wavelength_note") or ""),
            str(calib.get("radiation") or ""),
            str(fig.get("caption") or ""),
            str(fig.get("context") or ""),
            str(fig.get("page_extra") or ""),
            str(fig.get("source_pdf") or ""),
        ]
    )
    key = text.lower().replace("α", "a").replace("ɑ", "a")
    table = {"Cu": 1.5406, "Co": 1.78897, "Mo": 0.71073, "Fe": 1.9373, "Cr": 2.2909}
    for name, wl in table.items():
        n = name.lower()
        if any(s in key for s in (f"{n} k", f"{n}kα", f"{n}ka", f"{n}靶", f"{n}-kα", f"{n}-ka")):
            return wl, name
    if "钴靶" in text:
        return 1.78897, "Co"
    try:
        wl = float(calib.get("wavelength_angstrom") or 0)
    except (TypeError, ValueError):
        wl = 0.0
    if wl:
        best = min(table, key=lambda k: abs(table[k] - wl))
        if abs(table[best] - wl) < 0.05:
            return table[best], best
        return wl, str(calib.get("wavelength_note") or calib.get("radiation") or "")
    return 1.5406, "Cu"


def pick_xy_for_fit(xy_files: list[str]) -> str | None:
    """优先拟合更可能含杂相或主实验曲线的那一条（如 800 °C、5 K），否则第一条。"""
    if not xy_files:
        return None
    scored = []
    for path in xy_files:
        name = Path(path).stem.lower()
        score = 0
        if "800" in name:
            score += 30
        if re.search(r"(^|_)5(_|$)|5_k|5k", name):
            score += 10
        if "曲线1" in Path(path).stem or name.endswith("_00"):
            score += 1
        scored.append((score, path))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1]


def enrich_figure_digitize(
    fig: dict[str, Any],
    *,
    out_dir: Path,
    model: str,
    api_key: str | None,
    max_side: int,
    max_retries: int,
    thinking: bool,
    overwrite: bool,
    dry_run: bool,
    do_digitize: bool,
    do_review: bool,
    usage_acc: dict[str, int],
    timeout: float = DEFAULT_CALL_TIMEOUT,
    max_tokens: int = 2048,
    review_timeout: float = 15.0,
) -> None:
    """读刻度、数字化、可选叠加复核，结果写回 fig。"""
    from calib_read import read_calibration, review_overlay
    from digitize_crop import digitize_figure

    crop = Path(fig["crop_path"])
    context = fig.get("context") or ""
    if not do_digitize or dry_run or api_key is None:
        return
    if not overwrite and recover_cached_digitize(fig, out_dir):
        print(f"  [定标/描线] {crop.name} 使用已有结果，跳过模型调用（{len((fig.get('digitize') or {}).get('xy_files') or [])} 条 .xy）")
        calib = fig.get("calib") or {}
        extra = []
        if calib.get("summary"):
            extra.append("坐标定标：" + str(calib["summary"]))
        if calib.get("wavelength_note"):
            extra.append("辐射源：" + str(calib["wavelength_note"]))
        if extra:
            fig["context"] = (context + "\n" + "\n".join(extra)).strip()
        return
    reused = isinstance(fig.get("calib"), dict) and fig["calib"].get("status") == "ok" and fig["calib"].get("ticks")
    if reused:
        calib = fig["calib"]
        print(f"  [定标] {crop.name} 使用已读取的刻度，跳过重复调用")
    else:
        print(f"  [定标] {crop.name} ...")
        calib = read_calibration(
            crop,
            model=model,
            api_key=api_key,
            context=context,
            max_side=max_side,
            max_retries=max(1, min(int(max_retries), 2)),
            thinking=thinking,
            dry_run=False,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        add_usage(usage_acc, calib.get("usage"))
        print(
            f"    {calib.get('summary')}  {format_usage(calib.get('usage'))}"
            if calib.get("status") == "ok"
            else f"    定标未完成：{calib.get('error') or calib.get('status')}"
        )
    if fig.get("_skip_masks") and isinstance(calib, dict):
        calib["mask_boxes_1000"] = []
    fig["calib"] = calib

    if calib.get("status") == "not_2theta":
        fig["skip_reason"] = "not_2theta"
        fig["digitize"] = {"status": "skipped", "error": calib.get("error") or "非实验室 2θ XRD", "n_traces": 0, "xy_files": []}
        print(f"    跳过数字化：{calib.get('error')}")
    elif calib.get("status") == "ok":
        dig = digitize_figure(crop, calib, out_dir, overwrite=overwrite)
        fig["digitize"] = dig
        if dig.get("status") in {"ok", "skipped"}:
            print(
                f"    描线 {dig.get('method')}：{dig.get('n_traces')} 条 "
                f"{'、'.join(dig.get('trace_names') or [])}  残差 {dig.get('max_residual_deg')}"
            )
            if do_review and dig.get("validate") and Path(dig["validate"]).is_file():
                try:
                    review = review_overlay(
                        crop,
                        Path(dig["validate"]),
                        model=model,
                        api_key=api_key,
                        expected_curves=calib.get("curve_count") or dig.get("n_traces"),
                        max_side=min(int(max_side), 1280),
                        max_retries=max(1, min(int(max_retries), 2)),
                        thinking=thinking,
                        timeout=review_timeout,
                        max_tokens=512,
                    )
                    add_usage(usage_acc, review.get("usage"))
                    from calib_read import apply_zero_trace_review
                    review = apply_zero_trace_review(review, dig.get("xy_files") or [])
                    fig["review"] = review
                    findings = "；".join(review.get("findings") or [])
                    print(f"    复核 {review.get('verdict')}  {findings}  {format_usage(review.get('usage'))}")
                except Exception as exc:
                    fig["review"] = {"verdict": "uncertain", "error": redact_secret(str(exc), api_key)}
                    print(f"    复核失败：{fig['review']['error']}", file=sys.stderr)
        else:
            print(f"    描线失败：{dig.get('error')}", file=sys.stderr)
    # 把定标信息补进识图背景
    extra = []
    if calib.get("summary"):
        extra.append("坐标定标：" + calib["summary"])
    if calib.get("wavelength_note"):
        extra.append("辐射源：" + calib["wavelength_note"])
    if fig.get("digitize", {}).get("xy_files"):
        extra.append("已从裁图数字化曲线，峰位请优先用附件峰表，不要只用读图估计。")
    if extra:
        fig["context"] = (context + "\n" + "\n".join(extra)).strip()


def write_detect_markdown(detects: list[dict[str, Any]], usage_all: dict[str, int], model: str) -> str:
    lines = [
        "## PDF 找图与裁剪",
        "",
        f"- 定位+校正合计 {format_usage(usage_all)}；费用 {estimate_cost_cny(model, usage_all)}",
        "",
    ]
    for d in detects:
        lines.append(f"### {d.get('pdf', '')}")
        lines.append("")
        lines.append(
            f"- 页数：{d.get('page_count')}；候选页：{d.get('candidate_pages') or '无'}；"
            f"裁出 XRD 图 {len(d.get('figures') or [])} 幅"
        )
        if d.get("usage"):
            lines.append(f"- 本 PDF 定位/校正：{format_usage(d.get('usage'))}")
        lines.append("")
        figs = d.get("figures") or []
        if not figs:
            lines.append("未裁出 XRD 图。")
            lines.append("")
            continue
        lines.append("| 裁图 | 页码 | 图号 | 图注 |")
        lines.append("| --- | --- | --- | --- |")
        for fig in figs:
            cap = (fig.get("caption") or "").replace("|", "\\|")
            if len(cap) > 160:
                cap = cap[:157] + "…"
            lines.append(
                f"| {fig.get('crop_name') or ''} | {fig.get('page')} | "
                f"{(fig.get('figure_label') or '').replace('|', '\\|')} | {cap} |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从文献 PDF 找出 XRD 图并裁剪，再调用百炼视觉模型做版式/峰位/物相/合成路线识图。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例（PowerShell）：\n"
            "  python pdf_xrd.py .\\examples\\papers --dry-run\n"
            "  python pdf_xrd.py .\\paper.pdf --context \"Cu Kα，样品为铁氧化物\"\n"
            "  python pdf_xrd.py .\\papers --pages 5 --model qwen3.8-flash\n"
            "  python pdf_xrd.py .\\paper.pdf --pages 4-8,12 --workers 4\n"
            "  python pdf_xrd.py .\\paper.pdf --locate-model qwen3.6-flash --fit\n"
        ),
    )
    parser.add_argument("input", help="一个 PDF 文件，或只含 PDF 的文件夹（不递归）")
    parser.add_argument("--out-dir", default="results_pdf", help="输出目录（默认 results_pdf）")
    parser.add_argument("--pages", default=None, help="限定页码，例如 5 或 4-6,8（1 开始）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"识图等步骤的默认模型（默认 {DEFAULT_MODEL}）")
    parser.add_argument(
        "--locate-model",
        default=DEFAULT_LOCATE_MODEL,
        help="定位所用模型。默认 deepseek-v4.1-flash-1017。auto：与 qwen3.6-flash 并行，按坐标框质量择优",
    )
    parser.add_argument(
        "--phase-model",
        default=DEFAULT_PHASE_MODEL,
        help="物相解释模型（默认 qwen3.5-omni-flash；可用 qwen3.8-flash / qwen3.6-flash。精确 2θ 以描点为准）",
    )
    parser.add_argument("--context", default=None, help="附加样品/辐射源等背景，会与图注一起送给识图步骤")
    parser.add_argument("--tasks", default="all", help="识图任务，默认全部（layout,peaks,phase,synthesis）")
    parser.add_argument("--workers", type=int, default=4, help="识图/数字化并发数（默认 4；定位页不受此上限限制）")
    parser.add_argument(
        "--locate-workers",
        type=int,
        default=0,
        help="定位并发数，0 表示所有候选页同时发出（默认 0）",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="裁图 DPI（默认 200；定位页另按 --locate-max-side 缩小）")
    parser.add_argument(
        "--locate-max-side",
        type=int,
        default=DEFAULT_LOCATE_MAX_SIDE,
        help=f"定位用整页图像的长边像素（默认 {DEFAULT_LOCATE_MAX_SIDE}）",
    )
    parser.add_argument(
        "--call-timeout",
        type=float,
        default=DEFAULT_CALL_TIMEOUT,
        help=f"单次接口超时秒数，超时后重试一次（默认 {DEFAULT_CALL_TIMEOUT:.0f}）",
    )
    parser.add_argument(
        "--hedge-after",
        type=float,
        default=DEFAULT_HEDGE_AFTER,
        help=f"定位请求超过该秒数仍未返回则再发一趟，先到先用（默认 {DEFAULT_HEDGE_AFTER:.0f}；0 关闭）",
    )
    parser.add_argument("--no-hedge", action="store_true", help="关闭定位的对冲请求")
    parser.add_argument(
        "--all-keyword-pages",
        action="store_true",
        help="图注已命中时仍把仅正文提到 XRD 的页送给定位模型（默认会跳过这类页）",
    )
    parser.add_argument("--pad", type=float, default=DEFAULT_PAD, help="裁框外扩比例（默认 0.03，即 3%%）")
    parser.add_argument("--no-refine", action="store_true", help="关闭二次校正（仍会按 --pad 外扩）")
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE, help="上传图像长边像素上限（识图/裁图）")
    parser.add_argument("--max-retries", type=int, default=5, help="接口失败重试次数（定位/定标实际最多 2 次以便控制时延）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有定位结果和识图 JSON")
    parser.add_argument("--thinking", action="store_true", help="打开模型思考模式（默认关闭）")
    parser.add_argument("--dry-run", action="store_true", help="只做文本筛页和渲染，不调用接口")
    parser.add_argument("--no-digitize", action="store_true", help="跳过刻度读取与曲线数字化，只裁图+读图")
    parser.add_argument("--no-review", action="store_true", help="不对叠加校验图做视觉复核（可省一次调用）")
    parser.add_argument("--no-mask", action="store_true", help="描线前不遮盖图例/文字框")
    parser.add_argument("--fit", action="store_true", help="若已安装 DARA 环境，则对数字化曲线做全谱拟合")
    parser.add_argument("--fit-all-curves", action="store_true", help="加 --fit 时拟合一幅图里的全部曲线（默认每图只拟合一条代表性曲线）")
    parser.add_argument("--no-fit", action="store_true", help="明确跳过 DARA 拟合（默认即跳过，除非加 --fit）")
    parser.add_argument(
        "--cif-cache",
        default=None,
        help="CIF 缓存目录（默认本工具目录下的 cod_cache/，可用环境变量 XRD_COD_CACHE）",
    )
    return parser.parse_args(argv)


def _figure_timing_bucket() -> dict[str, Any]:
    return {"digitize_s": None, "phase_s": None, "crop_name": ""}


def process_one_figure(
    fig: dict[str, Any],
    *,
    out_dir: Path,
    args: argparse.Namespace,
    api_key: str | None,
    tasks: list[str],
    usage_step2: dict[str, int],
    usage_detect: dict[str, int],
) -> dict[str, Any]:
    """对已裁好的一幅图：数字化与识图并行。"""
    crop_name = fig.get("crop_name") or Path(fig.get("crop_path") or "crop").name
    timing = _figure_timing_bucket()
    timing["crop_name"] = crop_name
    if args.no_mask:
        fig["_skip_masks"] = True

    # 先读刻度（也得到 plot_box），再与描线/识图重叠；相邻子图可从数据框切出后再识图
    if not args.dry_run and not args.no_digitize and api_key is not None and not (
        isinstance(fig.get("calib"), dict) and fig["calib"].get("status") == "ok"
    ):
        from calib_read import read_calibration
        t_cal = time.perf_counter()
        print(f"  [定标] {crop_name} ...")
        try:
            calib = read_calibration(
                Path(fig["crop_path"]),
                model=resolve_vision_model(args.locate_model, CALIB_MODEL_WHEN_AUTO),
                api_key=api_key,
                context=fig.get("context"),
                max_side=args.max_side,
                max_retries=max(1, min(int(args.max_retries), 2)),
                thinking=args.thinking,
                dry_run=False,
                timeout=args.call_timeout,
                max_tokens=2048,
            )
        except Exception as exc:
            calib = {"status": "failed", "error": redact_secret(exc, api_key), "ticks": []}
            print(f"    定标调用异常：{calib['error']}", file=sys.stderr)
        add_usage(usage_step2, calib.get("usage"))
        if fig.get("_skip_masks"):
            calib["mask_boxes_1000"] = []
        fig["calib"] = calib
        timing["calib_s"] = round(time.perf_counter() - t_cal, 2)
        if calib.get("status") == "ok":
            print(f"    {calib.get('summary')}  {format_usage(calib.get('usage'))}")
        elif calib.get("status") == "not_2theta":
            fig["skip_reason"] = "not_2theta"
            print(f"    非实验室 2θ XRD，跳过：{calib.get('error')}")
        else:
            print(f"    定标未完成：{calib.get('error') or calib.get('status')}")
        box = calib.get("plot_box_1000") if isinstance(calib, dict) else None
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                x0, y0, x1, y1 = (float(v) for v in box)
                x0, x1 = sorted((x0, x1))
                y0, y1 = sorted((y0, y1))
                if (x1 - x0) >= 200 and (x1 - x0) <= 850 and (y1 - y0) >= 200:
                    from PIL import Image
                    src = Path(fig["crop_path"])
                    dest = src.with_name(src.stem + "_panel.png")
                    with Image.open(src) as im:
                        w, h = im.size
                        padw, padh = (x1 - x0) * 0.04, (y1 - y0) * 0.04
                        px = [
                            max(0, int((x0 - padw) / 1000.0 * w)),
                            max(0, int((y0 - padh) / 1000.0 * h)),
                            min(w, int((x1 + padw) / 1000.0 * w)),
                            min(h, int((y1 + padh) / 1000.0 * h)),
                        ]
                        if px[2] - px[0] >= 40 and px[3] - px[1] >= 40:
                            im.crop(tuple(px)).save(dest, format="PNG")
                            fig["phase_image"] = str(dest)
                            print(f"    按 plot_box 切出识图裁区 {dest.name}")
            except Exception:
                pass

    def do_digitize(include_review: bool) -> None:
        if args.dry_run or args.no_digitize or api_key is None:
            return
        if fig.get("skip_reason") == "not_2theta" or (fig.get("calib") or {}).get("status") == "not_2theta":
            fig["digitize"] = fig.get("digitize") or {"status": "skipped", "error": "非实验室 2θ XRD", "n_traces": 0, "xy_files": []}
            return
        t0 = time.perf_counter()
        enrich_figure_digitize(
            fig,
            out_dir=out_dir,
            model=resolve_vision_model(args.locate_model, CALIB_MODEL_WHEN_AUTO),
            api_key=api_key,
            max_side=args.max_side,
            max_retries=args.max_retries,
            thinking=args.thinking,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            do_digitize=not args.no_digitize,
            do_review=include_review,
            usage_acc=usage_step2,
            timeout=args.call_timeout,
            max_tokens=2048,
            review_timeout=min(15.0, float(args.call_timeout) or 15.0),
        )
        if args.no_mask and isinstance(fig.get("calib"), dict):
            fig["calib"]["mask_boxes_1000"] = []
        timing["digitize_s"] = round(time.perf_counter() - t0, 2)

    def do_review() -> None:
        if args.dry_run or args.no_review or api_key is None:
            return
        if fig.get("review"):
            return
        dig = fig.get("digitize") or {}
        calib = fig.get("calib") or {}
        validate = dig.get("validate")
        if not validate or not Path(validate).is_file():
            return
        t0 = time.perf_counter()
        try:
            from calib_read import review_overlay
            review = review_overlay(
                Path(fig["crop_path"]),
                Path(validate),
                model=resolve_vision_model(args.locate_model, CALIB_MODEL_WHEN_AUTO),
                api_key=api_key,
                expected_curves=calib.get("curve_count") or dig.get("n_traces"),
                max_side=min(int(args.max_side), 1280),
                max_retries=max(1, min(int(args.max_retries), 2)),
                thinking=args.thinking,
                timeout=min(15.0, float(args.call_timeout) or 15.0),
                max_tokens=512,
            )
            add_usage(usage_step2, review.get("usage"))
            from calib_read import apply_zero_trace_review
            review = apply_zero_trace_review(review, (fig.get("digitize") or {}).get("xy_files") or [])
            fig["review"] = review
            findings = "；".join(review.get("findings") or [])
            print(f"    复核 {review.get('verdict')}  {findings}  {format_usage(review.get('usage'))}")
        except Exception as exc:
            fig["review"] = {"verdict": "uncertain", "error": redact_secret(str(exc), api_key)}
            print(f"    复核失败：{fig['review']['error']}", file=sys.stderr)
        timing["review_s"] = round(time.perf_counter() - t0, 2)

    # 默认 all 时，物相调用只做 phase+synthesis：精确峰位已由描点给出，
    # 少生成 layout/peaks 可避免 3.8 超时重试把识图拖到 70 s+。用户显式 --tasks 仍照做。
    phase_tasks = list(tasks)
    if list(tasks) == list(ALL_TASKS):
        phase_tasks = ["phase", "synthesis"]

    def do_phase() -> dict[str, Any]:
        if args.dry_run:
            return attach_pdf_fields(
                {
                    "file": crop_name,
                    "stem": Path(crop_name).stem,
                    "json_path": "",
                    "status": "dry-run",
                    "model": args.phase_model or args.model,
                    "data_file": "",
                    "error": "",
                    "parsed": None,
                    "elapsed_s": None,
                    "usage": None,
                },
                fig,
            )
        t0 = time.perf_counter()
        if fig.get("skip_reason") == "not_2theta" or (fig.get("calib") or {}).get("status") == "not_2theta":
            rec = attach_pdf_fields(
                {
                    "file": crop_name,
                    "stem": Path(crop_name).stem,
                    "json_path": "",
                    "status": "skipped",
                    "model": args.phase_model or args.model,
                    "data_file": "",
                    "error": (fig.get("calib") or {}).get("error") or "非实验室 2θ XRD，已跳过识图",
                    "parsed": None,
                    "elapsed_s": 0.0,
                    "usage": None,
                },
                fig,
            )
            timing["phase_s"] = round(time.perf_counter() - t0, 2)
            return rec
        try:
            from digitize_crop import format_digitized_peak_table
            peak_table = format_digitized_peak_table(fig)
        except Exception:
            peak_table = None
        rec = process_one(
            Path(fig.get("phase_image") or fig["crop_path"]),
            out_dir=out_dir,
            model=args.phase_model or args.model,
            tasks=phase_tasks,
            context=fig.get("context"),
            data_arg=None,
            max_side=args.max_side,
            max_retries=max(1, min(int(args.max_retries), 2)),
            overwrite=args.overwrite,
            dry_run=False,
            api_key=api_key,
            thinking=args.thinking,
            peak_table=peak_table,
            timeout=max(55.0, float(args.call_timeout) or 55.0),
            max_tokens=2048,
        )
        timing["phase_s"] = round(time.perf_counter() - t0, 2)
        add_usage(usage_detect, rec.get("usage") if isinstance(rec.get("usage"), dict) else None)
        # 识图若用了按 plot_box 切出的子图，把 JSON 同步到原裁图名，便于报告对应
        if fig.get("phase_image") and Path(fig["phase_image"]).name != Path(fig["crop_path"]).name:
            srcj = out_dir / f"{Path(fig['phase_image']).name}.json"
            dstj = out_dir / f"{Path(fig['crop_path']).name}.json"
            if srcj.is_file():
                dstj.write_text(srcj.read_text(encoding="utf-8"), encoding="utf-8")
                rec["json_path"] = str(dstj)
                rec["file"] = Path(fig["crop_path"]).name
        return attach_pdf_fields(rec, fig)

    dig_exc = None
    rec: dict[str, Any] | None = None
    # 先描线得到峰表，再让物相与复核并行（复核不再挡住识图）
    try:
        do_digitize(False)
    except Exception as exc:
        dig_exc = exc
        print(f"    定标/描线异常 {crop_name}：{redact_secret(exc, api_key)}", file=sys.stderr)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_phase = pool.submit(do_phase)
            fut_rev = pool.submit(do_review) if (not args.no_review and not args.dry_run) else None
            rec = fut_phase.result()
            if fut_rev is not None:
                try:
                    fut_rev.result()
                except Exception:
                    pass
    except Exception as exc:
        rec = attach_pdf_fields(
            {
                "file": crop_name,
                "stem": Path(crop_name).stem,
                "json_path": "",
                "status": "failed",
                "model": args.phase_model or args.model,
                "data_file": "",
                "error": redact_secret(f"未捕获异常：{exc}", api_key),
                "parsed": None,
                "elapsed_s": None,
                "usage": None,
            },
            fig,
        )
    if dig_exc and not fig.get("digitize"):
        fig["digitize"] = {"status": "failed", "error": redact_secret(str(dig_exc), api_key)}
    # 数字化完成后把峰表补进记录（不必再调模型）
    xy_files = (fig.get("digitize") or {}).get("xy_files") or []
    if xy_files:
        try:
            fig["peak_table"] = build_peak_table(xy_files)
            rec["peak_table"] = fig["peak_table"]
            rec["xy_files"] = xy_files
        except Exception:
            pass
    rec["stage_s"] = timing
    rec = attach_pdf_fields(rec, fig)
    # 把识图结果写回 fig，供 figures_index 列出（物相正文在 <crop>.png.json）
    if isinstance(rec, dict):
        parsed = rec.get("parsed") or rec.get("result") or {}
        phase_obj = None
        if isinstance(parsed, dict):
            phase_obj = parsed.get("phase") if isinstance(parsed.get("phase"), dict) else parsed
        if phase_obj:
            fig["phase"] = phase_obj
        fig["phase_status"] = rec.get("status")
        fig["phase_model"] = rec.get("model") or fig.get("phase_model")
        if rec.get("error"):
            fig["phase_error"] = rec.get("error")
    print(
        f"    [计时] {crop_name} 定标 {timing.get('calib_s')} s，数字化 {timing['digitize_s']} s，"
        f"复核 {timing.get('review_s')} s，识图 {timing['phase_s']} s（峰表到位后识图与复核并行）"
    )
    return rec


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv)
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        pdfs = collect_pdfs(input_path)
        tasks = parse_tasks(args.tasks)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.dpi < 72 or args.dpi > 600:
        print("--dpi 建议在 72–600 之间。", file=sys.stderr)
        return 2
    if args.pad < 0 or args.pad > 0.3:
        print("--pad 应在 0 到 0.3 之间。", file=sys.stderr)
        return 2

    api_key = None
    if not args.dry_run:
        try:
            api_key = get_api_key()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    hedge_after = None if args.no_hedge or float(args.hedge_after or 0) <= 0 else float(args.hedge_after)
    print(
        f"共 {len(pdfs)} 个 PDF，输出目录 {out_dir.resolve()}，"
        f"定位 {args.locate_model}，识图 {args.phase_model or args.model}"
        f"{'，对冲 '+str(hedge_after)+'s' if hedge_after else '，不对冲'}"
    )
    t0 = time.perf_counter()
    # --fit 时提前探测 DARA/BGMN，与后面的定位/识图重叠，避免拟合阶段再付启动成本
    if (not args.dry_run) and args.fit and not args.no_fit:
        def _warm_dara() -> None:
            try:
                from phase_fit import probe_dara
                probe_dara()
            except Exception:
                pass
        threading.Thread(target=_warm_dara, name="dara-warm", daemon=True).start()
    detects: list[dict[str, Any]] = []
    all_figs: list[dict[str, Any]] = []
    usage_detect = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage_step2 = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    fig_lock = threading.Lock()
    fig_jobs: list[Any] = []
    records_map: dict[str, dict[str, Any]] = {}
    t_first_crop: float | None = None
    t_locate_done: float | None = None

    fig_pool = ThreadPoolExecutor(max_workers=max(1, args.workers))

    def on_figures(figs: list[dict[str, Any]]) -> None:
        nonlocal t_first_crop
        if not figs:
            return
        with fig_lock:
            if t_first_crop is None:
                t_first_crop = time.perf_counter() - t0
                print(f"  [计时] 第一幅裁图出现 @ {t_first_crop:.1f} s，开始数字化+识图")
            all_figs.extend(figs)
        if args.dry_run:
            return
        for fig in figs:
            if args.no_mask:
                fig["_skip_masks"] = True
            fut = fig_pool.submit(
                process_one_figure,
                fig,
                out_dir=out_dir,
                args=args,
                api_key=api_key,
                tasks=tasks,
                usage_step2=usage_step2,
                usage_detect=usage_detect,
            )
            fig_jobs.append((fig, fut))

    try:
        for pdf in pdfs:
            print(f"\n======== {pdf.name} ========")
            t_pdf = time.perf_counter()
            try:
                det = process_pdf(
                    pdf,
                    out_dir=out_dir,
                    pages_spec=args.pages,
                    model=resolve_vision_model(args.locate_model, CALIB_MODEL_WHEN_AUTO),
                    api_key=api_key,
                    dpi=args.dpi,
                    pad=args.pad,
                    refine=not args.no_refine,
                    max_side=args.max_side,
                    max_retries=args.max_retries,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    thinking=args.thinking,
                    workers=max(1, args.workers),
                    user_context=args.context,
                    locate_model=args.locate_model or "auto",
                    locate_max_side=args.locate_max_side,
                    locate_workers=args.locate_workers,
                    call_timeout=args.call_timeout,
                    hedge_after=hedge_after,
                    caption_priority=not args.all_keyword_pages,
                    on_figures=on_figures,
                )
            except Exception as exc:
                print(redact_secret(traceback.format_exc(), api_key), file=sys.stderr)
                detects.append(
                    {
                        "pdf": pdf.name,
                        "status": "failed",
                        "error": redact_secret(str(exc), api_key),
                        "figures": [],
                    }
                )
                continue
            det["locate_wall_s"] = round(time.perf_counter() - t_pdf, 2)
            print(f"  [计时] {pdf.name} 筛页+定位+裁图墙钟 {det['locate_wall_s']} s")
            detects.append(det)
            add_usage(usage_detect, det.get("usage"))
            # 无回调漏网的图（例如缓存路径）已在 process_pdf 里触发 on_figures
            with fig_lock:
                known = {f.get("crop_path") for f in all_figs}
            for fig in det.get("figures") or []:
                if fig.get("crop_path") not in known:
                    on_figures([fig])
        t_locate_done = time.perf_counter() - t0

        records: list[dict[str, Any]] = []
        if fig_jobs:
            for fig, fut in fig_jobs:
                key = fig.get("crop_path") or fig.get("crop_name") or str(id(fig))
                try:
                    rec = fut.result()
                except Exception as exc:
                    rec = attach_pdf_fields(
                        {
                            "file": fig.get("crop_name") or "",
                            "stem": Path(fig.get("crop_name") or "crop").stem,
                            "json_path": "",
                            "status": "failed",
                            "model": args.phase_model or args.model,
                            "data_file": "",
                            "error": redact_secret(f"未捕获异常：{exc}", api_key),
                            "parsed": None,
                            "elapsed_s": None,
                            "usage": None,
                        },
                        fig,
                    )
                records_map[key] = rec
            # 按 all_figs 顺序输出
            with fig_lock:
                order = list(all_figs)
            records = [records_map.get(f.get("crop_path") or f.get("crop_name") or "", {}) for f in order]
            records = [r for r in records if r]
        elif args.dry_run:
            with fig_lock:
                for fig in all_figs:
                    records.append(
                        attach_pdf_fields(
                            {
                                "file": fig.get("crop_name") or "",
                                "stem": Path(fig.get("crop_name") or "crop").stem,
                                "status": "dry-run",
                                "model": args.phase_model or args.model,
                            },
                            fig,
                        )
                    )
        t_figures_done = time.perf_counter() - t0
    finally:
        fig_pool.shutdown(wait=True)

    # 第 3 步（可选）：DARA 拟合。多图/多曲线并行，CIF 走全局缓存；无 CIF 或已发表精修图则跳过。
    fit_notes: list[str] = []
    t_fit_s = 0.0
    if all_figs and (not args.dry_run) and args.fit and not args.no_fit:
        from phase_fit import (
            default_cif_cache,
            fetch_cifs,
            fit_xy,
            formulas_from_candidates,
            interpret_prompt,
            is_published_rietveld,
            probe_dara,
        )

        probe = probe_dara()
        if not probe.get("available"):
            fit_notes.append("DARA 环境不可用，已跳过拟合。")
            fit_notes.append((probe.get("error") or "").strip())
            print("\n[第 3 步] DARA 不可用，已跳过。安装方法见《使用说明.md》和 phase_fit.py 顶部说明。")
        else:
            print(f"\n[第 3 步] 使用 {probe.get('python')} 中的 dara {probe.get('version')} 并行拟合")
            if args.cif_cache:
                cache = Path(args.cif_cache).expanduser().resolve()
            else:
                cache = default_cif_cache().resolve()
            print(f"    CIF 缓存：{cache}")
            rec_by_crop = {r.get("crop_path"): r for r in records if r.get("crop_path")}

            def run_fit_job(fig: dict[str, Any], xy_pick: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
                rec = rec_by_crop.get(fig.get("crop_path") or "")
                parsed = rec.get("parsed") if isinstance(rec, dict) else None
                cands = []
                if isinstance(parsed, dict):
                    phase = parsed.get("phase") or {}
                    cands = phase.get("candidates") or []
                items = formulas_from_candidates(cands, fig.get("caption") or "")
                t_cif = time.perf_counter()
                refs, err = fetch_cifs(items, cache)
                cif_s = round(time.perf_counter() - t_cif, 2)
                if err:
                    print(f"    COD {Path(xy_pick).name}：", "；".join(err[:4]))
                if not refs:
                    return fig, xy_pick, {
                        "status": "skipped",
                        "summary": "未下载到 CIF，跳过拟合",
                        "message": "；".join(err[:4]) if err else "无 CIF",
                        "cif_s": cif_s,
                    }
                fit_dir = out_dir / "fits" / Path(xy_pick).stem
                wl, rad = resolve_radiation(fig)
                print(f"    拟合 {Path(xy_pick).name}，辐射 {rad} {wl} Å，CIF {len(refs)} 个")
                t_fit = time.perf_counter()
                fit_res = fit_xy(
                    Path(xy_pick),
                    refs,
                    fit_dir,
                    wavelength_a=wl,
                    radiation=rad,
                    probe=probe,
                    timeout=60,
                )
                fit_res["cif_s"] = cif_s
                fit_res["dara_s"] = round(time.perf_counter() - t_fit, 2)
                return fig, xy_pick, fit_res

            jobs: list[tuple[dict[str, Any], str]] = []
            for fig in all_figs:
                rec = rec_by_crop.get(fig.get("crop_path") or "")
                if is_published_rietveld(
                    fig.get("caption"),
                    fig.get("figure_label"),
                    fig.get("description"),
                    fig.get("notes"),
                    (rec or {}).get("review_findings"),
                ):
                    msg = f"{fig.get('crop_name')}: 识别为已发表的 Rietveld 精修图，跳过拟合"
                    print("    " + msg)
                    fig["fit_summary"] = "已发表 Rietveld 精修图，已跳过"
                    fit_notes.append(msg)
                    if rec:
                        rec["fit_summary"] = fig["fit_summary"]
                    continue
                if not (fig.get("digitize") or {}).get("xy_files"):
                    recover_cached_digitize(fig, out_dir)
                xy_files = list((fig.get("digitize") or {}).get("xy_files") or [])
                if not xy_files:
                    fit_notes.append(f"{fig.get('crop_name')}: 无数字化曲线，跳过拟合")
                    continue
                # 多条曲线并行拟合；仍优先把代表性曲线排在前面
                pick = pick_xy_for_fit(xy_files)
                if args.fit_all_curves:
                    ordered = ([pick] if pick else []) + [p for p in xy_files if p != pick]
                else:
                    ordered = [pick] if pick else xy_files[:1]
                for path in ordered:
                    if path:
                        jobs.append((fig, path))

            t_fit0 = time.perf_counter()
            if jobs:
                nfit = min(max(2, args.workers), len(jobs))
                results: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
                with ThreadPoolExecutor(max_workers=nfit) as pool:
                    futs = [pool.submit(run_fit_job, fig, xy) for fig, xy in jobs]
                    for fut in as_completed(futs):
                        try:
                            results.append(fut.result())
                        except Exception as exc:
                            print(f"    拟合异常：{redact_secret(exc, api_key)}", file=sys.stderr)
                # 解读可以并行，但不阻塞报告主路径太久
                def interpret_one(fig, xy_pick, fit_res):
                    if fit_res.get("status") != "ok" or not api_key or not fit_res.get("report"):
                        return
                    try:
                        t_interp = time.perf_counter()
                        prompt = interpret_prompt(fit_res["report"], fig.get("caption") or "", fig.get("peak_table") or "")
                        png = fit_res.get("fit_png")
                        data_url = encode_image(Path(png), max_side=args.max_side)["data_url"] if png else []
                        raw, usage = call_model(
                            data_url,
                            prompt,
                            model=args.phase_model or args.model,
                            api_key=api_key,
                            max_retries=2,
                            thinking=args.thinking,
                            timeout=max(20.0, float(args.call_timeout) or 20.0),
                            max_tokens=1024,
                        )
                        parsed_i, _ = extract_json(raw)
                        add_usage(usage_detect, usage)
                        if parsed_i:
                            fit_res["interp"] = parsed_i
                            extra = "；解读：" + str(parsed_i.get("summary") or "")
                            fit_res["summary"] = (fit_res.get("summary") or "") + extra
                        print(f"    [计时] 拟合解读 {Path(xy_pick).name} {round(time.perf_counter()-t_interp, 1)} s")
                    except Exception as exc:
                        print(f"    拟合解读失败：{redact_secret(exc, api_key)}", file=sys.stderr)

                with ThreadPoolExecutor(max_workers=min(4, max(1, len(results)))) as pool:
                    list(pool.map(lambda row: interpret_one(*row), results))

                # 写回每张图：优先保留代表性曲线的结果
                by_fig: dict[str, list[tuple[str, dict[str, Any]]]] = {}
                for fig, xy_pick, fit_res in results:
                    key = fig.get("crop_path") or fig.get("crop_name")
                    by_fig.setdefault(key, []).append((xy_pick, fit_res))
                    print(
                        f"    {fig.get('crop_name')} / {Path(xy_pick).name}: "
                        f"{fit_res.get('summary') or fit_res.get('message') or fit_res.get('status')}"
                        f"  (CIF {fit_res.get('cif_s')} s, DARA {fit_res.get('dara_s')} s)"
                    )
                for fig in all_figs:
                    key = fig.get("crop_path") or fig.get("crop_name")
                    rows = by_fig.get(key) or []
                    if not rows:
                        continue
                    pick = pick_xy_for_fit([p for p, _ in rows]) or rows[0][0]
                    chosen = next((r for p, r in rows if p == pick), rows[0][1])
                    fig["fit"] = chosen
                    fig["fits"] = {Path(p).name: r for p, r in rows}
                    fig["fit_summary"] = chosen.get("summary") or chosen.get("message") or chosen.get("status")
                    rec = rec_by_crop.get(fig.get("crop_path") or "")
                    if rec:
                        rec["fit_summary"] = fig["fit_summary"]
                    fit_notes.append(f"{fig.get('crop_name')}: {fig['fit_summary']}")
            t_fit_s = round(time.perf_counter() - t_fit0, 2)
            print(f"    [计时] DARA 阶段墙钟 {t_fit_s} s")
    elif not args.fit:
        # 不在这里探测 DARA（会启动独立解释器，白白加数秒）。需要拟合时再探测。
        fit_notes.append("第 3 步未运行（默认关闭）。加 --fit 才会调用 DARA。")

    usage_read = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for rec in records:
        add_usage(usage_read, rec.get("usage") if isinstance(rec.get("usage"), dict) else None)
    # usage_detect 在流式路径里已经计入识图 token；再加 usage_read 会重复。
    # 识图 usage 只从 records 汇总一次，定位 usage 留在 detects / usage_detect 的「定位部分」。
    usage_locate = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for d in detects:
        add_usage(usage_locate, d.get("usage"))
    # enrich 的定标/复核记在 usage_step2；不要把识图再加进 usage_detect
    usage_all = {
        k: usage_locate.get(k, 0) + usage_step2.get(k, 0) + usage_read.get(k, 0)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    }

    elapsed = round(time.perf_counter() - t0, 1)
    # 阶段墙钟（允许重叠，所以分项之和可以大于总墙钟）
    timing_lines = [
        f"总墙钟 {elapsed} s（Asia/Seoul {datetime.now().strftime('%Y-%m-%d %H:%M')}）",
        f"筛页+定位+裁图墙钟 {round(t_locate_done or 0, 1)} s"
        + (f"（第一幅裁图 @ {t_first_crop:.1f} s）" if t_first_crop is not None else ""),
        f"裁图后处理（数字化∥识图）结束 @ {round(t_figures_done, 1)} s",
    ]
    if t_fit_s:
        timing_lines.append(f"DARA 阶段墙钟 {t_fit_s} s")
    for rec in records:
        st = rec.get("stage_s") or {}
        if st:
            timing_lines.append(
                f"  {st.get('crop_name')}: 数字化 {st.get('digitize_s')} s，识图 {st.get('phase_s')} s"
            )
    print("\n[计时汇总] " + "；".join(timing_lines[:4]))
    for line in timing_lines[4:]:
        print("  " + line)

    intro = write_detect_markdown(
        detects,
        usage_locate,
        args.locate_model or args.model,
    )
    intro += "\n## 数字化与拟合\n\n"
    intro += f"- 第 2 步（读刻度/复核）{format_usage(usage_step2)}；费用 {estimate_cost_cny(args.locate_model or args.model, usage_step2)}\n"
    for fig in all_figs:
        dig = fig.get("digitize") or {}
        if not dig:
            continue
        intro += (
            f"- {fig.get('crop_name')}: 描线 {dig.get('status')}（{dig.get('method') or ''}），"
            f"{dig.get('n_traces') or 0} 条；定标 {(fig.get('calib') or {}).get('summary') or '无'}；"
            f"复核 {(fig.get('review') or {}).get('verdict') or '未做'}\n"
        )
    if fit_notes:
        intro += "- 第 3 步（DARA）：\n"
        for note in fit_notes:
            intro += f"  - {note}\n"
    intro += (
        f"\n- 本轮墙钟时间：{elapsed} 秒（Asia/Seoul {datetime.now().strftime('%Y-%m-%d %H:%M')}）\n"
        f"- 阶段：{'；'.join(timing_lines)}\n"
        f"- 识图阶段：{format_usage(usage_read)}；费用 {estimate_cost_cny(args.phase_model or args.model, usage_read)}\n"
        f"- 全过程合计：{format_usage(usage_all)}；费用 {estimate_cost_cny(args.phase_model or args.model, usage_all)}\n"
    )

    summary_path = out_dir / "summary.csv"
    peaks_path = out_dir / "peaks.csv"
    report_path = out_dir / "report.md"
    write_summary_csv(records, summary_path)
    write_peaks_csv(records, peaks_path)
    write_report_md(
        records,
        report_path,
        meta={
            "model": args.phase_model or args.model,
            "tasks": tasks,
            "out_dir": str(out_dir.resolve()),
            "pdfs": "；".join(p.name for p in pdfs),
            "intro_md": intro,
        },
    )

    index_path = out_dir / "figures_index.json"
    index_path.write_text(
        json.dumps(
            {
                "model": args.phase_model or args.model,
                "locate_model": args.locate_model,
                "dry_run": args.dry_run,
                "pdfs": [d.get("pdf") for d in detects],
                "usage_detect": usage_locate,
                "usage_read": usage_read,
                "usage_all": usage_all,
                "elapsed_s": elapsed,
                "usage_step2": usage_step2,
                "timing": timing_lines,
                "figures": [
                    {
                        "source_pdf": fig.get("source_pdf"),
                        "page": fig.get("page"),
                        "figure_label": fig.get("figure_label"),
                        "caption": fig.get("caption"),
                        "crop_name": fig.get("crop_name"),
                        "crop_path": fig.get("crop_path"),
                        "box_1000": fig.get("box_1000"),
                        "confidence": fig.get("confidence"),
                        "calib": {k: (fig.get("calib") or {}).get(k) for k in (
                            "status", "summary", "ticks", "curve_count", "wavelength_angstrom", "wavelength_note",
                            "curves", "plot_box_1000", "mask_boxes_1000", "x_range", "x_label", "y_label",
                            "legend", "notes", "radiation", "target", "x_quantity",
                        )},
                        "digitize": fig.get("digitize"),
                        "review": fig.get("review"),
                        "fit_summary": fig.get("fit_summary"),
                        "phase": fig.get("phase"),
                        "phase_status": fig.get("phase_status"),
                        "phase_error": fig.get("phase_error"),
                        "phase_model": fig.get("phase_model"),
                    }
                    for fig in all_figs
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    ok = sum(1 for r in records if r.get("status") in {"ok", "dry-run", "skipped"})
    failed = sum(1 for r in records if r.get("status") not in {"ok", "dry-run", "skipped", None} and r)
    print(
        f"\n完成：裁图 {len(all_figs)} 幅，识图成功/演练/跳过 {ok}，失败 {failed}。"
        f"  用时 {elapsed} s。  {format_usage(usage_all)}；{estimate_cost_cny(args.phase_model or args.model, usage_all)}"
        f"\n  {summary_path}\n  {peaks_path}\n  {report_path}\n  {index_path}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
