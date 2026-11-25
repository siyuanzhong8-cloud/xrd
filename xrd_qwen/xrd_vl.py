#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量调用阿里云百炼通义千问视觉模型，识读 XRD 图谱。

API Key 只从环境变量 DASHSCOPE_API_KEY 读取，绝不写入代码、也不会打印。
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

# 延迟导入 openai，便于 --dry-run 在未安装 openai 时仍能检查图片编码
# （正常安装 requirements.txt 后两者都应可用）

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DATA_SUFFIXES = {".xy", ".txt", ".csv", ".dat", ".xye"}
DEFAULT_MODEL = "qwen3.5-omni-flash"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MAX_SIDE = 2048
ALL_TASKS = ("layout", "peaks", "phase", "synthesis")
TASK_ALIASES = {
    "layout": "layout",
    "版式": "layout",
    "peaks": "peaks",
    "读图": "peaks",
    "peak": "peaks",
    "phase": "phase",
    "物相": "phase",
    "synthesis": "synthesis",
    "合成": "synthesis",
    "路线": "synthesis",
}

# 本账号在国内兼容域名上实测可用的视觉模型（2026-11）。qwen-vl-max / qwen-vl-plus / qwen3-vl-flash 已返回 404（下线）。
KNOWN_VISION_MODELS = [
    "qwen3.8-flash",
    "qwen3.6-flash",
    "qwen3.5-flash",
    "qwen3.7-flash",
    "qwen3-vl-plus",
    "deepseek-v4.1-flash-1017",
]


def get_api_key() -> str:
    """只从环境变量读取密钥，缺失时给出可操作的提示，绝不回显密钥内容。"""
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "未检测到环境变量 DASHSCOPE_API_KEY。\n"
            "PowerShell 临时设置：$env:DASHSCOPE_API_KEY = \"sk-你的密钥\"\n"
            "永久设置（需关闭并重开终端后生效）：setx DASHSCOPE_API_KEY \"sk-你的密钥\"\n"
            "申请方式见《使用说明.md》。调试编码可用 --dry-run，无需密钥。"
        )
    return key


def redact_secret(text: Any, api_key: str | None = None) -> str:
    """从日志和落盘错误信息中去掉 Key，避免意外泄露。"""
    out = "" if text is None else str(text)
    key = (api_key or os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if key:
        out = out.replace(key, "[REDACTED]")
    return re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", out)


def usage_from_completion(completion: Any) -> dict[str, int | None] | None:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def format_usage(usage: dict | None) -> str:
    if not usage:
        return "用量未知"
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    return f"token 输入 {prompt} / 输出 {completion} / 合计 {total}"


def collect_images(input_path: Path) -> list[Path]:
    """输入可以是单张图或文件夹；文件夹下只扫一层，不递归。"""
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(
                f"不支持的图片格式：{input_path.suffix}。"
                "请使用 png/jpg/jpeg/bmp/tif/tiff/webp。"
            )
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            p
            for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            raise ValueError(f"文件夹里没有支持的图片：{input_path}")
        return files
    raise FileNotFoundError(f"找不到输入路径：{input_path}")


def find_data_file(
    image_path: Path,
    data_arg: Path | None,
    *,
    allow_mismatched_stem: bool = False,
) -> Path | None:
    """按同名规则为图片配对原始谱图数据。

    --data 指向文件：默认仅当文件名（不含扩展名）与图片相同才使用；
    若只处理一张图，允许文件名不同（调用方已明确指定该数据文件）。
    --data 指向文件夹：在其中查找与图片同名的 .xy/.csv/.txt 等。
    优先顺序：.xy > .xye > .csv > .dat > .txt。
    """
    if data_arg is None:
        return None
    stem = image_path.stem
    if data_arg.is_file():
        if data_arg.stem == stem or allow_mismatched_stem:
            return data_arg
        return None
    if data_arg.is_dir():
        for ext in (".xy", ".xye", ".csv", ".dat", ".txt"):
            candidate = data_arg / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
    return None


def encode_image(
    path: Path,
    max_side: int = DEFAULT_MAX_SIDE,
    *,
    prefer_jpeg: bool = False,
    jpeg_quality: int = 85,
) -> dict[str, Any]:
    """把图片读入、必要时缩放/转码，再编码为 data URL。

    tif/tiff/bmp 统一转为 PNG 再发送；长边超过 max_side 时按比例缩小。
    prefer_jpeg=True 时改发 JPEG（适合整页定位，显著减小上传体积）。
    返回字典供调用和 --dry-run 检查，不打印 base64 内容。
    """
    suffix = path.suffix.lower()
    with Image.open(path) as im:
        # 多页 TIFF 只取第一页
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        orig_w, orig_h = im.size
        orig_mode = im.mode

        # 带透明通道的图转到 RGB（白底），避免 JPEG 或部分模型处理异常
        if suffix in {".tif", ".tiff", ".bmp"}:
            out_format, mime = "PNG", "image/png"
            if im.mode in {"RGBA", "LA", "P"}:
                rgba = im.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.split()[-1])
                im = background
            else:
                im = im.convert("RGB")
        elif suffix == ".webp":
            out_format, mime = "WEBP", "image/webp"
            im = im.convert("RGB") if im.mode not in {"RGB", "RGBA"} else im
        elif suffix in {".jpg", ".jpeg"}:
            out_format, mime = "JPEG", "image/jpeg"
            if im.mode != "RGB":
                im = im.convert("RGB")
        else:
            # .png 及其他
            out_format, mime = "PNG", "image/png"
            if im.mode == "P":
                im = im.convert("RGBA")

        # 长边超限则等比缩放，减少 token 消耗并避免接口拒收过大图
        w, h = im.size
        resized = False
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            im = im.resize(new_size, Image.Resampling.LANCZOS)
            resized = True
            w, h = im.size

        if prefer_jpeg:
            if im.mode not in {"RGB", "L"}:
                im = im.convert("RGB")
            elif im.mode == "L":
                im = im.convert("RGB")
            out_format, mime = "JPEG", "image/jpeg"
        buf = io.BytesIO()
        save_kwargs: dict[str, Any] = {}
        if out_format == "JPEG":
            save_kwargs["quality"] = int(jpeg_quality) if prefer_jpeg else 92
            save_kwargs["optimize"] = True
        im.save(buf, format=out_format, **save_kwargs)
        raw = buf.getvalue()

    b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    return {
        "data_url": data_url,
        "mime": mime,
        "format": out_format,
        "orig_size": (orig_w, orig_h),
        "send_size": (w, h),
        "resized": resized,
        "nbytes": len(raw),
        "orig_mode": orig_mode,
        "path": str(path),
    }


def load_peak_table_text(data_path: Path) -> str:
    """读取配对的原始数据并寻峰，失败时返回说明文字而不是中断整张图。"""
    # 局部导入，避免循环依赖以外的硬失败
    try:
        from xy_peaks import find_peaks_xy, format_peak_table, load_xy
    except ImportError as exc:
        return f"（无法导入 xy_peaks.py：{exc}）"
    try:
        x, y = load_xy(data_path)
        peaks = find_peaks_xy(x, y)
        return format_peak_table(peaks, title=f"数据文件：{data_path.name}")
    except Exception as exc:  # noqa: BLE001 读数据失败不应让整张图放弃
        return f"（读取或寻峰失败：{data_path.name}，原因：{exc}）"


def parse_tasks(text: str) -> list[str]:
    """解析 --tasks，允许中英文别名，去重且保持 ALL_TASKS 中的顺序。"""
    if not text or text.strip().lower() in {"all", "全部"}:
        return list(ALL_TASKS)
    wanted: list[str] = []
    for part in re.split(r"[,，\s]+", text.strip()):
        if not part:
            continue
        key = TASK_ALIASES.get(part.lower()) or TASK_ALIASES.get(part)
        if not key:
            raise ValueError(
                f"未知任务 “{part}”。可选：layout/peaks/phase/synthesis（或 版式/读图/物相/合成）"
            )
        if key not in wanted:
            wanted.append(key)
    if not wanted:
        raise ValueError("至少选择一项任务。")
    return [t for t in ALL_TASKS if t in wanted]


def build_prompt(
    tasks: list[str],
    context: str | None,
    peak_table: str | None,
    image_name: str,
) -> str:
    """构造用户提示词：要求只输出 JSON，字段值用中文，并注明不确定性。"""
    task_spec = {
        "layout": (
            "版式（写入 layout 字段）：判断是单图还是多子图及子图数量；"
            "读出各子图坐标轴标签、单位与大致范围；曲线条数与图例文字；"
            "是否出现 PDF/JCPDS/ICDD 标准卡片竖线（stick/tick）标注，有则记下卡片号或物相名。"
        ),
        "peaks": (
            "读图（写入 peaks 字段）：按曲线分别列出主要衍射峰。"
            "每条峰给出 two_theta（数值，单位 °）、relative_intensity（最强峰记 100）、"
            "hkl（图上有标注就照抄，没有则填 null，不要编造）、confidence（0 到 1 的小数）。"
            "若下方提供了由原始数据算出的峰表，峰位和相对强度必须优先采用该表，"
            "再结合图上的图例、hkl、卡片竖线做解释；不要用读图结果覆盖数据峰位。"
        ),
        "phase": (
            "物相判断（写入 phase 字段）：给出可能物相、判断依据，以及明确的不确定性说明。"
            "依据应引用峰位、相对强度、hkl 或标准卡片，不要只给物相名称。"
            "只列出峰表或图上有证据支持的相：峰位能对上，或图中已标注/有卡片竖线。"
            "样品背景里出现的原料、掺杂剂、添加剂、弥散相或目标组成，若峰表和图上均无对应衍射证据，不要列为物相候选。"
            "若信息不足，候选可以多列并降低置信度，禁止把猜测写成定论。"
        ),
        "synthesis": (
            "合成/复合路线建议（写入 synthesis 字段）：在已有物相判断和样品背景的前提下，"
            "给出可能的合成或复合路线、简要理由，并逐条写明不确定性。"
            "强调仅为文献经验层面的参考，不能代替实验方案设计与安全评估。"
        ),
    }

    lines = [
        "你是材料化学领域的 XRD 图谱助手。请仔细阅读这张 X 射线衍射图（以及我可能附上的样品背景和原始数据峰表）。",
        f"图片文件名：{image_name}",
        "",
        "请只输出一个 JSON 对象，不要输出 Markdown 代码围栏，不要输出 JSON 以外的任何文字（包括解释、道歉或前言）。",
        "JSON 中字符串字段请使用中文。数值保持数字类型。图上没有的信息填 null，不要编造。",
        "物相判断和合成路线必须写明不确定性，避免断定语气。",
        "",
        "本请求需要完成的任务：",
    ]
    for t in tasks:
        lines.append(f"- {task_spec[t]}")

    # 完整字段模板，未请求的任务也给出空结构，便于后续表格汇总
    schema = {
        "layout": {
            "figure_type": "单图 或 多子图",
            "subplot_count": 1,
            "axes": [
                {
                    "subplot_index": 1,
                    "x_label": "例如 2θ (°)",
                    "y_label": "例如 Intensity (a.u.)",
                    "x_range": [10, 80],
                    "y_range": [0, "未知则 null"],
                }
            ],
            "curve_count": 1,
            "legend": ["曲线或样品名，无图例则空数组"],
            "has_standard_card_lines": False,
            "standard_card_notes": "有 PDF/JCPDS 竖线时的说明，否则 null",
        },
        "peaks": {
            "curves": [
                {
                    "curve_name": "图例名或曲线1",
                    "peaks": [
                        {
                            "two_theta": 25.3,
                            "relative_intensity": 100,
                            "hkl": "101",
                            "confidence": 0.8,
                        }
                    ],
                }
            ]
        },
        "phase": {
            "candidates": [
                {
                    "name": "物相名",
                    "evidence": "依据（峰位/强度/hkl/卡片）",
                    "confidence": "低 或 中 或 高",
                    "uncertainty": "不确定之处",
                }
            ],
            "summary": "一句话总结",
            "uncertainty_note": "必须填写：本次判断的总体局限，例如仅靠读图、缺少精修等",
        },
        "synthesis": {
            "suggestions": [
                {
                    "route": "可能的合成或复合路线",
                    "rationale": "简要理由",
                    "uncertainty": "该建议的不确定之处",
                }
            ],
            "uncertainty_note": "必须填写：建议仅供参考，需结合实验条件、文献和安全规范验证",
        },
    }
    used_schema = {k: schema[k] for k in ALL_TASKS if k in tasks}
    lines.append("")
    lines.append("请严格使用下面的 JSON 结构（未列出的顶层键不要添加；值按实际填写）：")
    lines.append(json.dumps(used_schema, ensure_ascii=False, indent=2))

    if context and context.strip():
        lines.append("")
        lines.append("样品背景（请在物相判断和路线建议中加以考虑，但仍须标注不确定性）。")
        lines.append("注意：背景文字里的组成或添加剂不是物相证据；没有对应峰就不要列入 phase.candidates。")
        lines.append(context.strip())

    if peak_table and peak_table.strip():
        lines.append("")
        lines.append(
            "以下峰位来自与图片配对的原始谱图数据（仪器导出的 2θ–强度），"
            "精度显著高于从图片估读。读峰和物相判断时请优先使用本表："
        )
        lines.append(peak_table.strip())

    return "\n".join(lines)


def extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """容错解析模型输出：去掉代码围栏、截取最外层花括号、容忍轻微杂质。"""
    if text is None:
        return None, "模型返回为空"
    raw = text.strip()
    if not raw:
        return None, "模型返回为空字符串"

    # 去掉 ```json ... ``` 或 ``` ... ```
    fence = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", raw)
    if fence:
        raw = fence.group(1).strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, None
        return None, f"顶层不是对象，而是 {type(obj).__name__}"
    except json.JSONDecodeError:
        pass

    # 截取第一个 { 到最后一个 }
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj, None
        except json.JSONDecodeError:
            # 再尝试去掉尾逗号
            cleaned = re.sub(r",\s*([}\]])", r"\1", snippet)
            try:
                obj = json.loads(cleaned)
                if isinstance(obj, dict):
                    return obj, None
            except json.JSONDecodeError as exc:
                repaired = _repair_json_snippet(cleaned)
                if repaired is not None:
                    return repaired, None
                salvaged = _salvage_phase_object(raw)
                if salvaged is not None:
                    return salvaged, None
                return None, f"JSON 解析失败：{exc}"
    salvaged = _salvage_phase_object(raw)
    if salvaged is not None:
        return salvaged, None
    return None, "未找到可解析的 JSON 对象"


def _repair_json_snippet(snippet: str) -> dict[str, Any] | None:
    """补上模型漏写的结尾括号后重试解析。"""
    s = snippet.rstrip()
    for _ in range(6):
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError as exc:
            msg = str(exc)
            if "Expecting" in msg and s.count("{") > s.count("}"):
                s = s + "}"
                continue
            if "Expecting" in msg and s.count("[") > s.count("]"):
                s = s + "]"
                continue
            return None
    return None


def _salvage_phase_object(text: str) -> dict[str, Any] | None:
    """JSON 残缺时从 raw 文本里捞出物相名称，避免相位结果完全空掉。"""
    if not text:
        return None
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', text)
    summary = None
    m = re.search(r'"summary"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    if m:
        summary = m.group(1)
    if not names and not summary:
        return None
    return {
        "phase": {
            "candidates": [
                {"name": n, "evidence": "（从略有残缺的模型 JSON 挽救）"}
                for n in names
            ],
            "summary": summary or "",
        }
    }


def _is_rate_limited(exc: BaseException) -> bool:
    """判断是否为 429 / 限流类错误。"""
    if "RateLimit" in type(exc).__name__:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def _retry_seconds(exc: BaseException, attempt: int) -> float:
    """指数退避，限流时等更久；若响应头带 Retry-After 则优先采用。"""
    retry_after = None
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None) or {}
        try:
            retry_after = float(headers.get("Retry-After") or headers.get("retry-after") or 0) or None
        except (TypeError, ValueError):
            retry_after = None
    if retry_after and 0 < retry_after < 120:
        return retry_after
    base = 4.0 if _is_rate_limited(exc) else 1.5
    return min(60.0, base * (2 ** attempt))


def call_model(
    data_url: str | list[str],
    prompt: str,
    model: str,
    api_key: str,
    max_retries: int = 5,
    timeout: float = 180.0,
    thinking: bool = False,
    max_tokens: int = 8192,
) -> tuple[str, dict[str, int | None] | None]:
    """调用百炼 OpenAI 兼容接口，失败按指数退避重试，限流（429）额外拉长等待。

    data_url 可以是单个 data URL，或一组（用于原图+叠加图复核）。
    返回 (模型文本, usage 字典或 None)。
    """
    from openai import OpenAI

    urls = list(data_url) if isinstance(data_url, (list, tuple)) else [data_url]
    user_content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": u}} for u in urls if u
    ]
    user_content.append({"type": "text", "text": prompt})

    client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL, timeout=timeout)
    last_exc: BaseException | None = None
    attempts = max(1, int(max_retries))
    for attempt in range(attempts):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是严谨的 XRD 分析助手。只输出合法 JSON 对象，字段值用中文，不编造读不到的数据。",
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                temperature=0.2,
                max_tokens=int(max_tokens) if max_tokens else 8192,
                extra_body={"enable_thinking": bool(thinking)},
            )
            choice = completion.choices[0]
            content = choice.message.content or ""
            if not str(content).strip():
                nxt = None
                if model not in {"qwen3.8-flash", "qwen3.6-flash"}:
                    nxt = "qwen3.8-flash"
                elif model == "qwen3.8-flash":
                    nxt = "qwen3.6-flash"
                if nxt:
                    print(
                        f"  [提示] 模型 {model} 返回空正文，已改用 {nxt} 重试（不会打印密钥）。",
                        file=sys.stderr,
                    )
                    model = nxt
                    continue
            return content, usage_from_completion(completion)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 需统一做重试分类
            last_exc = exc
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            # 4xx 中除 429 外（如 401、400 模型名错误）重试无意义
            if status is not None and 400 <= int(status) < 500 and int(status) != 429:
                msg = str(exc).lower()
                overdue = ("overdue" in msg) or ("access denied" in msg) or ("good standing" in msg)
                nxt = None
                # 默认物相模型（如 3.5-omni）欠费/失败时改走 3.8；3.8 若 403 再走 3.6
                if overdue and model not in {"qwen3.8-flash", "qwen3.6-flash"}:
                    nxt = "qwen3.8-flash"
                    why = f"模型 {model} 返回账户状态/欠费错误，已改用 {nxt} 重试（不会打印密钥）。"
                elif int(status) == 403 and "3.8" in (model or ""):
                    nxt = "qwen3.6-flash"
                    why = (
                        "qwen3.8-flash 返回 403：免费额度已用尽。"
                        "若要用付费调用，请在百炼控制台关闭「仅使用免费额度」，并确认账户已充值。"
                        "本次已改用 qwen3.6-flash 重试，以便流水线完成（不会打印密钥）。"
                    )
                elif model not in {"qwen3.8-flash", "qwen3.6-flash"}:
                    nxt = "qwen3.8-flash"
                    why = f"模型 {model} 调用失败（HTTP {status}），已改用 {nxt} 重试（不会打印密钥）。"
                if nxt and nxt != model:
                    print("  [提示] " + why, file=sys.stderr)
                    model = nxt
                    continue
                raise
            if attempt >= attempts - 1:
                # 最后一次仍失败：默认模型改走 3.8，3.8 再改走 3.6
                nxt = None
                if model not in {"qwen3.8-flash", "qwen3.6-flash"}:
                    nxt = "qwen3.8-flash"
                elif model == "qwen3.8-flash":
                    nxt = "qwen3.6-flash"
                if nxt:
                    print(
                        f"  [提示] 模型 {model} 多次调用失败，已改用 {nxt} 再试一次（不会打印密钥）。",
                        file=sys.stderr,
                    )
                    model = nxt
                    attempts += 1
                    continue
                break
            wait = _retry_seconds(exc, attempt)
            kind = "限流 429" if _is_rate_limited(exc) else "调用失败"
            print(
                f"  [{kind}，第 {attempt + 1}/{attempts} 次，{wait:.1f} 秒后重试] {redact_secret(exc, api_key)}",
                file=sys.stderr,
            )
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def call_model_hedged(
    data_url: str | list[str],
    prompt: str,
    model: str,
    api_key: str,
    *,
    max_retries: int = 2,
    timeout: float = 25.0,
    thinking: bool = False,
    max_tokens: int = 1536,
    hedge_after: float | None = 12.0,
    hedge_model: str | None = None,
) -> tuple[str, dict[str, int | None] | None]:
    """若首请求超过 hedge_after 秒仍未返回，再发一趟（可用备用模型）；先完成且成功的结果获胜。

    hedge_after 为 None 或 <=0 时退化为普通 call_model（带 timeout 与有限重试）。
    未完成的请求无法真正取消，只会丢弃其结果。
    """
    if not hedge_after or hedge_after <= 0:
        return call_model(
            data_url,
            prompt,
            model=model,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            thinking=thinking,
            max_tokens=max_tokens,
        )

    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    def _run(which: str, use_model: str) -> tuple[str, tuple[str, dict[str, int | None] | None]]:
        return which, call_model(
            data_url,
            prompt,
            model=use_model,
            api_key=api_key,
            max_retries=max(1, int(max_retries)),
            timeout=timeout,
            thinking=thinking,
            max_tokens=max_tokens,
        )

    alt = (hedge_model or model).strip() or model
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_run, "primary", model)
        done, _ = wait([first], timeout=float(hedge_after))
        if first in done:
            _which, payload = first.result()
            return payload
        second = pool.submit(_run, "hedge", alt)
        finished, _ = wait([first, second], return_when=FIRST_COMPLETED)
        errors: list[BaseException] = []
        for fut in list(finished) + ([first, second]):
            if not fut.done():
                continue
            try:
                _which, payload = fut.result()
                return payload
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise errors[-1]
        raise TimeoutError(f"定位请求在 {timeout} 秒内未返回")


def result_json_path(out_dir: Path, image_path: Path) -> Path:
    """用完整文件名区分同主文件名、不同后缀的图（如 a.png 与 a.tif）。"""
    return out_dir / f"{image_path.name}.json"


def process_one(
    image_path: Path,
    *,
    out_dir: Path,
    model: str,
    tasks: list[str],
    context: str | None,
    data_arg: Path | None,
    max_side: int,
    max_retries: int,
    overwrite: bool,
    dry_run: bool,
    api_key: str | None,
    allow_mismatched_data: bool = False,
    thinking: bool = False,
    peak_table: str | None = None,
    timeout: float = 180.0,
    max_tokens: int = 8192,
) -> dict[str, Any]:
    """处理单张图片，写入 results/<stem>.json，并返回汇总用的记录。"""
    dest = result_json_path(out_dir, image_path)
    record: dict[str, Any] = {
        "file": image_path.name,
        "stem": image_path.stem,
        "json_path": str(dest),
        "status": "",
        "model": model,
        "data_file": "",
        "error": "",
        "parsed": None,
        "elapsed_s": None,
        "usage": None,
        "thinking": thinking,
    }

    if dest.exists() and not overwrite:
        record["status"] = "skipped"
        # 尽量读出已有结果，方便重新生成汇总表
        try:
            existing = json.loads(dest.read_text(encoding="utf-8"))
            record["parsed"] = existing.get("result") if isinstance(existing, dict) else None
            if record["parsed"] is None and isinstance(existing, dict) and "layout" in existing:
                record["parsed"] = existing
        except Exception:
            pass
        print(f"[跳过] {image_path.name}（已有 {dest.name}，需要重跑请加 --overwrite）")
        return record

    try:
        encoded = encode_image(image_path, max_side=max_side)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = redact_secret(f"图片编码失败：{exc}", api_key)
        print(f"[失败] {image_path.name}：{record['error']}", file=sys.stderr)
        return record

    data_file = find_data_file(
        image_path, data_arg, allow_mismatched_stem=allow_mismatched_data
    )
    # 可由调用方（如 pdf_xrd 数字化后的峰表）直接传入；未传入时再按 --data 配对
    given_peak_table = peak_table
    if data_file is not None:
        record["data_file"] = data_file.name
        if given_peak_table is None:
            peak_table = load_peak_table_text(data_file)
            # 单文件 --data 配到多张不同名图片时，find_data_file 仍可能返回该文件；
            # 仅当 stem 一致，或输入本身就是单图时才真正附上。
            if data_arg is not None and data_arg.is_file() and data_file.stem != image_path.stem:
                # 仍附上，但在文本里标明文件名，避免张冠李戴却完全无声
                peak_table = "注意：该数据文件名与图片不完全相同，请谨慎对应。\n" + peak_table
        else:
            peak_table = given_peak_table
    elif given_peak_table:
        record["data_file"] = "digitized.xy"
        peak_table = given_peak_table

    prompt = build_prompt(tasks, context, peak_table, image_path.name)

    if dry_run:
        record["status"] = "dry-run"
        dry_info = {
            "status": "dry-run",
            "file": image_path.name,
            "model": model,
            "tasks": tasks,
            "mime": encoded["mime"],
            "orig_size": list(encoded["orig_size"]),
            "send_size": list(encoded["send_size"]),
            "resized": encoded["resized"],
            "nbytes": encoded["nbytes"],
            "data_url_prefix": encoded["data_url"][:40] + "...",
            "data_url_chars": len(encoded["data_url"]),
            "prompt_chars": len(prompt),
            "data_file": record["data_file"],
            "prompt_preview": prompt[:800],
        }
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(dry_info, ensure_ascii=False, indent=2), encoding="utf-8")
        size = f"{encoded['orig_size'][0]}x{encoded['orig_size'][1]}"
        send = f"{encoded['send_size'][0]}x{encoded['send_size'][1]}"
        flag = "，已缩放" if encoded["resized"] else ""
        print(
            f"[演练] {image_path.name}  {size} -> {send}{flag}  "
            f"{encoded['mime']}  {encoded['nbytes']} 字节  提示词 {len(prompt)} 字"
        )
        return record

    assert api_key is not None
    started = time.perf_counter()
    try:
        raw_text, usage = call_model(
            encoded["data_url"],
            prompt,
            model=model,
            api_key=api_key,
            max_retries=max_retries,
            thinking=thinking,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        elapsed = round(time.perf_counter() - started, 2)
        record["elapsed_s"] = elapsed
        record["usage"] = usage
    except Exception as exc:
        elapsed = round(time.perf_counter() - started, 2)
        record["elapsed_s"] = elapsed
        record["status"] = "failed"
        record["error"] = redact_secret(f"接口调用失败：{exc}", api_key)
        payload = {
            "status": "failed",
            "file": image_path.name,
            "model": model,
            "error": record["error"],
            "tasks": tasks,
            "elapsed_s": elapsed,
        }
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[失败] {image_path.name}：{record['error']}", file=sys.stderr)
        return record

    parsed, err = extract_json(raw_text)
    if parsed is None and model not in {"qwen3.8-flash", "qwen3.6-flash"}:
        # 解析失败再换 3.8 试一次，避免相位空结果
        try:
            print(
                f"  [提示] 模型 {model} 输出无法解析，已改用 qwen3.8-flash 重试（不会打印密钥）。",
                file=sys.stderr,
            )
            raw2, usage2 = call_model(
                encoded["data_url"],
                prompt,
                model="qwen3.8-flash",
                api_key=api_key,
                max_retries=max(1, min(int(max_retries), 2)),
                thinking=thinking,
                timeout=timeout,
                max_tokens=max_tokens,
            )
            raw_text = raw2 or raw_text
            if usage2:
                record["usage"] = usage2
            parsed, err = extract_json(raw_text)
        except Exception as exc:
            record.setdefault("error", redact_secret(f"回退解析重试失败：{exc}", api_key))
    dest.parent.mkdir(parents=True, exist_ok=True)
    if parsed is None:
        record["status"] = "parse_failed"
        record["error"] = err or "解析失败"
        payload = {
            "status": "parse_failed",
            "file": image_path.name,
            "model": model,
            "error": record["error"],
            "raw_text": raw_text,
            "tasks": tasks,
            "data_file": record["data_file"],
            "elapsed_s": record["elapsed_s"],
            "usage": record["usage"],
        }
        dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[解析失败] {image_path.name}：已保存原始文本 -> {dest}  "
            f"{record['elapsed_s']}s  {format_usage(record['usage'])}"
        )
        return record

    record["status"] = "ok"
    record["parsed"] = parsed
    payload = {
        "status": "ok",
        "file": image_path.name,
        "model": model,
        "tasks": tasks,
        "data_file": record["data_file"],
        "elapsed_s": record["elapsed_s"],
        "usage": record["usage"],
        "thinking": thinking,
        "result": parsed,
    }
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[完成] {image_path.name} -> {dest}  "
        f"{record['elapsed_s']}s  {format_usage(record['usage'])}"
    )
    return record


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def flatten_phase_names(parsed: dict | None) -> str:
    if not isinstance(parsed, dict):
        return ""
    phase = parsed.get("phase") or {}
    names = []
    for c in _as_list(phase.get("candidates")):
        if isinstance(c, dict) and c.get("name"):
            names.append(str(c["name"]))
    if names:
        return "；".join(names)
    return str(phase.get("summary") or "")


def flatten_synthesis(parsed: dict | None) -> str:
    if not isinstance(parsed, dict):
        return ""
    syn = parsed.get("synthesis") or {}
    routes = []
    for s in _as_list(syn.get("suggestions")):
        if isinstance(s, dict) and s.get("route"):
            routes.append(str(s["route"]))
    return "；".join(routes)


def write_summary_csv(records: list[dict[str, Any]], path: Path) -> None:
    """汇总表，UTF-8 BOM，方便 Excel 直接打开。"""
    fields = [
        "文件名",
        "状态",
        "模型",
        "配对数据",
        "图类型",
        "子图数",
        "曲线数",
        "有标准卡片竖线",
        "物相候选",
        "合成路线摘要",
        "耗时秒",
        "输入Token",
        "输出Token",
        "总Token",
        "错误信息",
        "结果JSON",
    ]
    # pdf_xrd.py 传入的可选来源信息；仅当至少一条记录带有这些字段时才追加列，不影响纯图片流程
    if any(r.get("source_pdf") or r.get("page") or r.get("caption") or r.get("crop_path") for r in records):
        fields = fields + ["来源PDF", "页码", "图号", "图注", "裁图路径"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            parsed = r.get("parsed") if isinstance(r.get("parsed"), dict) else {}
            layout = parsed.get("layout") or {}
            row = {
                "文件名": r.get("file", ""),
                "状态": r.get("status", ""),
                "模型": r.get("model", ""),
                "配对数据": r.get("data_file", ""),
                "图类型": layout.get("figure_type", ""),
                "子图数": layout.get("subplot_count", ""),
                "曲线数": layout.get("curve_count", ""),
                "有标准卡片竖线": layout.get("has_standard_card_lines", ""),
                "物相候选": flatten_phase_names(parsed),
                "合成路线摘要": flatten_synthesis(parsed),
                "耗时秒": r.get("elapsed_s", ""),
                "输入Token": (r.get("usage") or {}).get("prompt_tokens", "") if isinstance(r.get("usage"), dict) else "",
                "输出Token": (r.get("usage") or {}).get("completion_tokens", "") if isinstance(r.get("usage"), dict) else "",
                "总Token": (r.get("usage") or {}).get("total_tokens", "") if isinstance(r.get("usage"), dict) else "",
                "错误信息": r.get("error", ""),
                "结果JSON": r.get("json_path", ""),
            }
            if "来源PDF" in fields:
                row.update({
                    "来源PDF": r.get("source_pdf", ""),
                    "页码": r.get("page", ""),
                    "图号": r.get("figure_label", ""),
                    "图注": r.get("caption", ""),
                    "裁图路径": r.get("crop_path", ""),
                })
            writer.writerow(row)


def write_peaks_csv(records: list[dict[str, Any]], path: Path) -> None:
    """把各图读出的峰位展开成一张总表。"""
    fields = [
        "文件名",
        "曲线名",
        "2θ",
        "相对强度",
        "hkl",
        "置信度",
        "来源",
    ]
    if any(r.get("source_pdf") or r.get("page") for r in records):
        fields = ["来源PDF", "页码", "图号"] + fields
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            parsed = r.get("parsed") if isinstance(r.get("parsed"), dict) else {}
            peaks_obj = parsed.get("peaks") or {}
            curves = peaks_obj.get("curves") if isinstance(peaks_obj, dict) else None
            if not curves:
                continue
            source = "原始数据+模型" if r.get("data_file") else "模型读图"
            for curve in curves:
                if not isinstance(curve, dict):
                    continue
                name = curve.get("curve_name") or ""
                for peak in _as_list(curve.get("peaks")):
                    if not isinstance(peak, dict):
                        continue
                    row = {
                        "文件名": r.get("file", ""),
                        "曲线名": name,
                        "2θ": peak.get("two_theta", ""),
                        "相对强度": peak.get("relative_intensity", ""),
                        "hkl": peak.get("hkl", ""),
                        "置信度": peak.get("confidence", ""),
                        "来源": source,
                    }
                    if "来源PDF" in fields:
                        row.update({
                            "来源PDF": r.get("source_pdf", ""),
                            "页码": r.get("page", ""),
                            "图号": r.get("figure_label", ""),
                        })
                    writer.writerow(row)


def _report_relpath(target: Any, report_path: Path) -> str:
    """把裁图路径转成相对报告文件的 POSIX 路径，便于 Markdown 内嵌图片。"""
    if not target:
        return ""
    try:
        p = Path(str(target))
        base = report_path.parent.resolve()
        try:
            return p.resolve().relative_to(base).as_posix()
        except ValueError:
            return p.as_posix()
    except Exception:
        return str(target)


def _md_escape(text: Any) -> str:
    s = "" if text is None else str(text)
    return s.replace("|", "\\|").replace("\n", " ")


def write_report_md(records: list[dict[str, Any]], path: Path, meta: dict[str, Any]) -> None:
    """生成给人阅读的 Markdown 汇总报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        "# XRD 识图汇总报告",
        "",
        f"- 生成时间（Asia/Seoul）：{now}",
        f"- 模型：{meta.get('model', '')}",
        f"- 任务：{', '.join(meta.get('tasks', []))}",
        f"- 图片数量：{len(records)}",
        f"- 输出目录：{meta.get('out_dir', '')}",
        "",
    ]
    if meta.get("pdfs"):
        lines.append(f"- 来源 PDF：{meta['pdfs']}")
        lines.append("")
    if meta.get("intro_md"):
        lines.append(str(meta["intro_md"]).rstrip())
        lines.append("")
    lines.extend([
        "> **重要提示：** 视觉模型从图片估读的峰位和强度精度有限，物相与合成路线仅供参考，"
        "存在不确定性，必须人工核对。若同时提供了 `.xy` 等原始数据，应优先采信数据寻峰结果。",
        "",
        "## 一览",
        "",
    ])
    has_pdf_meta = any(r.get("source_pdf") or r.get("page") for r in records)
    if has_pdf_meta:
        lines.append("| 文件 | 来源 PDF | 页码 | 图号 | 状态 | 耗时 | Token | 图类型 | 曲线数 | 物相候选 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    else:
        lines.append("| 文件 | 状态 | 耗时 | Token | 图类型 | 曲线数 | 物相候选 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in records:
        parsed = r.get("parsed") if isinstance(r.get("parsed"), dict) else {}
        layout = parsed.get("layout") or {}
        common = dict(
            file=_md_escape(r.get("file")),
            status=_md_escape(r.get("status")),
            elapsed=_md_escape(f"{r['elapsed_s']} s" if r.get("elapsed_s") is not None else ""),
            tokens=_md_escape(format_usage(r.get("usage")) if r.get("usage") else ""),
            ftype=_md_escape(layout.get("figure_type")),
            curves=_md_escape(layout.get("curve_count")),
            phase=_md_escape(flatten_phase_names(parsed)),
        )
        if has_pdf_meta:
            lines.append(
                "| {file} | {pdf} | {page} | {fig} | {status} | {elapsed} | {tokens} | {ftype} | {curves} | {phase} |".format(
                    pdf=_md_escape(r.get("source_pdf")),
                    page=_md_escape(r.get("page")),
                    fig=_md_escape(r.get("figure_label")),
                    **common,
                )
            )
        else:
            lines.append(
                "| {file} | {status} | {elapsed} | {tokens} | {ftype} | {curves} | {phase} |".format(**common)
            )

    for r in records:
        lines.extend(["", f"## {r.get('file', '')}", ""])
        lines.append(f"- 状态：{r.get('status') or ''}")
        if r.get("source_pdf"):
            lines.append(f"- 来源 PDF：{r['source_pdf']}")
        if r.get("page") not in (None, ""):
            lines.append(f"- 页码：第 {r['page']} 页")
        if r.get("figure_label"):
            lines.append(f"- 图号：{r['figure_label']}")
        if r.get("caption"):
            lines.append(f"- 图注：{r['caption']}")
        if r.get("crop_path"):
            crop_disp = r["crop_path"]
            lines.append(f"- 裁图路径：{crop_disp}")
            rel = _report_relpath(crop_disp, path)
            if rel:
                lines.append("")
                lines.append(f"![裁图]({rel})")
                lines.append("")
        if r.get("elapsed_s") is not None:
            lines.append(f"- 耗时：{r['elapsed_s']} 秒")
        if r.get("usage"):
            lines.append(f"- {format_usage(r.get('usage'))}")
        if r.get("data_file"):
            lines.append(f"- 配对原始数据：{r['data_file']}")
        if r.get("error"):
            lines.append(f"- 错误：{r['error']}")
        if r.get("xy_files"):
            lines.append("- 数字化曲线：" + "；".join(str(x) for x in r["xy_files"]))
        if r.get("replot_path"):
            lines.append(f"- 重绘图：{r['replot_path']}")
            rel = _report_relpath(r["replot_path"], path)
            if rel:
                lines.append("")
                lines.append(f"![重绘]({rel})")
                lines.append("")
        if r.get("validate_path"):
            lines.append(f"- 叠加校验图：{r['validate_path']}")
            rel = _report_relpath(r["validate_path"], path)
            if rel:
                lines.append("")
                lines.append(f"![叠加校验]({rel})")
                lines.append("")
        if r.get("review_verdict"):
            lines.append(f"- 叠加图复核：{r['review_verdict']}" + (f"（{r.get('review_findings') or ''}）" if r.get("review_findings") else ""))
        if r.get("calib_summary"):
            lines.append(f"- 坐标定标：{r['calib_summary']}")
        if r.get("fit_summary"):
            lines.append(f"- DARA 拟合：{r['fit_summary']}")
        parsed = r.get("parsed") if isinstance(r.get("parsed"), dict) else None
        if not parsed:
            if r.get("status") == "dry-run":
                lines.append("- 本次为演练（--dry-run），未调用模型。")
            continue

        layout = parsed.get("layout") or {}
        if layout:
            lines.append("")
            lines.append("### 版式")
            lines.append("")
            lines.append(f"- 类型：{layout.get('figure_type', '')}；子图数：{layout.get('subplot_count', '')}")
            lines.append(f"- 曲线数：{layout.get('curve_count', '')}")
            legend = layout.get("legend") or []
            if legend:
                lines.append(f"- 图例：{'；'.join(str(x) for x in legend)}")
            card = layout.get("has_standard_card_lines")
            notes = layout.get("standard_card_notes")
            lines.append(f"- 标准卡片竖线：{'有' if card else '无或无法判断'}" + (f"（{notes}）" if notes else ""))
            for axis in _as_list(layout.get("axes")):
                if not isinstance(axis, dict):
                    continue
                lines.append(
                    f"- 坐标轴（子图 {axis.get('subplot_index', '?')}）："
                    f"X={axis.get('x_label')} 范围 {axis.get('x_range')}；"
                    f"Y={axis.get('y_label')} 范围 {axis.get('y_range')}"
                )

        peaks_obj = parsed.get("peaks") or {}
        curves = peaks_obj.get("curves") if isinstance(peaks_obj, dict) else []
        if curves:
            lines.extend(["", "### 主要峰", ""])
            for curve in curves:
                if not isinstance(curve, dict):
                    continue
                lines.append(f"**{curve.get('curve_name') or '曲线'}**")
                lines.append("")
                lines.append("| 2θ (°) | 相对强度 | hkl | 置信度 |")
                lines.append("| --- | --- | --- | --- |")
                for peak in _as_list(curve.get("peaks")):
                    if not isinstance(peak, dict):
                        continue
                    hkl = peak.get("hkl")
                    hkl = "" if hkl is None else hkl
                    lines.append(
                        f"| {_md_escape(peak.get('two_theta'))} | "
                        f"{_md_escape(peak.get('relative_intensity'))} | "
                        f"{_md_escape(hkl)} | {_md_escape(peak.get('confidence'))} |"
                    )
                lines.append("")

        phase = parsed.get("phase") or {}
        if phase:
            lines.extend(["### 物相判断", ""])
            if phase.get("summary"):
                lines.append(phase["summary"])
                lines.append("")
            for c in _as_list(phase.get("candidates")):
                if not isinstance(c, dict):
                    continue
                lines.append(f"- **{c.get('name', '')}**（置信度：{c.get('confidence', '')}）")
                if c.get("evidence"):
                    lines.append(f"  - 依据：{c['evidence']}")
                if c.get("uncertainty"):
                    lines.append(f"  - 不确定性：{c['uncertainty']}")
            if phase.get("uncertainty_note"):
                lines.append("")
                lines.append(f"> 不确定性说明：{phase['uncertainty_note']}")

        syn = parsed.get("synthesis") or {}
        if syn:
            lines.extend(["", "### 合成/复合路线建议", ""])
            for s in _as_list(syn.get("suggestions")):
                if not isinstance(s, dict):
                    continue
                lines.append(f"- **{s.get('route', '')}**")
                if s.get("rationale"):
                    lines.append(f"  - 理由：{s['rationale']}")
                if s.get("uncertainty"):
                    lines.append(f"  - 不确定性：{s['uncertainty']}")
            if syn.get("uncertainty_note"):
                lines.append("")
                lines.append(f"> 不确定性说明：{syn['uncertainty_note']}")

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用阿里云百炼通义千问视觉模型批量识读 XRD 图，输出 JSON / CSV / Markdown 报告。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例（PowerShell）：\n"
            "  python xrd_vl.py .\\examples --dry-run\n"
            "  python xrd_vl.py .\\examples --data .\\examples --context \"锐钛矿 TiO2，Cu Kα\"\n"
            "  python xrd_vl.py .\\图.png --model qwen3.8-flash --tasks layout,peaks,phase\n"
            "\n"
            "有原始 .xy 数据时请加上 --data，读峰会准得多。详细说明见《使用说明.md》。"
        ),
    )
    parser.add_argument("input", help="图片文件或包含图片的文件夹")
    parser.add_argument(
        "--out-dir",
        default="results",
        help="输出目录（默认 results，内含每张图的 json、summary.csv、peaks.csv、report.md）",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"模型名称（默认 {DEFAULT_MODEL}）。实测可用：{', '.join(KNOWN_VISION_MODELS)}",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="开启模型思考模式（默认关闭，调用时传 enable_thinking=False 以加快速度）",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="任务子集，逗号分隔：layout,peaks,phase,synthesis（默认全部）",
    )
    parser.add_argument(
        "--context",
        default=None,
        help="样品背景文字，如前驱体、元素、合成方法、辐射源（Cu Kα 等）",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="与图片同名的原始谱图数据文件或其所在文件夹（.xy/.txt/.csv）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并发数（默认 4）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="已有结果也强制重跑",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不调用 API，只检查图片编码和提示词构造",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="失败重试次数（默认 5，含限流 429 的指数退避）",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=DEFAULT_MAX_SIDE,
        help=f"图片长边像素上限，超出则等比缩小（默认 {DEFAULT_MAX_SIDE}）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        tasks = parse_tasks(args.tasks)
        images = collect_images(Path(args.input))
    except (OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    if args.workers < 1:
        print("错误：--workers 至少为 1", file=sys.stderr)
        return 1

    data_arg = Path(args.data) if args.data else None
    if data_arg is not None and not data_arg.exists():
        print(f"错误：找不到 --data 路径：{data_arg}", file=sys.stderr)
        return 1

    api_key = None
    if not args.dry_run:
        try:
            api_key = get_api_key()
        except RuntimeError as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"待处理 {len(images)} 张图，模型={args.model}，任务={','.join(tasks)}，"
        f"并发={args.workers}，输出目录={out_dir}"
        + ("，思考模式开启" if args.thinking else "，思考模式关闭")
        + ("，演练模式（不调用接口）" if args.dry_run else "")
    )

    records: list[dict[str, Any]] = []
    # 单 worker 时按顺序跑，方便观察日志；多 worker 用线程池
    if args.workers == 1 or len(images) == 1:
        for img in images:
            records.append(
                process_one(
                    img,
                    out_dir=out_dir,
                    model=args.model,
                    tasks=tasks,
                    context=args.context,
                    data_arg=data_arg,
                    max_side=args.max_side,
                    max_retries=args.max_retries,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    api_key=api_key,
                    allow_mismatched_data=(len(images) == 1),
                    thinking=args.thinking,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_one,
                    img,
                    out_dir=out_dir,
                    model=args.model,
                    tasks=tasks,
                    context=args.context,
                    data_arg=data_arg,
                    max_side=args.max_side,
                    max_retries=args.max_retries,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    api_key=api_key,
                    allow_mismatched_data=(len(images) == 1),
                    thinking=args.thinking,
                ): img
                for img in images
            }
            # 按完成顺序收集，最后再按文件名排序，保证汇总表稳定
            done: dict[str, dict[str, Any]] = {}
            for fut in as_completed(futures):
                img = futures[fut]
                try:
                    done[str(img)] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    done[str(img)] = {
                        "file": img.name,
                        "stem": img.stem,
                        "json_path": str(result_json_path(out_dir, img)),
                        "status": "failed",
                        "model": args.model,
                        "data_file": "",
                        "error": redact_secret(f"未捕获异常：{exc}", api_key),
                        "parsed": None,
                        "elapsed_s": None,
                        "usage": None,
                    }
                    print(redact_secret(traceback.format_exc(), api_key), file=sys.stderr)
            records = [done[str(img)] for img in images]

    summary_path = out_dir / "summary.csv"
    peaks_path = out_dir / "peaks.csv"
    report_path = out_dir / "report.md"
    write_summary_csv(records, summary_path)
    write_peaks_csv(records, peaks_path)
    write_report_md(
        records,
        report_path,
        meta={"model": args.model, "tasks": tasks, "out_dir": str(out_dir.resolve())},
    )

    ok = sum(1 for r in records if r["status"] in {"ok", "dry-run", "skipped"})
    failed = sum(1 for r in records if r["status"] not in {"ok", "dry-run", "skipped"})
    print(
        f"\n汇总：成功/演练/跳过 {ok}，失败 {failed}。"
        f"\n  {summary_path}\n  {peaks_path}\n  {report_path}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
