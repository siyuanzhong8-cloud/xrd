#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 1 步补充：让视觉模型在裁好的 XRD 图上读出坐标刻度、曲线与靶材线索。

刻度位置用来替代工作台原来依赖的 Windows OCR，供 digitize_crop 做像素→2θ 定标。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from xrd_vl import call_model, encode_image, extract_json, format_usage, redact_secret

# 常用靶材的 Kα 波长（Å）。Co 论文常用 Co Kα；缺省按 Cu Kα。
WAVELENGTHS_A = {
    "cu": 1.5406,
    "co": 1.78897,
    "mo": 0.71073,
    "fe": 1.9373,
    "cr": 2.2909,
    "ag": 0.5609,
}


def build_calib_prompt(image_name: str, context: str | None) -> str:
    schema = {
        "is_xrd": True,
        "x_axis": {
            "label": "2θ (°)",
            "unit": "°",
            "x_quantity": "2theta",
            "x_range": [20, 90],
            "ticks": [
                {"value": 20, "x_1000": 160},
                {"value": 30, "x_1000": 280},
            ],
        },
        "y_axis": {"label": "Intensity (a.u.)"},
        "plot_box_1000": [80, 40, 920, 780],
        "curve_count": 6,
        "curves": [
            {
                "name": "图例或面板上的样品名",
                "color": "黑色",
                "color_rgb": [0, 0, 0],
                "style": "line",
                "stack_order": 1,
            }
        ],
        "legend": "图例文字摘要，没有则 null",
        "radiation": "Cu Kα 或 Co Kα，图上或背景能判断才填，否则 null",
        "target": "Cu 或 Co，否则 null",
        "wavelength_angstrom": 1.5406,
        "mask_boxes_1000": [
            {"box_1000": [80, 50, 260, 160], "kind": "legend"},
            {"box_1000": [200, 620, 520, 700], "kind": "label"},
        ],
        "notes": "其它有助于定标或分线的观察",
    }
    lines = [
        "请阅读这张已经裁出的 X 射线衍射图，提取用于数字化曲线的定标信息。",
        f"图片文件名：{image_name}",
        "",
        "坐标约定：所有位置都相对于【本张裁图】，归一化到 0–1000，原点在左上角。",
        "x_1000 是横坐标，y_1000 是纵坐标（向下为正）。",
        "",
        "x 轴刻度（最重要）：",
        "- 至少读出 2 个、能读几个给几个刻度数字，优先两端和中间。",
        "- x_1000 必须对准刻度【短竖线/tick mark 的中心】，不要用数字左边缘，也不要只用轴标题位置。",
        "- value 是刻度上的 2θ 数值，单位度，保持数字类型。",
        "- x_range 必须是横轴【两端端点】的 2θ 值，不是第一根已标注刻度。",
        "  若轴从 5° 画到 90°、第一根数字却是 10°，x_range 应为 [5, 90]。",
        "- 同时给出 x_quantity：实验室粉末/薄膜 XRD 填 '2theta'；若横坐标是晶面间距 d、散射矢量 Q、飞行时间 TOF，或这是中子/同步辐射 d 图，分别填 'd'、'Q'、'tof'、'neutron'。",
        "- x 轴单位写在 x_axis.label 和 x_axis.unit（例如 '°' 或 'Å'）。",
        "",
        "曲线：",
        "- curve_count 为实验曲线条数（堆叠的多条谱要分别计数；标准卡片竖线、网格、拟合差谱如能区分请不要算进实验曲线）。",
        "- 每条给出颜色（中文名 + 可选 RGB）、图例/面板名称、线型（line 或 fill）。",
        "- 若为上下堆叠，stack_order 从上到下从 1 编号。",
        "",
        "plot_box_1000 是【主】XRD 数据框（同一套 2θ 坐标、含坐标轴与刻度数字）的外接框 [x0,y0,x1,y1]。",
        "若裁图里还拼了相邻的柱状图、示意图、或 2θ 范围不同的放大图/inset，plot_box_1000 只包主图，不要把它们框进去。",
        "把 2θ 范围不同的放大图、inset、标准卡片竖线密集区、相邻子图残片放进 mask_boxes_1000（kind 用 inset/zoom/legend/label/sticks）。",
        "mask_boxes_1000：需要在描线前遮掉的区域，避免把图例色条/线段、图内化学式或 hkl 文字、相邻子图残片当成曲线。",
        "只框这些标注，千万不要遮住衍射峰、坐标轴和刻度数字。没有则给空数组。",
        "靶材/辐射源：结合图上文字和我提供的图注/背景判断（Cu Kα、Co Kα 等），看不出则填 null，不要猜。",
        "",
        "只输出一个 JSON 对象，不要 Markdown 围栏，不要其它文字。结构如下：",
        json.dumps(schema, ensure_ascii=False, indent=2),
    ]
    if context and context.strip():
        lines.extend(["", "图注与背景（供判断样品名、靶材，但仍须以图像为准）：", context.strip()])
    return "\n".join(lines)


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_ticks(parsed: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    """把模型给出的刻度整理为带像素坐标的列表。"""
    x_axis = parsed.get("x_axis") if isinstance(parsed.get("x_axis"), dict) else {}
    raw = x_axis.get("ticks") or parsed.get("ticks") or []
    ticks: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return ticks
    for item in raw:
        if not isinstance(item, dict):
            continue
        value = _as_float(item.get("value") if "value" in item else item.get("two_theta"))
        x1000 = _as_float(
            item.get("x_1000")
            if item.get("x_1000") is not None
            else item.get("x")
            if item.get("x") is not None
            else item.get("x_norm")
        )
        xpx = _as_float(item.get("x_px") or item.get("pixel_x"))
        if value is None:
            continue
        if xpx is None and x1000 is not None:
            # 若误把 0–1 的小数当成 0–1000
            if 0 <= x1000 <= 1.5:
                x1000 *= 1000
            xpx = x1000 / 1000.0 * width
        elif x1000 is None and xpx is not None:
            x1000 = xpx / max(width, 1) * 1000
        if xpx is None or not (0 <= xpx <= width * 1.05):
            continue
        ticks.append(
            {
                "value": value,
                "x_1000": round(float(x1000), 2) if x1000 is not None else None,
                "x_px": round(float(xpx), 2),
            }
        )
    ticks.sort(key=lambda t: t["x_px"])
    # 去掉几乎重叠的重复读数
    cleaned: list[dict[str, Any]] = []
    for t in ticks:
        if cleaned and abs(t["x_px"] - cleaned[-1]["x_px"]) < 2:
            continue
        cleaned.append(t)
    return cleaned


def infer_wavelength(calib: dict[str, Any], context: str | None = None) -> tuple[float, str]:
    """返回 (波长 Å, 来源说明)。优先用模型给出的数值或靶材名。"""
    direct = _as_float(calib.get("wavelength_angstrom"))
    if direct and 0.3 < direct < 3.5:
        return direct, f"模型读出 {direct} Å"
    blob = " ".join(
        str(x)
        for x in (
            calib.get("radiation"),
            calib.get("target"),
            context or "",
        )
        if x
    ).lower().replace("α", "a").replace("ka", "ka")
    mapping = [
        ("co k", "co"),
        ("cobalt", "co"),
        ("钴靶", "co"),
        ("co靶", "co"),
        ("mo k", "mo"),
        ("钼靶", "mo"),
        ("fe k", "fe"),
        ("铁靶", "fe"),
        ("cr k", "cr"),
        ("铬靶", "cr"),
        ("cu k", "cu"),
        ("copper", "cu"),
        ("铜靶", "cu"),
        ("cu靶", "cu"),
    ]
    for needle, key in mapping:
        if needle in blob:
            return WAVELENGTHS_A[key], f"由「{needle}」判定为 {key.upper()} Kα（{WAVELENGTHS_A[key]} Å）"
    # 单独一个 Co / Cu 单词（避免 copper 里的 cu 已处理）
    if calib.get("target"):
        t = str(calib["target"]).strip().lower()
        if t in WAVELENGTHS_A:
            return WAVELENGTHS_A[t], f"靶材 {t.upper()} Kα（{WAVELENGTHS_A[t]} Å）"
    return WAVELENGTHS_A["cu"], "未读到靶材，默认 Cu Kα 1.5406 Å（请核对立，钴靶论文尤其容易错）"


def calib_summary(calib: dict[str, Any]) -> str:
    ticks = calib.get("ticks") or []
    tick_txt = ", ".join(f"{t['value']:g}°@{t['x_px']:.0f}px" for t in ticks[:10])
    curves = calib.get("curve_count")
    wave = calib.get("wavelength_note") or ""
    return f"刻度 {len(ticks)} 个（{tick_txt}）；曲线约 {curves} 条；{wave}"


def parse_mask_boxes(parsed: dict[str, Any] | None) -> list[dict[str, Any]]:
    """整理模型给出的遮罩框，过滤过大或无效的区域。"""
    if not isinstance(parsed, dict):
        return []
    raw = parsed.get("mask_boxes_1000") or parsed.get("mask_boxes") or []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 4:
            box = [_as_float(v) for v in item]
            kind = "label"
        elif isinstance(item, dict):
            box = item.get("box_1000") or item.get("box") or item.get("bbox")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            box = [_as_float(v) for v in box]
            kind = str(item.get("kind") or "label")
        else:
            continue
        if any(v is None for v in box):
            continue
        x0, x1 = sorted((float(box[0]), float(box[2])))
        y0, y1 = sorted((float(box[1]), float(box[3])))
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(1000.0, x1), min(1000.0, y1)
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0) / 1e6
        # 单框超过画面 18% 或多半是误把数据区框进去了
        if x1 - x0 < 8 or y1 - y0 < 8 or area > 0.18:
            continue
        out.append({"box_1000": [x0, y0, x1, y1], "kind": kind, "area_frac": round(area, 4)})
    # 总面积也不宜太大
    total = sum(b["area_frac"] for b in out)
    if total > 0.35:
        out.sort(key=lambda b: b["area_frac"])
        kept, acc = [], 0.0
        for b in out:
            if acc + b["area_frac"] > 0.35:
                continue
            kept.append(b)
            acc += b["area_frac"]
        out = kept
    return out



def looks_like_non_2theta(calib: dict[str, Any], parsed: dict[str, Any] | None = None) -> str | None:
    """若横坐标不是实验室 2θ，返回原因，否则 None。"""
    parsed = parsed or {}
    x_axis = parsed.get("x_axis") if isinstance(parsed.get("x_axis"), dict) else {}
    if not isinstance(x_axis, dict):
        x_axis = {}
    quantity = str(x_axis.get("x_quantity") or parsed.get("x_quantity") or calib.get("x_quantity") or "").strip().lower()
    label = " ".join(
        str(x or "")
        for x in (
            x_axis.get("label"),
            x_axis.get("unit"),
            calib.get("x_label"),
            parsed.get("notes"),
            calib.get("notes"),
        )
    ).lower()
    blob = quantity + " " + label
    blob = blob.replace("å", "angstrom").replace("Å", "angstrom").replace("α", "a")
    # 明确的非 2θ 坐标系
    if any(k in blob for k in ("tof", "time of flight", "flight time", "飞行时间")):
        return "横坐标是飞行时间 TOF，不是实验室 2θ"
    if any(k in blob for k in ("neutron", "中子衍射", "中子精修")) and "2theta" not in quantity and "2θ" not in label and "2 theta" not in label:
        return "这是中子衍射图，不是实验室 X 射线 2θ 图"
    if quantity in {"d", "d-spacing", "dspacing"} or (
        ("d-spacing" in blob or "d spacing" in blob or "晶面间距" in blob or "d (angstrom" in blob or "d(angstrom" in blob or "d (a)" in blob or "d(a)" in blob or "d / a" in blob or "d/a" in blob)
        and "2θ" not in label and "2theta" not in quantity and "2 theta" not in label
    ):
        return "横坐标是晶面间距 d，不是 2θ"
    if quantity in {"q", "q-space"} or (any(k in blob for k in (" q (", "q (a", "q/a", "scattering vector", "散射矢量")) and "2θ" not in label):
        return "横坐标是散射矢量 Q，不是 2θ"
    # 数值启发：典型 d/Q 轴在 0–8，且标签含 Å 而不含 2θ
    rng = calib.get("x_range") or x_axis.get("x_range")
    try:
        lo, hi = float(rng[0]), float(rng[1]) if rng and len(rng) >= 2 else (None, None)
    except (TypeError, ValueError, IndexError):
        lo, hi = None, None
    if lo is not None and hi is not None and 0 <= lo < hi <= 8:
        has_angstrom = any(k in blob for k in ("angstrom", "a-1", "a^-1", "nm-1", "Å", "å"))
        has_2th = any(k in blob for k in ("2θ", "2theta", "2-theta", "2 theta", "two-theta"))
        if has_2th:
            return None
        if has_angstrom:
            return f"横轴范围 {lo:g}–{hi:g} 且单位像 Å/Q，不像实验室 2θ"
        if quantity and quantity not in {"2theta", "2-theta", "two-theta", "deg", "degree", "degrees"}:
            return f"横轴范围 {lo:g}–{hi:g}，x_quantity={quantity}，不像实验室 2θ"
        # 实验室粉末 2θ 几乎不会整个落在 0–8°；无 2θ 标签时按 d/Q 拒绝
        return f"横轴范围 {lo:g}–{hi:g}，不像实验室 2θ（更像 d 或 Q）"
    return None


def read_calibration(
    image_path: Path,
    *,
    model: str,
    api_key: str,
    context: str | None,
    max_side: int,
    max_retries: int,
    thinking: bool,
    dry_run: bool = False,
    timeout: float = 180.0,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """对裁图调用视觉模型，返回定标字典（含 ticks、curves、usage）。"""
    with Image.open(image_path) as im:
        width, height = im.size
    prompt = build_calib_prompt(image_path.name, context)
    result: dict[str, Any] = {
        "status": "",
        "width": width,
        "height": height,
        "ticks": [],
        "curves": [],
        "curve_count": None,
        "usage": None,
        "error": "",
        "raw_parsed": None,
    }
    if dry_run:
        result["status"] = "dry-run"
        return result

    encoded = encode_image(image_path, max_side=max_side)
    raw, usage = call_model(
        encoded["data_url"],
        prompt,
        model=model,
        api_key=api_key,
        max_retries=max_retries,
        thinking=thinking,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    result["usage"] = usage
    parsed, err = extract_json(raw)
    if parsed is None:
        result["status"] = "parse_failed"
        result["error"] = err or "定标 JSON 解析失败"
        return result
    result["raw_parsed"] = parsed
    if parsed.get("is_xrd") is False:
        result["status"] = "not_xrd"
        result["error"] = "定标模型认为这不是 XRD 图"
        return result
    ticks = parse_ticks(parsed, width, height)
    result["ticks"] = ticks
    x_axis = parsed.get("x_axis") if isinstance(parsed.get("x_axis"), dict) else {}
    rng = x_axis.get("x_range") or parsed.get("x_range")
    if isinstance(rng, (list, tuple)) and len(rng) >= 2:
        a, b = _as_float(rng[0]), _as_float(rng[1])
        result["x_range"] = [a, b] if a is not None and b is not None else None
    else:
        result["x_range"] = None
    result["x_label"] = x_axis.get("label")
    result["y_label"] = (parsed.get("y_axis") or {}).get("label") if isinstance(parsed.get("y_axis"), dict) else None
    result["plot_box_1000"] = parsed.get("plot_box_1000")
    result["curve_count"] = parsed.get("curve_count")
    result["curves"] = parsed.get("curves") if isinstance(parsed.get("curves"), list) else []
    result["legend"] = parsed.get("legend")
    result["radiation"] = parsed.get("radiation")
    result["target"] = parsed.get("target")
    wave, note = infer_wavelength(parsed, context)
    result["wavelength_angstrom"] = wave
    result["wavelength_note"] = note
    result["notes"] = parsed.get("notes")
    result["mask_boxes_1000"] = parse_mask_boxes(parsed)
    result["x_quantity"] = (x_axis.get("x_quantity") if isinstance(x_axis, dict) else None) or parsed.get("x_quantity")
    reason = looks_like_non_2theta(result, parsed)
    if reason:
        result["status"] = "not_2theta"
        result["error"] = reason
        result["summary"] = reason
        return result
    if len(ticks) < 2:
        result["status"] = "need_ticks"
        result["error"] = f"有效 x 轴刻度不足 2 个（得到 {len(ticks)} 个），无法定标"
        return result
    result["status"] = "ok"
    result["summary"] = calib_summary(result)
    return result



def apply_zero_trace_review(review: dict[str, Any] | None, xy_files: list) -> dict[str, Any]:
    """只要有一条导出的 .xy 强度全为零，复核必须为 fail。"""
    review = dict(review or {})
    zeros: list[str] = []
    import numpy as np
    from pathlib import Path as _P
    for raw in xy_files or []:
        path = _P(raw)
        if not path.is_file():
            continue
        ys = []
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) >= 2:
                    ys.append(float(parts[1]))
        except Exception:
            continue
        if not ys or float(max(ys)) <= 1e-9:
            zeros.append(path.name)
    if zeros:
        review["verdict"] = "fail"
        findings = [f"{len(zeros)} 条曲线强度全为零：{', '.join(zeros[:6])}"]
        findings.extend(str(x) for x in (review.get("findings") or []) if x)
        review["findings"] = findings[:8]
        review["zero_xy"] = zeros
    return review


def review_overlay(
    original: Path,
    overlay: Path,
    *,
    model: str,
    api_key: str,
    expected_curves: int | None,
    max_side: int,
    max_retries: int,
    thinking: bool,
    timeout: float = 25.0,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """让模型对比原图与叠加校验图，类似工作台 vision_review。"""
    prompt = (
        "第一张是原始 XRD 裁图，第二张是在原图几何上叠加了程序描出的曲线。"
        "请比较：是否漏描、错位、把标注/网格/卡片竖线当成曲线、或多描。"
        f"{'图上实验曲线大约应有 ' + str(expected_curves) + ' 条。' if expected_curves else ''}"
        "只输出 JSON："
        '{"trace_count": 6, "verdict": "pass 或 review 或 fail 或 uncertain",'
        '"findings": ["简短中文发现"], "ok": true}'
        "。verdict：基本一致为 pass，有明显缺陷为 review，出现全零/塌缩描线或严重错位为 fail，看不清为 uncertain。若某条实验曲线被描成一条零强度直线，必须判 fail。"
        "不要输出其它文字。"
    )
    enc1 = encode_image(original, max_side=max_side)
    enc2 = encode_image(overlay, max_side=max_side)
    raw, usage = call_model(
        [enc1["data_url"], enc2["data_url"]],
        prompt,
        model=model,
        api_key=api_key,
        max_retries=max_retries,
        thinking=thinking,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    parsed, err = extract_json(raw)
    out: dict[str, Any] = {"usage": usage, "error": err or "", "verdict": "uncertain", "findings": [], "trace_count": None}
    if not parsed:
        out["error"] = err or "复核解析失败"
        return out
    verdict = parsed.get("verdict")
    out["verdict"] = verdict if verdict in {"pass", "review", "fail", "uncertain"} else "uncertain"
    findings = parsed.get("findings")
    if isinstance(findings, str):
        findings = [findings]
    out["findings"] = [str(x)[:300] for x in findings[:8]] if isinstance(findings, list) else []
    try:
        tc = int(parsed["trace_count"]) if parsed.get("trace_count") is not None else None
        out["trace_count"] = tc
    except (TypeError, ValueError):
        pass
    return out
