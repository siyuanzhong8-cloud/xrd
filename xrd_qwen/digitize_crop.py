#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 2 步：用模型读出的刻度把裁图数字化成 .xy，并画出重绘与叠加校验图。

优先尝试工作台 color_traces 的分格面板描线；失败则调用 vendor/digitize_xrd.py
（必须 --refs none）。不再依赖 Windows OCR。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VENDOR = Path(__file__).resolve().parent / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))


def snap_ticks_to_marks(crop_path: Path, ticks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """用程序在底轴检测到的刻度短线中心，替换模型给的像素位置。

    模型读数字通常准，但像素常整体偏掉。若检测到的短线比读出的刻度多，
    且读数是等差数列，则按同样步长向外延拓（例如读到 20–80，图上还有 90）。
    """
    if len(ticks) < 2:
        return ticks
    try:
        sys.path.insert(0, str(VENDOR))
        from digitize_xrd import auto_frame, detect_ticks, load_ink
    except Exception:
        return ticks
    try:
        ink, _gray, _w, _h = load_ink(str(crop_path))
        xl, xr, _yt, yb = auto_frame(ink)
        marks = [float(c) for c, _a, _b in detect_ticks(ink, yb, xl, xr)]
    except Exception:
        return ticks
    values = [float(t["value"]) for t in ticks]
    if len(marks) < 2:
        return ticks

    import numpy as np

    n = len(values)
    yv = np.array(values, dtype=float)
    best = None
    if len(marks) >= n:
        for s in range(len(marks) - n + 1):
            win = np.array(marks[s : s + n])
            A = np.column_stack([win, np.ones(n)])
            (b, a), *_ = np.linalg.lstsq(A, yv, rcond=None)
            resid = float(np.max(np.abs(A @ np.array([b, a]) - yv)))
            if best is None or resid < best[0]:
                best = (resid, s, win, float(b), float(a))
    if best is None or best[0] > 0.6:
        return ticks

    resid, start, win, b, a = best
    # 窗口两侧若还有短线，按读数的中位步长延拓
    steps = np.diff(yv)
    step = float(np.median(steps)) if len(steps) else 0.0
    xs = list(marks)
    vs: list[float] = [None] * len(marks)  # type: ignore
    for i, val in enumerate(values):
        vs[start + i] = val
    if abs(step) >= 0.5:
        for i in range(start - 1, -1, -1):
            vs[i] = vs[i + 1] - step
        for i in range(start + n, len(marks)):
            vs[i] = vs[i - 1] + step
    snapped = []
    for x, v in zip(xs, vs):
        if v is None:
            continue
        snapped.append({"value": float(v), "x_px": float(x), "x_1000": None, "snapped": True})
    return snapped or ticks


def fit_anchors(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """线性定标：2θ = a + b * x_px。至少 2 个刻度。"""
    if len(ticks) < 2:
        raise ValueError("定标至少需要 2 个 x 轴刻度")
    xs = np.array([float(t["x_px"]) for t in ticks], dtype=float)
    ys = np.array([float(t["value"]) for t in ticks], dtype=float)
    if np.ptp(xs) < 5 or np.ptp(ys) < 1e-6:
        raise ValueError("刻度的像素位置或读数几乎没有跨度，无法定标")
    A = np.column_stack([xs, np.ones_like(xs)])
    (b, a), *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = b * xs + a
    resid = pred - ys
    max_res = float(np.max(np.abs(resid)))
    return {
        "a": float(a),
        "b": float(b),
        "max_residual_deg": max_res,
        "residuals": [float(v) for v in resid],
        "px_per_deg": float(abs(1.0 / b)) if b else None,
        "ok": max_res <= 0.4,
    }


def _curve_name(calib: dict[str, Any], index: int, total: int) -> str:
    curves = calib.get("curves") if isinstance(calib.get("curves"), list) else []
    # 优先按 stack_order 从上到下对应 digitize 的 trace 顺序（同样从上到下）
    ordered = []
    for c in curves:
        if isinstance(c, dict):
            ordered.append(c)
    if ordered and all(isinstance(c.get("stack_order"), (int, float)) for c in ordered):
        ordered = sorted(ordered, key=lambda c: float(c.get("stack_order") or 0))
    if 0 <= index < len(ordered) and ordered[index].get("name"):
        return str(ordered[index]["name"]).strip()
    return f"曲线{index + 1}"


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    return cleaned.strip("._") or "trace"


def _load_xy_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        except ValueError:
            continue
    if len(xs) < 5:
        raise ValueError(f"{path.name} 点数不足")
    x, y = np.asarray(xs), np.asarray(ys)
    order = np.argsort(x)
    return x[order], y[order]


def _list_trace_xy(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    # 兼容旧版相对路径把 traces_xy 套了一层的情况
    cands = [folder]
    if folder.is_dir():
        cands.extend(sorted(folder.rglob("traces_xy")))
    files: list[Path] = []
    seen = set()
    for base in cands:
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir() if base.name == "traces_xy" or base == folder else []):
            if p.suffix.lower() == ".xy" and "_grid" not in p.stem and p.resolve() not in seen:
                seen.add(p.resolve())
                files.append(p)
        if base.name != "traces_xy":
            for p in sorted(base.glob("*.xy")):
                if "_grid" not in p.stem and p.resolve() not in seen:
                    seen.add(p.resolve())
                    files.append(p)
    return files


def try_framed_panels(
    crop_path: Path,
    run_dir: Path,
    slope: float,
    intercept: float,
    wavelength: float,
) -> bool:
    try:
        from color_traces import extract_framed_panels
    except Exception:
        return False
    try:
        return bool(
            extract_framed_panels(
                crop_path,
                run_dir,
                slope,
                intercept,
                wavelength,
                0.0,
                2.5,
            )
        )
    except Exception:
        return False


def run_digitize_xrd(
    crop_path: Path,
    run_dir: Path,
    ticks: list[dict[str, Any]],
    wavelength: float,
    auto_ticks: bool = False,
) -> dict[str, Any]:
    """调用 vendor/digitize_xrd.py，禁用默认 TiO2 参考峰。"""
    script = VENDOR / "digitize_xrd.py"
    # 必须用绝对路径：digitize_xrd 若在 run_dir 下再接到相对 outdir，会把结果套一层
    crop_path = Path(crop_path).resolve()
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    anchors = ",".join(f"{t['x_px']:.2f}:{t['value']:.6g}" for t in ticks)
    cmd = [
        sys.executable,
        str(script),
        "--img",
        str(crop_path),
        "--anchors",
        anchors,
        "--lamb",
        str(wavelength),
        "--refs",
        "none",
        "--outdir",
        str(run_dir),
        "--resample",
        "0.05",
    ]
    if auto_ticks:
        cmd.append("--auto-ticks")
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    proc = subprocess.run(
        cmd,
        cwd=str(run_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=180,
    )
    log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    (run_dir / "digitize_xrd.log").write_text(log, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"digitize_xrd 退出码 {proc.returncode}。"
            f"日志已写入 {run_dir / 'digitize_xrd.log'}"
        )
    return {"log": log, "auto_ticks": auto_ticks}


def _curve_color(calib: dict[str, Any], index: int) -> tuple[float, float, float]:
    palette = [
        (0.10, 0.10, 0.10),
        (0.75, 0.15, 0.15),
        (0.15, 0.45, 0.80),
        (0.15, 0.60, 0.25),
        (0.70, 0.20, 0.70),
        (0.90, 0.50, 0.10),
        (0.20, 0.60, 0.60),
        (0.50, 0.40, 0.10),
    ]
    curves = calib.get("curves") if isinstance(calib.get("curves"), list) else []
    if 0 <= index < len(curves) and isinstance(curves[index], dict):
        rgb = curves[index].get("color_rgb")
        if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
            try:
                r, g, b = [float(v) for v in rgb[:3]]
                if max(r, g, b) > 1.5:
                    r, g, b = r / 255.0, g / 255.0, b / 255.0
                return (r, g, b)
            except (TypeError, ValueError):
                pass
    return palette[index % len(palette)]


def write_plots(
    crop_path: Path,
    traces: list[dict[str, Any]],
    run_dir: Path,
    calib: dict[str, Any],
    fit: dict[str, Any],
) -> dict[str, Path]:
    """重绘图 + 原图叠加校验图 + 刻度位置示意。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    # 中文字体：能用则用，没有就回退
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]

    replot = run_dir / "replot.png"
    validate = run_dir / "overlay_check.png"
    tick_preview = run_dir / "calib_ticks.png"

    n = max(1, len(traces))
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.8 * n + 0.8), sharex=True)
    if n == 1:
        axes = [axes]
    for i, tr in enumerate(traces):
        ax = axes[i]
        color = _curve_color(calib, i)
        ax.plot(tr["x"], tr["y"], color=color, lw=0.9, label=tr["name"])
        ax.set_ylabel(tr["name"], fontsize=8)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
        ymax = float(np.max(tr["y"])) if len(tr["y"]) else 1
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
    axes[-1].set_xlabel(r"$2\theta$ (°)")
    title = crop_path.name
    if fit.get("max_residual_deg") is not None:
        title += f"  定标残差 ≤ {fit['max_residual_deg']:.3f}°"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(replot, dpi=140)
    plt.close(fig)

    with Image.open(crop_path) as im:
        rgb = np.asarray(im.convert("RGB"))
        height, width = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 10 * height / max(width, 1)))
    ax.imshow(rgb)
    a, b = fit["a"], fit["b"]
    for i, tr in enumerate(traces):
        if b == 0:
            continue
        xs = (tr["x"] - a) / b
        # 在原图上只画曲线的「高度」需要原始像素 y；若没有 apex，就只画 x 对应的一条示意
        if tr.get("y_px") is not None:
            ax.plot(xs, tr["y_px"], color=_curve_color(calib, i), lw=1.0, alpha=0.85, label=tr["name"])
        else:
            # 没有像素 y 时，把强度映射到图框中下部作示意是误导，不如只标名称
            pass
    if any(tr.get("y_px") is not None for tr in traces):
        ax.legend(loc="upper right", fontsize=7, frameon=True, fancybox=False)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    ax.set_title("提取曲线叠加在原图上（校验用）", fontsize=10)
    fig.tight_layout()
    fig.savefig(validate, dpi=140)
    plt.close(fig)

    # 刻度位置：若没有像素 y，仍在原图上画出 tick 竖线，便于检查定标
    preview = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(preview)
    for t in calib.get("ticks") or []:
        x = float(t["x_px"])
        draw.line([(x, 0), (x, height)], fill=(220, 30, 30), width=1)
        draw.text((x + 2, 4), f"{t['value']:g}", fill=(220, 30, 30))
    preview.save(tick_preview)

    return {"replot": replot, "validate": validate, "ticks": tick_preview}


def _traces_from_xy_dir(
    xy_dir: Path,
    calib: dict[str, Any],
    validate_png: Path | None,
) -> list[dict[str, Any]]:
    traces = []
    files = _list_trace_xy(xy_dir)
    # 尝试从 digitize_xrd / color_traces 的 validate.png 里没有 apex；
    # 若存在原始 validate，稍后再叠加我们自己的图。
    for i, path in enumerate(files):
        x, y = _load_xy_pairs(path)
        traces.append({"name": _curve_name(calib, i, len(files)), "x": x, "y": y, "path": path, "y_px": None})
    return traces


def _try_load_apex_from_validate_log(run_dir: Path) -> None:
    """占位：digitize_xrd 的 apex 没有单独导出。校验图以它自带的 validate.png 为底。"""
    return None


def export_named_xy(traces: list[dict[str, Any]], dest_dir: Path, stem: str) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, tr in enumerate(traces):
        name = _safe_name(tr["name"])
        path = dest_dir / f"{stem}_{i:02d}_{name}.xy"
        with path.open("w", encoding="utf-8") as f:
            f.write(f"# {tr['name']}\n# 2theta  Intensity  (digitized from figure)\n")
            for xx, yy in zip(tr["x"], tr["y"]):
                f.write(f"{xx:.5f} {yy:.4f}\n")
        tr["export_path"] = path
        out.append(path)
    return out


def apply_mask_boxes(crop_path: Path, boxes: list[dict[str, Any]] | None, dest: Path) -> Path | None:
    """把图例/文字框涂成白色，避免描线把标注当成曲线。框过大时调用方应已过滤。"""
    if not boxes:
        return None
    crop_path = Path(crop_path)
    with Image.open(crop_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        draw = ImageDraw.Draw(im)
        applied = 0
        for item in boxes:
            box = (item or {}).get("box_1000") if isinstance(item, dict) else item
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                x0, y0, x1, y1 = (float(v) for v in box)
            except (TypeError, ValueError):
                continue
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            px = [int(round(x0 / 1000.0 * w)), int(round(y0 / 1000.0 * h)),
                  int(round(x1 / 1000.0 * w)), int(round(y1 / 1000.0 * h))]
            if px[2] - px[0] < 2 or px[3] - px[1] < 2:
                continue
            draw.rectangle(px, fill=(255, 255, 255))
            applied += 1
        if not applied:
            return None
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, format="PNG")
        return dest



def _usable_plot_box(box) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(1000.0, x1), min(1000.0, y1)
    w, h = x1 - x0, y1 - y0
    # 必须明显小于整幅裁图，否则没有「相邻子图」可切
    if w < 200 or h < 200 or w > 850 or h > 920:
        return None
    return [x0, y0, x1, y1]


def _outside_plot_masks(plot_box: list[float], pad: float = 8.0) -> list[dict]:
    x0, y0, x1, y1 = plot_box
    x0, y0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
    x1, y1 = min(1000.0, x1 + pad), min(1000.0, y1 + pad)
    masks = []
    if x0 > 20:
        masks.append({"box_1000": [0.0, 0.0, x0, 1000.0], "kind": "outside-plot"})
    if x1 < 980:
        masks.append({"box_1000": [x1, 0.0, 1000.0, 1000.0], "kind": "outside-plot"})
    if y0 > 20:
        masks.append({"box_1000": [x0, 0.0, x1, y0], "kind": "outside-plot"})
    if y1 < 980:
        masks.append({"box_1000": [x0, y1, x1, 1000.0], "kind": "outside-plot"})
    return masks


def _ticks_disagree_with_plot(ticks: list[dict], plot_box: list[float]) -> bool:
    xs = []
    for t in ticks or []:
        x = t.get("x_1000")
        if x is None and t.get("x_px") is not None:
            continue
        try:
            xs.append(float(x))
        except (TypeError, ValueError):
            continue
    if len(xs) < 2:
        return False
    if max(xs) > 1005 or min(xs) < -5:
        return True
    # 超过 25% 的刻度落在数据框横向范围外，认为定标被相邻子图拉歪
    outside = sum(1 for x in xs if x < plot_box[0] - 20 or x > plot_box[2] + 20)
    return outside >= max(1, int(0.25 * len(xs)))



def format_digitized_peak_table(fig: dict) -> str | None:
    """把程序描点得到的峰位格式化为识图用的峰表。精确 2θ 以描点为准。"""
    dig = fig.get("digitize") or {}
    peaks = dig.get("peaks") if isinstance(dig.get("peaks"), dict) else None
    lines = [
        "以下峰位由程序从图像曲线数字化得到，2θ 已按坐标轴刻度线性定标，请把它当作精确峰位；",
        "不要用读图目测的峰位覆盖本表。图上的 hkl、图例、物相标记只用于解释。",
    ]
    any_p = False
    if peaks:
        for name, plist in peaks.items():
            lines.append(f"曲线 {name}:")
            if not plist:
                lines.append("  （未检出显著峰）")
                continue
            any_p = True
            imax = max((p.get("intensity") or 0) for p in plist) or 1
            for p in plist:
                tt = p.get("two_theta")
                I = p.get("intensity") or 0
                rel = 100.0 * float(I) / float(imax)
                lines.append(f"  2θ={float(tt):.2f}°  I={float(I):.1f}  相对强度≈{rel:.0f}")
    else:
        # 回退：从 xy 再找一次
        try:
            from trace_xrd import find_trace_peaks
        except Exception:
            find_trace_peaks = None
        for path in dig.get("xy_files") or []:
            path = Path(path)
            if not path.is_file():
                continue
            try:
                x, y = _load_xy_pairs(path)
            except Exception:
                continue
            name = path.stem
            if find_trace_peaks is None:
                continue
            plist = find_trace_peaks(x, y)
            lines.append(f"曲线 {name}:")
            if not plist:
                lines.append("  （未检出显著峰）")
                continue
            any_p = True
            imax = max((p.get("intensity") or 0) for p in plist) or 1
            for p in plist:
                rel = 100.0 * float(p["intensity"]) / float(imax)
                lines.append(f"  2θ={p['two_theta']:.2f}°  I={p['intensity']:.1f}  相对强度≈{rel:.0f}")
    flags = dig.get("flags") or []
    if flags:
        lines.append("定标/描点备注：" + "；".join(str(f) for f in flags))
    if not any_p:
        return None
    return "\n".join(lines)


def digitize_figure(
    crop_path: Path,
    calib: dict[str, Any],
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """对一张裁图执行描线，写出 .xy、重绘图和叠加校验图。"""
    crop_path = Path(crop_path)
    run_dir = Path(out_dir) / "digitize" / crop_path.stem
    result: dict[str, Any] = {
        "status": "",
        "method": "",
        "run_dir": str(run_dir),
        "xy_files": [],
        "replot": "",
        "validate": "",
        "ticks_preview": "",
        "fit": None,
        "n_traces": 0,
        "error": "",
        "trace_names": [],
    }
    ticks = calib.get("ticks") or []
    xr = calib.get("x_range")
    if len(ticks) < 2 and not (isinstance(xr, (list, tuple)) and len(xr) >= 2):
        xa = calib.get("x_axis") if isinstance(calib.get("x_axis"), dict) else {}
        xr = xa.get("x_range")
        if isinstance(xr, (list, tuple)) and len(xr) >= 2:
            calib["x_range"] = list(xr)
    if len(ticks) < 2 and not (isinstance(calib.get("x_range"), (list, tuple)) and len(calib.get("x_range") or []) >= 2):
        result["status"] = "failed"
        result["error"] = "缺少至少 2 个有效刻度或 x_range，无法数字化"
        return result
    # 若还没有 x_range，用刻度读数补上，便于识别 d/Q 轴
    if not (isinstance(calib.get("x_range"), (list, tuple)) and len(calib.get("x_range") or []) >= 2):
        vals = [float(t["value"]) for t in ticks if t.get("value") is not None]
        if len(vals) >= 2:
            calib["x_range"] = [min(vals), max(vals)]
    try:
        from calib_read import looks_like_non_2theta
        reason = looks_like_non_2theta(calib, calib.get("raw_parsed") if isinstance(calib.get("raw_parsed"), dict) else {})
    except Exception:
        reason = None
    if reason:
        result["status"] = "not_2theta"
        result["error"] = reason
        return result

    if run_dir.exists() and not overwrite and _list_trace_xy(run_dir / "traces_xy") and (run_dir / "replot.png").is_file():
        traces = _traces_from_xy_dir(run_dir / "traces_xy", calib, None)
        result.update(
            {
                "status": "skipped",
                "method": "cached",
                "n_traces": len(traces),
                "trace_names": [t["name"] for t in traces],
                "xy_files": [str(p) for p in (t.get("export_path") or t["path"] for t in traces)],
                "replot": str(run_dir / "replot.png"),
                "validate": str(run_dir / "overlay_check.png") if (run_dir / "overlay_check.png").is_file() else str(run_dir / "validate.png"),
            }
        )
        # 重新导出一份带名字的 xy 到 run_dir 上一级的 xy/
        exported = export_named_xy(traces, Path(out_dir) / "xy", crop_path.stem)
        result["xy_files"] = [str(p) for p in exported]
        return result

    if run_dir.exists() and overwrite:
        # 只清本图的数字化目录，避免误删其它图
        import shutil

        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 优先走程序化框内描点（精确 2θ 来自刻度短线，不依赖模型读峰）
    try:
        from trace_xrd import digitize_in_frame
        tr = digitize_in_frame(crop_path, calib, run_dir, mask=True)
        traces = tr["traces"]
        fit = tr["fit"]
        plots = write_plots(crop_path, traces, run_dir, calib, fit)
        # 框内叠加图更准确，作为主 validate；保留 write_plots 的 overlay_check / replot
        validate_path = Path(tr["validate"]) if tr.get("validate") else plots["validate"]
        exported = export_named_xy(traces, Path(out_dir) / "xy", crop_path.stem)
        result.update(
            {
                "status": "ok",
                "method": "frame_trace",
                "n_traces": len(traces),
                "trace_names": [t["name"] for t in traces],
                "xy_files": [str(p) for p in exported],
                "replot": str(plots["replot"]),
                "validate": str(validate_path),
                "ticks_preview": str(plots["ticks"]),
                "used_auto_ticks": False,
                "max_residual_deg": fit.get("max_residual_deg"),
                "fit": fit,
                "frame": tr.get("frame"),
                "peak_check": tr.get("peak_check"),
                "flags": tr.get("flags") or [],
                "peaks": {t["name"]: t.get("peaks") or [] for t in traces},
            }
        )
        return result
    except Exception as exc:
        result["frame_trace_error"] = str(exc)
        # 回退到旧的 vendor 描线

    work_path = crop_path
    masks = list(calib.get("mask_boxes_1000") or calib.get("mask_boxes") or [])
    plot_box = calib.get("plot_box_1000")
    tight = _usable_plot_box(plot_box)
    if tight:
        # 把数据框以外（相邻柱状图、拼图残片）涂白，避免拉歪 2θ 轴
        masks = list(masks) + _outside_plot_masks(tight)
        result["plot_box_1000"] = tight
    if masks:
        masked = apply_mask_boxes(crop_path, masks, run_dir / "masked.png")
        if masked is not None:
            work_path = masked
            result["masked"] = str(masked)
            result["n_masks"] = len(masks)

    ticks = snap_ticks_to_marks(work_path, ticks)
    # 若刻度横跨了相邻子图（x_1000 超出数据框或越出 0–1000），改用 x_range 对齐数据框左右边
    if tight and _ticks_disagree_with_plot(ticks, tight):
        rng = calib.get("x_range")
        if not (isinstance(rng, (list, tuple)) and len(rng) >= 2):
            vals = [float(t["value"]) for t in ticks if t.get("value") is not None]
            rng = [min(vals), max(vals)] if len(vals) >= 2 else None
        if rng:
            calib["x_range"] = [float(rng[0]), float(rng[1])]
            # 强制走下面的 auto_frame 回退（len(ticks)<=3 条件）
            ticks = ticks[:2] if len(ticks) >= 2 else ticks
            result["tick_source"] = "plot_box_x_range"
    # 若只能检测到左右边框、短刻度线不够，优先用模型读出的 x_range 对准框图两端
    rng = calib.get("x_range")
    if isinstance(rng, (list, tuple)) and len(rng) >= 2 and len(ticks) <= 3:
        try:
            sys.path.insert(0, str(VENDOR))
            from digitize_xrd import auto_frame, load_ink
            ink, _g, _w, _h = load_ink(str(work_path))
            xl, xr, _yt, _yb = auto_frame(ink)
            ticks = [
                {"value": float(rng[0]), "x_px": float(xl), "x_1000": None},
                {"value": float(rng[1]), "x_px": float(xr), "x_1000": None},
            ]
        except Exception:
            pass
    calib = dict(calib)
    calib["ticks"] = ticks
    try:
        fit = fit_anchors(ticks)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"刻度定标失败：{exc}"
        return result
    result["fit"] = {k: v for k, v in fit.items() if k != "residuals"} | {"residuals": fit["residuals"]}
    (run_dir / "calib.json").write_text(
        json.dumps({"ticks": ticks, "fit": result["fit"], "calib": {k: calib.get(k) for k in (
            "wavelength_angstrom", "wavelength_note", "radiation", "target",
            "curve_count", "legend", "x_label", "notes", "mask_boxes_1000",
            "plot_box_1000", "x_range",
        )}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    wavelength = float(calib.get("wavelength_angstrom") or 1.5406)
    method = ""
    used_auto = False
    try:
        if try_framed_panels(work_path, run_dir, fit["b"], fit["a"], wavelength):
            method = "framed_panels"
        else:
            # 模型读刻度数字较准，但像素位置经常整体偏了却仍「线性很好」。
            # 因此默认用程序检测的刻度短线中心来定像素，只用模型给出的数值。
            try:
                run_digitize_xrd(work_path, run_dir, ticks, wavelength, auto_ticks=True)
                method = "digitize_xrd_auto_ticks"
                used_auto = True
            except Exception as first:
                run_digitize_xrd(work_path, run_dir, ticks, wavelength, auto_ticks=False)
                method = "digitize_xrd"
                used_auto = False
    except Exception as exc:
        result["status"] = "failed"
        result["method"] = method
        result["error"] = f"描线失败：{exc}"
        (run_dir / "digitize_error.txt").write_text(
            result["error"] + "\n" + traceback.format_exc(), encoding="utf-8"
        )
        return result

    xy_dir = run_dir / "traces_xy"
    traces = _traces_from_xy_dir(xy_dir, calib, run_dir / "validate.png")
    if not traces:
        result["status"] = "failed"
        result["method"] = method
        result["error"] = "描线完成但没有写出可用的 .xy"
        return result

    # 若工作台已写出 validate.png，先保留；我们再出一份带图例名的 overlay_check.png
    plots = write_plots(crop_path, traces, run_dir, calib, fit)
    # 工作台 validate.png 往往带有像素轨迹，优先作为叠加校验图，同时保留我们的 replot
    wb_validate = run_dir / "validate.png"
    if wb_validate.is_file() and method != "framed_panels":
        # digitize_xrd 的 validate 是真正按 apex 画在原图上的，更适合做校验
        # 仍保留 overlay_check；若 write_plots 没有 y_px，用工作台那张作为主校验图
        validate_path = wb_validate
    elif wb_validate.is_file() and method == "framed_panels":
        validate_path = wb_validate
    else:
        validate_path = plots["validate"]

    exported = export_named_xy(traces, Path(out_dir) / "xy", crop_path.stem)
    result.update(
        {
            "status": "ok",
            "method": method,
            "n_traces": len(traces),
            "trace_names": [t["name"] for t in traces],
            "xy_files": [str(p) for p in exported],
            "replot": str(plots["replot"]),
            "validate": str(validate_path),
            "ticks_preview": str(plots["ticks"]),
            "used_auto_ticks": used_auto,
            "max_residual_deg": fit["max_residual_deg"],
        }
    )
    return result
