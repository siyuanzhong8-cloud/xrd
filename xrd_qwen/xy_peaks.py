#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 XRD 原始谱图数据（.xy / .txt / .csv）中寻峰。

有仪器导出的 2θ–强度数据时，用本脚本读峰通常比视觉模型读图更准确。
既可单独命令行使用，也可被 xrd_vl.py 导入，把峰表一并交给模型。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# 尝试导入可选依赖；在仅做数据加载的场景下给出更友好的报错
try:
    from scipy.signal import find_peaks
except ImportError:  # pragma: no cover
    find_peaks = None

DATA_SUFFIXES = (".xy", ".txt", ".csv", ".dat", ".xye")


def load_xy(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取两列 2θ、强度数据，自动跳过表头、注释和空行。

    支持逗号、分号、空白（空格/制表符）分隔，以及科学计数法。
    若有多于两列，默认取前两列（第 1 列 2θ，第 2 列强度）。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    two_theta: list[float] = []
    intensity: list[float] = []
    skipped = 0

    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # 常见注释：#、//、!、% 以及以文字开头的表头
            if line[0] in "#!%;" or line.startswith("//"):
                skipped += 1
                continue
            # 统一分隔符后再拆
            normalized = line.replace(";", " ").replace(",", " ").replace("\t", " ")
            parts = [p for p in normalized.split() if p]
            if len(parts) < 2:
                skipped += 1
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                skipped += 1
                continue
            two_theta.append(x)
            intensity.append(y)

    if len(two_theta) < 5:
        raise ValueError(
            f"{path.name} 有效数据点过少（{len(two_theta)} 行），"
            "请确认是两列 2θ、强度，且不是纯表头文件。"
        )

    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)
    # 若 2θ 不是单调递增，按 2θ 排序，避免后续寻峰/半高峰宽出错
    order = np.argsort(x)
    x, y = x[order], y[order]
    return x, y


def _degree_to_index_distance(two_theta: np.ndarray, min_distance_deg: float) -> int:
    """把最小峰间距（单位：°2θ）换算成 find_peaks 所需的点距。"""
    if len(two_theta) < 2:
        return 1
    step = float(np.median(np.diff(two_theta)))
    if step <= 0:
        return 1
    return max(1, int(round(min_distance_deg / step)))


def estimate_fwhm(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    peak_index: int,
    baseline: float,
) -> float | None:
    """在扣除局部基线后，用半高处左右插值估算 FWHM（单位：°2θ）。

    若峰形不完整或一侧无法降到半高，则返回 None。
    """
    height = float(intensity[peak_index] - baseline)
    if height <= 0:
        return None
    half = baseline + height / 2.0

    # 向左找到穿越半高的区间
    i = peak_index
    while i > 0 and intensity[i] > half:
        i -= 1
    if intensity[i] > half:
        return None
    if two_theta[i] == two_theta[i + 1]:
        left = two_theta[i]
    else:
        # 线性插值
        left = two_theta[i] + (half - intensity[i]) * (
            two_theta[i + 1] - two_theta[i]
        ) / (intensity[i + 1] - intensity[i])

    # 向右
    j = peak_index
    last = len(intensity) - 1
    while j < last and intensity[j] > half:
        j += 1
    if intensity[j] > half:
        return None
    if two_theta[j] == two_theta[j - 1]:
        right = two_theta[j]
    else:
        right = two_theta[j - 1] + (half - intensity[j - 1]) * (
            two_theta[j] - two_theta[j - 1]
        ) / (intensity[j] - intensity[j - 1])

    width = float(right - left)
    return width if width > 0 else None


def find_peaks_xy(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    prominence_frac: float = 0.02,
    height_frac: float = 0.03,
    min_distance_deg: float = 0.3,
) -> list[dict]:
    """用 scipy.signal.find_peaks 寻峰，并给出相对强度与 FWHM。

    参数
    ----
    prominence_frac : 突出度阈值，相对于（最大强度 - 基线）的比例。
    height_frac     : 最小峰高，相对于（最大强度 - 基线）的比例。
    min_distance_deg: 相邻峰最小间距，单位 °2θ。
    """
    if find_peaks is None:
        raise ImportError("未安装 scipy，请先执行：pip install scipy")

    # 用较低分位数估计整体背景，避免把隆起的背底当成峰高
    baseline = float(np.percentile(intensity, 10))
    span = float(np.max(intensity) - baseline)
    if span <= 0:
        return []

    peaks_idx, props = find_peaks(
        intensity,
        height=baseline + height_frac * span,
        prominence=prominence_frac * span,
        distance=_degree_to_index_distance(two_theta, min_distance_deg),
    )
    if len(peaks_idx) == 0:
        return []

    max_i = float(np.max(intensity[peaks_idx]))
    rows: list[dict] = []
    for k, idx in enumerate(peaks_idx):
        i = int(idx)
        raw_i = float(intensity[i])
        # 相对强度以最强峰为 100
        rel = 100.0 * raw_i / max_i if max_i > 0 else 0.0
        # 局部基线取整谱 10% 分位与峰两侧谷值的较大者，更贴近单峰
        left_valley = float(np.min(intensity[max(0, i - 20) : i + 1]))
        right_valley = float(np.min(intensity[i : min(len(intensity), i + 21)]))
        local_base = max(baseline, min(left_valley, right_valley))
        fwhm = estimate_fwhm(two_theta, intensity, i, local_base)
        rows.append(
            {
                "index": k + 1,
                "two_theta": round(float(two_theta[i]), 4),
                "intensity": round(raw_i, 4),
                "relative_intensity": round(rel, 2),
                "fwhm": None if fwhm is None else round(fwhm, 4),
                "prominence": round(float(props["prominences"][k]), 4),
            }
        )
    return rows


def format_peak_table(peaks: list[dict], title: str | None = None) -> str:
    """把峰表格式化成纯文本，便于写入文件或作为提示词交给模型。"""
    lines: list[str] = []
    if title:
        lines.append(title)
    lines.append("由原始谱图数据自动寻峰（比读图更准确，请优先采信本表）：")
    lines.append("序号\t2θ(°)\t强度\t相对强度(最强峰=100)\tFWHM(°)")
    if not peaks:
        lines.append("（未检出明显衍射峰，请检查数据或放宽寻峰阈值）")
        return "\n".join(lines)
    for p in peaks:
        fwhm = "NA" if p.get("fwhm") is None else f"{p['fwhm']:.4f}"
        lines.append(
            f"{p['index']}\t{p['two_theta']:.4f}\t{p['intensity']:.4f}\t"
            f"{p['relative_intensity']:.2f}\t{fwhm}"
        )
    return "\n".join(lines)


def save_peaks_csv(peaks: list[dict], out_path: str | Path) -> None:
    """将峰表写成 UTF-8 BOM 的 CSV，方便 Excel 打开。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "two_theta", "intensity", "relative_intensity", "fwhm", "prominence"]
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in peaks:
            writer.writerow(p)


def plot_spectrum(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    peaks: list[dict],
    out_path: str | Path,
    title: str | None = None,
) -> None:
    """画出谱图并标注寻到的峰。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("画图需要 matplotlib，请先执行：pip install matplotlib") from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    # 图例用英文，避免在未装中文字体的环境下缺字
    ax.plot(two_theta, intensity, color="#1f4e79", lw=0.9, label="Observed")
    if peaks:
        px = [p["two_theta"] for p in peaks]
        # 用最接近的数据点取强度，保证标记落在曲线上
        py = []
        for p in peaks:
            idx = int(np.argmin(np.abs(two_theta - p["two_theta"])))
            py.append(float(intensity[idx]))
        ax.scatter(px, py, s=18, c="#c0392b", zorder=3, label="Peak")
        ymax = max(np.max(intensity), 1.0)
        for p, y in zip(peaks, py):
            ax.annotate(
                f"{p['two_theta']:.2f}",
                (p["two_theta"], y),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=6.5,
                color="#c0392b",
                rotation=90,
            )
    ax.set_xlabel(r"$2\theta$ (°)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(title or "XRD peak picking")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 XRD 原始数据（.xy/.txt/.csv）寻峰，输出峰位、相对强度和 FWHM。",
        epilog="提示：有原始数据时请优先用本脚本读峰，不要只依赖模型读图。",
    )
    parser.add_argument("input", help="谱图数据文件路径")
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="峰表 CSV 输出路径（默认：与输入同名的 *_peaks.csv）",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="AUTO",
        default=None,
        help="同时出图。不跟路径时，默认保存为与输入同名的 *_peaks.png",
    )
    parser.add_argument(
        "--prominence",
        type=float,
        default=0.02,
        help="突出度阈值，占（最大强度-基线）的比例，默认 0.02",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=0.03,
        help="最小峰高比例，默认 0.03",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=0.3,
        help="相邻峰最小间距（°2θ），默认 0.3",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.input)
    try:
        x, y = load_xy(path)
        peaks = find_peaks_xy(
            x,
            y,
            prominence_frac=args.prominence,
            height_frac=args.height,
            min_distance_deg=args.distance,
        )
    except (OSError, ValueError, ImportError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    out_csv = Path(args.out) if args.out else path.with_name(f"{path.stem}_peaks.csv")
    save_peaks_csv(peaks, out_csv)
    print(format_peak_table(peaks, title=f"文件：{path.name}"))
    print(f"\n已写入峰表：{out_csv}")

    if args.plot is not None:
        plot_path = (
            path.with_name(f"{path.stem}_peaks.png")
            if args.plot == "AUTO"
            else Path(args.plot)
        )
        plot_spectrum(x, y, peaks, plot_path, title=path.name)
        print(f"已出图：{plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
