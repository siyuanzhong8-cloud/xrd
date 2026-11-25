#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成合成 XRD 示例图与对应的 .xy 数据，供本地演练，无需真实样品。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pseudo_voigt(x: np.ndarray, center: float, height: float, fwhm: float, eta: float = 0.5) -> np.ndarray:
    """简单伪 Voigt 峰形，用于合成谱图。"""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0
    gaussian = height * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    lorentz = height * (gamma**2) / ((x - center) ** 2 + gamma**2)
    return eta * lorentz + (1.0 - eta) * gaussian


def synthesize(
    x: np.ndarray,
    peaks: list[tuple[float, float, float]],
    noise: float = 1.2,
    seed: int = 0,
) -> np.ndarray:
    """peaks: (2θ, 相对强度, FWHM)。再叠加缓慢背底与噪声。"""
    rng = np.random.default_rng(seed)
    y = 8.0 + 25.0 * np.exp(-(x - x.min()) / 40.0)  # 低角略高的背底
    for cen, height, fwhm in peaks:
        y = y + pseudo_voigt(x, cen, height, fwhm)
    y = y + rng.normal(0.0, noise, size=x.shape)
    return np.clip(y, 0, None)


# 锐钛矿 TiO2（Cu Kα）常见峰，2θ / 相对强度 / FWHM / hkl
ANATASE = [
    (25.28, 100, 0.22, "101"),
    (37.80, 20, 0.24, "004"),
    (48.05, 28, 0.23, "200"),
    (53.89, 18, 0.25, "105"),
    (55.06, 16, 0.25, "211"),
    (62.69, 12, 0.26, "204"),
    (68.76, 6, 0.28, "116"),
    (70.31, 6, 0.28, "220"),
    (75.03, 8, 0.27, "215"),
]

# 六方 ZnO（Cu Kα）
ZNO = [
    (31.77, 57, 0.20, "100"),
    (34.42, 44, 0.20, "002"),
    (36.25, 100, 0.21, "101"),
    (47.54, 23, 0.22, "102"),
    (56.60, 32, 0.22, "110"),
    (62.86, 29, 0.23, "103"),
    (66.38, 4, 0.24, "200"),
    (67.96, 23, 0.23, "112"),
    (69.10, 11, 0.24, "201"),
]


def save_xy(path: Path, x: np.ndarray, y: np.ndarray, header: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {header}\n")
        f.write("# 2theta_deg  intensity\n")
        for a, b in zip(x, y):
            f.write(f"{a:.4f}  {b:.4f}\n")


def style_axes(ax, title: str) -> None:
    ax.set_xlabel(r"$2\theta$ (degree)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(10, 80)
    ax.set_ylim(bottom=0)


def make_anatase(out_dir: Path) -> None:
    x = np.arange(10.0, 80.01, 0.02)
    peaks = [(c, h * 40.0, w) for c, h, w, _ in ANATASE]
    y = synthesize(x, peaks, noise=2.0, seed=7)
    save_xy(out_dir / "TiO2_anatase.xy", x, y, "synthetic anatase TiO2, Cu Ka")

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=160)
    ax.plot(x, y, color="#1f4e79", lw=0.9)
    ymax = float(np.max(y))
    for c, h, w, hkl in ANATASE:
        # 在峰顶附近标注 hkl，模拟论文里常见的标峰
        idx = int(np.argmin(np.abs(x - c)))
        ax.annotate(
            hkl,
            (c, y[idx]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#1f4e79",
        )
    style_axes(ax, "Anatase TiO$_2$ (Cu K$\\alpha$)")
    fig.tight_layout()
    fig.savefig(out_dir / "TiO2_anatase.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_comparison(out_dir: Path) -> None:
    x = np.arange(10.0, 80.01, 0.02)
    y_tio2 = synthesize(x, [(c, h * 36.0, w) for c, h, w, _ in ANATASE], noise=1.8, seed=11)
    y_zno = synthesize(x, [(c, h * 36.0, w) for c, h, w, _ in ZNO], noise=1.8, seed=19)
    # 对比图本身不强制配对单一 .xy，但各写一份，方便单独练习
    save_xy(out_dir / "ZnO.xy", x, y_zno, "synthetic ZnO, Cu Ka")

    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=160)
    # 为避免重叠，ZnO 曲线上移
    offset = float(np.max(y_tio2)) * 0.55
    ax.plot(x, y_tio2, color="#1f4e79", lw=0.9, label="TiO$_2$ anatase")
    ax.plot(x, y_zno + offset, color="#b03a2e", lw=0.9, label="ZnO")

    # 图下方用竖线模拟 PDF/JCPDS 标准卡片标注（ZnO）
    y0, y1 = -offset * 0.08, offset * 0.10
    first = True
    for c, h, w, hkl in ZNO:
        ax.vlines(c, y0, y1 + (h / 100.0) * (y1 - y0) * 2, colors="#b03a2e", lw=0.8,
                  label="PDF 36-1451 ZnO" if first else None)
        first = False
    ax.axhline(0, color="#999999", lw=0.4)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(ax, "Comparison: anatase TiO$_2$ vs ZnO (Cu K$\\alpha$)")
    ax.set_ylim(y0 * 1.4, float(np.max(y_zno + offset)) * 1.08)
    fig.tight_layout()
    fig.savefig(out_dir / "ZnO_vs_TiO2.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_large_copy(out_dir: Path) -> None:
    """另存一张长边超过 2048 的图，用于检查自动缩放。"""
    src = plt.imread(out_dir / "TiO2_anatase.png")
    fig, ax = plt.subplots(figsize=(16, 10), dpi=220)
    ax.imshow(src)
    ax.axis("off")
    fig.savefig(out_dir / "TiO2_anatase_large.png", bbox_inches="tight", facecolor="white", dpi=220)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_anatase(out_dir)
    make_comparison(out_dir)
    make_large_copy(out_dir)
    print(f"示例已写入：{out_dir}")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f"  {p.name:28s}  {p.stat().st_size:8d} 字节")


if __name__ == "__main__":
    main()
