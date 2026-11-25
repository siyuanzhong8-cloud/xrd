#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""程序化 XRD 描点：检测坐标框、鲁棒定标、按颜色/基线描线，并做峰位交叉校验。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GRID_DEG = 0.02
PEAK_MISMATCH_DEG = 0.3

NAMED_RGB: dict[str, tuple[int, int, int]] = {
    "红": (200, 30, 30), "红色": (200, 30, 30), "red": (200, 30, 30),
    "蓝": (30, 80, 200), "蓝色": (30, 80, 200), "blue": (30, 80, 200),
    "黑": (25, 25, 25), "黑色": (25, 25, 25), "black": (25, 25, 25),
    "绿": (30, 140, 40), "绿色": (30, 140, 40), "green": (30, 140, 40),
    "紫": (140, 40, 170), "紫色": (140, 40, 170), "purple": (140, 40, 170),
    "violet": (140, 40, 170),
    "粉": (220, 70, 180), "粉色": (220, 70, 180), "洋红": (220, 70, 180),
    "magenta": (220, 70, 180), "pink": (220, 70, 180),
    "橙": (230, 120, 30), "橙色": (230, 120, 30), "orange": (230, 120, 30),
    "青": (20, 140, 140), "cyan": (20, 140, 140), "teal": (20, 140, 140),
    "棕": (120, 70, 30), "褐色": (120, 70, 30), "brown": (120, 70, 30),
    "灰": (120, 120, 120), "灰色": (120, 120, 120), "gray": (120, 120, 120),
    "grey": (120, 120, 120),
}


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def ink_mask(rgb: np.ndarray, white: int = 242) -> np.ndarray:
    return np.any(rgb < white, axis=2)


def gray_of(rgb: np.ndarray) -> np.ndarray:
    return np.mean(rgb, axis=2)


def _runs_1d(mask_row: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], np.asarray(mask_row, dtype=bool), [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends))  # [start, end)


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _cluster_lines(segs: list[tuple[int, int, int]], pos_tol: int = 2) -> list[dict[str, int]]:
    if not segs:
        return []
    groups = [[segs[0]]]
    for s in segs[1:]:
        if s[0] - groups[-1][-1][0] <= pos_tol:
            groups[-1].append(s)
        else:
            groups.append([s])
    lines = []
    for g in groups:
        pos = int(round(float(np.median([x[0] for x in g]))))
        a = min(x[1] for x in g)
        b = max(x[2] for x in g)
        thick = 1 + max(x[0] for x in g) - min(x[0] for x in g)
        lines.append({"pos": pos, "a": int(a), "b": int(b), "thick": int(thick), "length": int(b - a + 1)})
    return lines


def find_hv_lines(ink: np.ndarray) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    H, W = ink.shape
    min_h = max(70, int(0.12 * W))
    min_v = max(40, int(0.18 * H))
    hsegs: list[tuple[int, int, int]] = []
    for y in range(H):
        for s, e in _runs_1d(ink[y]):
            if e - s >= min_h:
                hsegs.append((y, s, e - 1))
    vsegs: list[tuple[int, int, int]] = []
    for x in range(W):
        for s, e in _runs_1d(ink[:, x]):
            if e - s >= min_v:
                vsegs.append((x, s, e - 1))
    hlines = [L for L in _cluster_lines(hsegs, 2) if L["length"] >= min_h]
    vlines = [L for L in _cluster_lines(vsegs, 2) if L["length"] >= min_v]
    return hlines, vlines


def frames_from_spines(
    ink: np.ndarray,
    hlines: list[dict[str, int]],
    vlines: list[dict[str, int]],
) -> list[dict[str, Any]]:
    """用细竖线脊柱配对构造数据框，顶底对齐到附近横线或竖线端点。"""
    H, W = ink.shape
    # 横线端点处补虚拟脊柱（左/右轴被裁到图片边缘时仍能成框）
    ends: list[int] = []
    for h in hlines:
        ends.extend([h["a"], h["b"]])
    ends = sorted(ends)
    clusters: list[list[int]] = []
    for e in ends:
        if clusters and e - clusters[-1][-1] <= 8:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    virtual = []
    for cl in clusters:
        if len(cl) < 2:
            continue
        pos = int(round(float(sum(cl) / len(cl))))
        virtual.append({"pos": pos, "a": 0, "b": H - 1, "thick": 1, "length": H, "virtual": True})

    # 脊柱必须落在长横线端点附近，避免把上下对齐的强峰连成“竖线框”
    def _near_h_end(v):
        for h in hlines:
            if abs(v["pos"] - h["a"]) <= 10 or abs(v["pos"] - h["b"]) <= 10:
                if _overlap(v["a"], v["b"], h["pos"] - 8, h["pos"] + 8) >= 0 or True:
                    # 竖线应穿过或邻接该横线
                    if v["a"] - 12 <= h["pos"] <= v["b"] + 12:
                        return True
        return False

    spines = [v for v in vlines if v["thick"] <= 8 and _near_h_end(v)]
    if len(spines) < 2:
        spines = [v for v in vlines if v["thick"] <= 8]
    # 合并虚拟脊柱（与已有真实脊柱距离太近则不用）
    for v in virtual:
        if all(abs(v["pos"] - s["pos"]) > 10 for s in spines):
            spines.append(v)
    spines = sorted(spines, key=lambda s: s["pos"])
    cands: list[dict[str, Any]] = []
    for i, left in enumerate(spines):
        for right in spines[i + 1 :]:
            width = right["pos"] - left["pos"]
            if width < 80:
                continue
            # 虚拟脊柱跨越整页，顶底改用真实脊柱或横线范围
            if left.get("virtual") and not right.get("virtual"):
                y0, y1 = int(right["a"]), int(right["b"])
            elif right.get("virtual") and not left.get("virtual"):
                y0, y1 = int(left["a"]), int(left["b"])
            else:
                y0 = max(0, min(left["a"], right["a"]))
                y1 = min(H - 1, max(left["b"], right["b"]))
            # 吸附到覆盖本框宽度的横线：顶取最上、底取最下（底轴常比竖线端点略高，竖线会伸进刻度数字）
            tops, bots = [], []
            for h in hlines:
                if _overlap(h["a"], h["b"], left["pos"], right["pos"]) < 0.5 * width:
                    continue
                if abs(h["pos"] - y0) <= 16 or h["pos"] <= y0 + 8:
                    if y0 - 20 <= h["pos"] <= y0 + 24:
                        tops.append(h["pos"])
                if abs(h["pos"] - y1) <= 20 or (y1 - 24 <= h["pos"] <= y1 + 8):
                    bots.append(h["pos"])
            if tops:
                y0 = min(tops)
            if bots:
                # 底轴是靠近竖线下端的那条实线，不要用更靠上的基线
                y1 = max(bots)
            height = y1 - y0
            if height < 50:
                continue
            ar = width / max(height, 1)
            if ar < 0.4 or ar > 8.5:
                continue
            if _overlap(left["a"], left["b"], y0, y1) < 0.70 * height:
                continue
            if _overlap(right["a"], right["b"], y0, y1) < 0.70 * height:
                continue
            cands.append({
                "box": [int(left["pos"]), int(y0), int(right["pos"]), int(y1)],
                "width": int(width),
                "height": int(height),
                "vscore": 2,
            })
    return cands


def _ink_stats(ink: np.ndarray, box: list[int]) -> dict[str, float]:
    x0, y0, x1, y1 = box
    pad = 2
    sl = ink[y0 + pad : max(y0 + pad + 1, y1 - pad), x0 + pad : max(x0 + pad + 1, x1 - pad)]
    if sl.size == 0:
        return {"ink_frac": 0.0, "long_col_frac": 1.0, "peak_col_frac": 0.0}
    h = sl.shape[0]
    long_cols = 0
    peak_cols = 0
    for x in range(sl.shape[1]):
        rr = _runs_1d(sl[:, x])
        if not rr:
            continue
        mx = max(e - s for s, e in rr)
        if mx >= 0.45 * h:
            long_cols += 1
        elif 3 <= mx <= 0.4 * h:
            peak_cols += 1
    n = max(sl.shape[1], 1)
    return {
        "ink_frac": float(sl.mean()),
        "long_col_frac": long_cols / n,
        "peak_col_frac": peak_cols / n,
    }



def longest_regular_marks(marks: list[float], rel_tol: float = 0.22) -> list[float]:
    """取出最长的近似等差刻度序列（丢掉邻图或次刻度混入的点）。"""
    xs = sorted(float(m) for m in marks)
    n = len(xs)
    if n <= 2:
        return xs
    best: list[float] = xs[:1]
    for i in range(n - 1):
        for j in range(i + 1, n):
            step = xs[j] - xs[i]
            if step < 8:
                continue
            seq = [xs[i], xs[j]]
            expect = xs[j] + step
            for k in range(j + 1, n):
                if abs(xs[k] - expect) <= rel_tol * step:
                    seq.append(xs[k])
                    expect = xs[k] + step
                elif xs[k] > expect + rel_tol * step:
                    expect += step
                    if abs(xs[k] - expect) <= rel_tol * step:
                        seq.append(xs[k])
                        expect = xs[k] + step
            if len(seq) > len(best):
                best = seq
    return best if len(best) >= 3 else xs



def _regularize_marks(marks: list[float]) -> list[float]:
    """若间距大致均匀，丢掉偏离等差的孤立点。"""
    marks = [float(m) for m in marks]
    if len(marks) >= 4:
        gaps = np.diff(marks)
        med = float(np.median(gaps))
        if med >= 8:
            keep = [marks[0]]
            for m, g in zip(marks[1:], gaps):
                if 0.55 * med <= g <= 1.55 * med:
                    keep.append(m)
                elif g < 0.4 * med:
                    continue
                else:
                    keep.append(m)
            marks = keep
    return marks


def _cluster_mean_xs(xs: list[int], gap: int = 3) -> list[tuple[float, int, list[int]]]:
    """把相邻像素收成刻度，返回 (中心x, 簇宽, 原x列表)。"""
    if not xs:
        return []
    clusters: list[list[int]] = [[xs[0]]]
    for x in xs[1:]:
        if x - clusters[-1][-1] <= gap:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [(float(np.mean(c)), len(c), c) for c in clusters]


def _axis_y_and_thick(ink: np.ndarray, box: list[int], axis_y: int | None) -> tuple[int, int]:
    """找底轴行位置与厚度，并跳过紧贴的第二条轴线（双线坐标轴）。"""
    x0, y0, x1, y1 = box
    H, W = ink.shape
    y_ax = int(axis_y if axis_y is not None else y1)
    y_ax = min(max(0, y_ax), H - 2)
    xa, xb = max(0, x0), min(W, x1 + 1)
    snapped = None
    for y in range(max(0, y_ax - 4), min(H - 1, y_ax + 10)):
        frac = float(ink[y, xa:xb].mean()) if xb > xa else 0.0
        if frac >= 0.45:
            snapped = y
            break
    if snapped is not None:
        y_ax = snapped
    # 连续“整行都黑”的厚度，再跳过中间空白后的第二条轴线
    def _thick_from(y_start: int) -> int:
        thick = 0
        for dy in range(0, 10):
            y = y_start + dy
            if y >= H:
                break
            frac = float(ink[y, xa:xb].mean()) if xb > xa else 0.0
            if frac >= 0.45:
                thick = dy + 1
            else:
                break
        return max(1, thick)

    thick = _thick_from(y_ax)
    y_after = y_ax + thick
    # 小间隙后再来一条横轴（常见于矢量图描边）
    for y in range(y_after, min(H, y_after + 6)):
        frac = float(ink[y, xa:xb].mean()) if xb > xa else 0.0
        if frac >= 0.45:
            thick = (y - y_ax) + _thick_from(y)
            break
        if frac >= 0.08:
            break
    return y_ax, thick


def detect_tick_marks_detailed(
    gray: np.ndarray,
    box: list[int],
    axis_y: int | None = None,
) -> list[dict[str, Any]]:
    """检测底轴短刻度，返回 [{x, length, dir}]。向下、轴下间隙、向上都会找，不再因向下为空而提前返回。"""
    x0, y0, x1, y1 = box
    H, W = gray.shape
    ink = gray < 200
    xa, xb = max(0, x0), min(W, x1 + 1)
    if xb - xa < 20:
        return []
    y_ax, thick = _axis_y_and_thick(ink, box, axis_y)
    width = max(1, xb - xa)

    def _down_runs(y_start: int, max_len: int = 16) -> np.ndarray:
        ext = np.zeros(xb - xa, dtype=int)
        for i, x in enumerate(range(xa, xb)):
            run = 0
            for y in range(y_start, min(H, y_start + max_len)):
                if ink[y, x]:
                    run += 1
                else:
                    break
            ext[i] = run
        return ext

    def _up_runs(y_start: int, max_len: int = 22) -> np.ndarray:
        ext = np.zeros(xb - xa, dtype=int)
        for i, x in enumerate(range(xa, xb)):
            run = 0
            for y in range(y_start, max(-1, y_start - max_len), -1):
                if y < 0:
                    break
                if ink[y, x]:
                    run += 1
                else:
                    break
            ext[i] = run
        return ext

    def _marks_from_runs(arr: np.ndarray, direction: str, min_run: int, max_run: int) -> list[dict[str, Any]]:
        # 若几乎每列都有相近的 run，这是轴线残边，不是刻度
        if arr.size and float((arr >= min_run).mean()) > 0.45:
            bg = int(np.median(arr))
            if bg >= min_run and float((np.abs(arr - bg) <= 1).mean()) > 0.6:
                return []
            min_run = max(min_run + 1, bg + 2)
        raw: list[tuple[int, int]] = []
        for i, run in enumerate(arr):
            if min_run <= int(run) <= max_run:
                raw.append((xa + i, int(run)))
        if not raw:
            return []
        # 按 x 聚类，长度取簇内最大
        xs = [x for x, _ in raw]
        clusters = _cluster_mean_xs(xs, gap=3)
        out: list[dict[str, Any]] = []
        run_at = {x: r for x, r in raw}
        for xm, _nw, members in clusters:
            length = max(run_at.get(x, 0) for x in members)
            out.append({"x": float(xm), "length": int(length), "dir": direction})
        return out

    sets: list[list[dict[str, Any]]] = []
    # 1) 轴下方紧贴的短突起
    down = _down_runs(y_ax + thick, 16)
    sets.append(_marks_from_runs(down, "down", 2, 16))
    # 2) 轴下先有小间隙再出现的刻度（厚轴线把“连接处”吃掉之后）
    gap_start = y_ax + thick
    for y in range(gap_start, min(H, gap_start + 8)):
        if float(ink[y, xa:xb].mean()) <= 0.04:
            continue
        # 已进入标签或刻度行
        down2 = _down_runs(y, 16)
        sets.append(_marks_from_runs(down2, "down", 2, 16))
        break
    # 3) 向上伸进绘图区的短刻度（不要把数据峰或左右脊整列算进去）
    up = _up_runs(y_ax - 1, 20)
    # 数据峰通常远长于刻度；刻度长度集中。先收 2–18 px，再丢掉明显更长的孤立柱
    up_marks = _marks_from_runs(up, "up", 2, 18)
    if up_marks:
        lengths = [m["length"] for m in up_marks]
        med = float(np.median(lengths))
        up_marks = [m for m in up_marks if m["length"] <= max(18, med + 6)]
    sets.append(up_marks)

    def _set_score(ms: list[dict[str, Any]]) -> float:
        if len(ms) < 3:
            return -1.0
        xs = sorted(m["x"] for m in ms)
        span = xs[-1] - xs[0]
        if span < 0.45 * width:
            return -0.5
        gaps = np.diff(xs)
        med = float(np.median(gaps)) or 1.0
        irregular = float(np.mean(np.abs(gaps - med) > 0.3 * med))
        mean_len = float(np.mean([m["length"] for m in ms]))
        # 真刻度通常比数字笔画更长、更整齐；不要只靠数量取胜
        return 2.0 * (span / width) + 4.0 * (1.0 - irregular) + 0.45 * mean_len + 0.15 * len(ms)

    # 仅当某一方向刻度太少时，才尝试合并；默认不要把数字笔画和真刻度混在一起
    nonempty = [ms for ms in sets if len(ms) >= 4]
    if len(nonempty) >= 2:
        # 若两个方向各自已经很规则，不再合并
        pass
    else:
        merged: dict[int, dict[str, Any]] = {}
        for ms in sets:
            for m in ms:
                key = int(round(m["x"]))
                oldm = merged.get(key)
                if oldm is None or m["length"] > oldm["length"]:
                    merged[key] = m
        if merged:
            sets.append(sorted(merged.values(), key=lambda z: z["x"]))

    best: list[dict[str, Any]] = []
    best_s = -2.0
    for ms in sets:
        ms = sorted(ms, key=lambda z: z["x"])
        # 丢掉偏离等差的孤立点（像素坐标）
        xs = _regularize_marks([m["x"] for m in ms])
        if xs:
            keep = []
            for m in ms:
                if any(abs(m["x"] - x) <= 2 for x in xs):
                    keep.append(m)
            ms = keep
        s = _set_score(ms)
        if s > best_s:
            best_s, best = s, ms
    return best


def detect_tick_marks(gray: np.ndarray, box: list[int], axis_y: int | None = None) -> list[float]:
    """沿底轴的短突起为刻度短线。同时检查向下、轴下间隙和向上（内向）刻度。"""
    return [float(m["x"]) for m in detect_tick_marks_detailed(gray, box, axis_y=axis_y)]


def major_mark_xs(detailed: list[dict[str, Any]]) -> list[float]:
    """从长短混杂的刻度里取出更长的主刻度（对应 10、20 这类标注）。"""
    if len(detailed) < 4:
        return [float(m["x"]) for m in detailed]
    lengths = np.array([int(m["length"]) for m in detailed], dtype=float)
    lo, hi = float(np.percentile(lengths, 30)), float(np.percentile(lengths, 75))
    # 双峰：主刻度明显更长
    if hi >= lo + 2:
        thr = (lo + hi) / 2.0
        majors = [float(m["x"]) for m in detailed if m["length"] >= thr]
        if len(majors) >= 2:
            return majors
    # 取相对更长的约 1/2 到 1/5（常见 1° 次刻度里每 10° 一根主刻度）
    med = float(np.median(lengths))
    longer = [float(m["x"]) for m in detailed if m["length"] >= med + 1.5]
    if 2 <= len(longer) <= max(3, len(detailed) // 2 + 1):
        return longer
    return [float(m["x"]) for m in detailed]


def _cut_through_plot(ink: np.ndarray, box: list[int]) -> bool:
    """左边或右边紧邻处仍是图谱内部（框切在图中间）。"""
    x0, y0, x1, y1 = box
    H, W = ink.shape
    mid0 = y0 + (y1 - y0) // 4
    mid1 = y1 - (y1 - y0) // 4
    if mid1 <= mid0:
        return False
    def band_frac(xa, xb):
        xa, xb = max(0, xa), min(W, xb)
        if xb <= xa:
            return 0.0
        return float(ink[mid0:mid1, xa:xb].mean())
    left = band_frac(x0 - 16, x0 - 2)
    right = band_frac(x1 + 2, x1 + 16)
    return left > 0.08 or right > 0.08


def score_frame(ink: np.ndarray, cand: dict[str, Any], marks: list[float], shape: tuple[int, int]) -> dict[str, Any]:
    H, W = shape
    box = cand["box"]
    st = _ink_stats(ink, box)
    area_frac = (cand["width"] * cand["height"]) / max(H * W, 1)
    s = 0.0
    s += 1.2 * cand.get("vscore", 0)
    regular = longest_regular_marks(marks)
    s += min(len(regular), 14) * 1.6
    # 全部短线若很不均匀（多半框进了邻图），扣分
    if len(marks) >= 4:
        gaps = np.diff(sorted(marks))
        med = float(np.median(gaps)) or 1.0
        bad = float(np.mean(np.abs(gaps - med) > 0.4 * med))
        s -= 6.0 * bad
    s += st["peak_col_frac"] * 6.0
    s -= st["long_col_frac"] * 12.0
    s -= max(0.0, st["ink_frac"] - 0.14) * 10.0
    if 0.06 <= area_frac <= 0.8:
        s += 1.2
    if area_frac > 0.88:
        s -= 2.5
    if area_frac < 0.04:
        s -= 2.0
    if _cut_through_plot(ink, box):
        s -= 4.0
    cand = dict(cand)
    cand["stats"] = st
    cand["n_ticks"] = len(marks)
    cand["marks"] = marks
    cand["area_frac"] = area_frac
    cand["score"] = float(s)
    return cand


def choose_frame(
    rgb: np.ndarray,
    plot_box_1000: list[float] | None = None,
) -> dict[str, Any] | None:
    ink = ink_mask(rgb)
    gray = gray_of(rgb)
    H, W = ink.shape
    hlines, vlines = find_hv_lines(ink)
    cands = frames_from_spines(ink, hlines, vlines)
    scored: list[dict[str, Any]] = []
    for c in cands:
        marks = detect_tick_marks(gray, c["box"], axis_y=c["box"][3])
        scored.append(score_frame(ink, c, marks, (H, W)))
    if not scored:
        return None
    # 若给出 plot_box，给重叠多的框加分，但不盲从（多子图时常会包进邻图）
    if plot_box_1000 and len(plot_box_1000) == 4:
        pb = [plot_box_1000[0] / 1000.0 * W, plot_box_1000[1] / 1000.0 * H,
              plot_box_1000[2] / 1000.0 * W, plot_box_1000[3] / 1000.0 * H]
        for c in scored:
            x0, y0, x1, y1 = c["box"]
            ow = _overlap(int(x0), int(x1), int(pb[0]), int(pb[2]))
            oh = _overlap(int(y0), int(y1), int(pb[1]), int(pb[3]))
            inter = ow * oh
            union = (x1 - x0) * (y1 - y0) + (pb[2] - pb[0]) * (pb[3] - pb[1]) - inter
            iou = inter / union if union > 0 else 0.0
            c["score"] += 1.5 * iou
            c["iou_hint"] = iou
    scored.sort(key=lambda c: -c["score"])
    uniq: list[dict[str, Any]] = []
    for c in scored:
        if any(
            abs(c["box"][0] - u["box"][0]) < 10
            and abs(c["box"][2] - u["box"][2]) < 10
            and abs(c["box"][1] - u["box"][1]) < 10
            and abs(c["box"][3] - u["box"][3]) < 10
            for u in uniq
        ):
            continue
        uniq.append(c)
    if not uniq:
        return None
    # 若大框里套着一个刻度更整齐的 XRD 小框（例如左边是极化曲线、右边才是 XRD），选小框
    def _reg(c):
        return longest_regular_marks(c.get("marks") or [])
    best = uniq[0]
    for inner in uniq[1:]:
        ix0, iy0, ix1, iy1 = inner["box"]
        ox0, oy0, ox1, oy1 = best["box"]
        inside = ix0 >= ox0 - 8 and iy0 >= oy0 - 8 and ix1 <= ox1 + 8 and iy1 <= oy1 + 8
        if not inside:
            continue
        if len(_reg(inner)) >= 6 and len(_reg(inner)) >= len(_reg(best)) - 1:
            # 小框面积至少是大框的 15%，避免选中碎片
            ia = max(1, (ix1 - ix0) * (iy1 - iy0))
            oa = max(1, (ox1 - ox0) * (oy1 - oy0))
            if ia / oa >= 0.15 and inner.get("stats", {}).get("long_col_frac", 0) <= 0.15:
                best = inner
    return best


def ransac_linear(xs: np.ndarray, ys: np.ndarray, thresh: float = 0.6, n_iter: int = 80) -> dict[str, Any]:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    if n < 2:
        raise ValueError("定标至少需要 2 个刻度")
    rng = np.random.default_rng(0)
    best: tuple[int, np.ndarray] | None = None
    if n == 2:
        inliers = np.array([0, 1], dtype=int)
    else:
        for _ in range(n_iter):
            i, j = rng.choice(n, 2, replace=False)
            if abs(xs[i] - xs[j]) < 3:
                continue
            b = (ys[j] - ys[i]) / (xs[j] - xs[i])
            a = ys[i] - b * xs[i]
            resid = np.abs(a + b * xs - ys)
            mask = resid <= thresh
            k = int(mask.sum())
            if best is None or k > best[0]:
                best = (k, np.where(mask)[0])
        inliers = best[1] if best and best[0] >= 2 else np.arange(n)
    A = np.column_stack([xs[inliers], np.ones(len(inliers))])
    (b, a), *_ = np.linalg.lstsq(A, ys[inliers], rcond=None)
    pred = b * xs + a
    resid = pred - ys
    max_res = float(np.max(np.abs(resid)))
    return {
        "a": float(a),
        "b": float(b),
        "max_residual_deg": max_res,
        "residuals": [float(v) for v in resid],
        "px_per_deg": float(abs(1.0 / b)) if b else None,
        "ok": max_res <= 0.5,
        "n_inliers": int(len(inliers)),
    }


def _tick_values(calib: dict[str, Any]) -> list[float]:
    vals: list[float] = []
    ticks = calib.get("ticks") if isinstance(calib.get("ticks"), list) else []
    if not ticks:
        xa = calib.get("x_axis")
        ticks = xa.get("ticks") if isinstance(xa, dict) and isinstance(xa.get("ticks"), list) else []
    for t in ticks:
        if isinstance(t, dict) and t.get("value") is not None:
            try:
                vals.append(float(t["value"]))
            except (TypeError, ValueError):
                pass
    return vals


def _regular_series(vals: list[float], rel_tol: float = 0.2) -> bool:
    if len(vals) < 3:
        return False
    gaps = np.diff(sorted(float(v) for v in vals))
    med = float(np.median(gaps))
    if med < 0.9:
        return False
    return float(np.median(np.abs(gaps - med))) <= rel_tol * med


def _x_range_of(calib: dict[str, Any]) -> tuple[float, float] | None:
    # 模型读出的整齐刻度读数优先于可能偏掉半格的 x_range（例如 10–80 被写成 5–75）
    tick_vals = _tick_values(calib)
    if _regular_series(tick_vals):
        return float(min(tick_vals)), float(max(tick_vals))
    for key in ("x_range",):
        xr = calib.get(key)
        if isinstance(xr, (list, tuple)) and len(xr) >= 2:
            try:
                lo, hi = float(xr[0]), float(xr[1])
                if hi > lo + 1:
                    return lo, hi
            except (TypeError, ValueError):
                pass
    xa = calib.get("x_axis")
    if isinstance(xa, dict):
        xr = xa.get("x_range")
        if isinstance(xr, (list, tuple)) and len(xr) >= 2:
            try:
                lo, hi = float(xr[0]), float(xr[1])
                if hi > lo + 1:
                    return lo, hi
            except (TypeError, ValueError):
                pass
        ticks = xa.get("ticks") if isinstance(xa.get("ticks"), list) else []
        vals = []
        for t in ticks:
            if isinstance(t, dict) and t.get("value") is not None:
                try:
                    vals.append(float(t["value"]))
                except (TypeError, ValueError):
                    pass
        if len(vals) >= 2:
            return min(vals), max(vals)
    if len(tick_vals) >= 2:
        return min(tick_vals), max(tick_vals)
    return None


def _vlm_ticks(calib: dict[str, Any], width: int) -> list[dict[str, float]]:
    out = []
    src = calib.get("ticks")
    if not isinstance(src, list):
        xa = calib.get("x_axis")
        src = xa.get("ticks") if isinstance(xa, dict) else []
    for t in src or []:
        if not isinstance(t, dict) or t.get("value") is None:
            continue
        try:
            val = float(t["value"])
        except (TypeError, ValueError):
            continue
        x_px = t.get("x_px")
        x1000 = t.get("x_1000")
        if x_px is None and x1000 is not None:
            try:
                x_px = float(x1000) / 1000.0 * width
            except (TypeError, ValueError):
                x_px = None
        rec = {"value": val}
        if x_px is not None:
            rec["x_px"] = float(x_px)
        out.append(rec)
    return out



def _values_for_marks(
    marks: list[float],
    lo: float,
    hi: float,
    frame: list[int] | None = None,
) -> list[float] | None:
    """均匀短线 -> 与 x_range 相容的等差读数（含 5° 次刻度，可向两端延拓）。"""
    n = len(marks)
    if n < 2 or hi <= lo:
        return None
    gaps = np.diff(marks)
    med = float(np.median(gaps))
    if med < 3:
        return None
    # 间距需大致均匀
    if float(np.median(np.abs(gaps - med))) > 0.35 * med:
        return None
    span = hi - lo
    # 优先：短线数目刚好覆盖 x_range（8 根 + 10–80 → 10,20,…,80），不要先错到半格 5,15,…,75
    if n >= 3:
        step_exact = span / (n - 1)
        for step in (10.0, 5.0, 2.0, 1.0, 20.0):
            if abs(step_exact - step) <= 0.12 * step:
                return [lo + i * step for i in range(n)]
    # 允许起点最多提前 2 格（未标注的次刻度），终点可超出 x_range 若干格（模型漏读了末尾刻度）
    # 禁止把 15 根 5° 短线强行拟合成 -20…120 这种远宽于 x_range 的 10° 序列
    first_on_spine = last_on_spine = False
    if frame is not None and marks:
        x0, _, x1, _ = frame
        first_on_spine = abs(float(marks[0]) - float(x0)) <= 8
        last_on_spine = abs(float(marks[-1]) - float(x1)) <= 8
    best: tuple[float, list[float]] | None = None
    k_max = 8 if (hi - lo) >= 15 else 3
    for step in (10.0, 5.0, 2.5, 2.0, 1.0, 20.0, 0.5):
        for k_start in range(0, k_max):
            start = lo - k_start * step
            end = start + (n - 1) * step
            if start > lo + 0.55 * step:
                continue
            if end + 0.55 * step < hi:
                continue
            if end > hi + 2.05 * step:
                continue
            if start < lo - 4.05 * step:
                continue
            if start < -1 and lo >= 0:
                continue
            score = abs(start - lo) + 0.25 * abs(end - hi)
            # 贴着左脊、却不贴右脊的第一根短线，多半是未标注起点（如 35°），不要钉成 x_range 左端
            if first_on_spine and not last_on_spine and k_start == 0:
                score += 3 * step
            elif first_on_spine and not last_on_spine and k_start >= 1:
                score -= 0.5 * step
            if best is None or score < best[0]:
                best = (score, [start + i * step for i in range(n)])
    return None if best is None else best[1]


def _ticks_filling_minors(
    marks: list[float],
    majors: list[float],
    major_vals: list[float],
) -> list[dict[str, Any]] | None:
    """用主刻度读数拟合，再给全部短线（含次刻度）填读数。"""
    if len(majors) < 2 or len(majors) != len(major_vals):
        return None
    fit = ransac_linear(np.array(majors, dtype=float), np.array(major_vals, dtype=float))
    if not fit.get("ok") and float(fit.get("max_residual_deg") or 99) > 0.8:
        return None
    a, bcoef = float(fit["a"]), float(fit["b"])
    if abs(bcoef) < 1e-9:
        return None
    # 次刻度步长：主刻度步长的公约数（10→5/2/1，5→1，1→1）
    major_step = float(np.median(np.diff(major_vals))) if len(major_vals) >= 2 else 10.0
    minor_step = 1.0
    for cand in (1.0, 2.0, 5.0, 0.5, 0.2, 0.1):
        if cand <= major_step + 1e-9 and abs(major_step / cand - round(major_step / cand)) < 0.05:
            minor_step = cand
            break
    out = []
    for x in marks:
        val = a + bcoef * float(x)
        snapped = round(val / minor_step) * minor_step
        if abs(snapped - val) > 0.45 * minor_step:
            snapped = val
        out.append({"value": float(snapped), "x_px": float(x), "snapped": True})
    return out if len(out) >= 2 else None



def _extend_marks_to_spines(
    marks: list[float],
    majors: list[float],
    frame: list[int],
) -> tuple[list[float], list[float]]:
    """若主刻度等差外推刚好落在左右脊上，把脊补成端点刻度（10°、90° 常与轴线重合）。"""
    marks = list(marks)
    majors = list(majors)
    if len(majors) < 2:
        return marks, majors
    step = float(np.median(np.diff(sorted(majors))))
    if step < 6:
        return marks, majors
    x0, _, x1, _ = frame
    first, last = majors[0], majors[-1]
    prev = first - step
    nxt = last + step
    # 只在脊几乎正好落在外推位置时补点（约 0.1 格）。0.25 格会把 105° 后的右脊错补成 110°。
    tol = max(6.0, 0.12 * step)
    if abs(prev - x0) <= tol:
        majors = [float(x0)] + majors
        if not any(abs(m - x0) <= 4 for m in marks):
            marks = [float(x0)] + marks
    if abs(nxt - x1) <= tol:
        majors = majors + [float(x1)]
        if not any(abs(m - x1) <= 4 for m in marks):
            marks = marks + [float(x1)]
    return marks, majors



def _snap_vlm_to_marks(
    vlm: list[dict[str, float]],
    marks: list[float],
    frame: list[int],
) -> list[dict[str, Any]] | None:
    """用模型读数作刻度，并在近处吸附到检测到的短线；略超出框的端点收到脊上。"""
    pts = [t for t in vlm if t.get("value") is not None and t.get("x_px") is not None]
    if len(pts) < 2:
        return None
    vals = [float(t["value"]) for t in pts]
    xs = [float(t["x_px"]) for t in pts]
    if not _regular_series(vals):
        return None
    x0, _, x1, _ = frame
    step_px = float(np.median(np.diff(xs))) if len(xs) >= 2 else 20.0
    # 略出框的端点保留原像素用于线性定标：把 90° 硬收到右脊会把 20–80 的好定标拉歪约 2°。
    far = max(40.0, 0.65 * abs(step_px))

    def _keep(val: float, x: float, snapped: bool = False) -> dict[str, Any] | None:
        if x < x0 - far or x > x1 + far:
            return None
        return {"value": float(val), "x_px": float(x), "snapped": bool(snapped)}

    # 版本 A：保留模型像素，不夹到脊
    plain = [p for p in (_keep(v, x) for v, x in zip(vals, xs)) if p]
    # 版本 A2：丢掉略出框的端点，只用框内点
    interior = [p for p in plain if x0 - 2 <= p["x_px"] <= x1 + 2]
    # 版本 B：仅当短线非常靠近模型像素时才吸附，仍不夹到脊
    snap_tol = max(5.0, 0.10 * abs(step_px))
    snapped_pts: list[dict[str, Any]] = []
    for val, x in zip(vals, xs):
        xc, snapped = float(x), False
        if marks:
            nearest = min(marks, key=lambda m: abs(float(m) - x))
            if abs(float(nearest) - x) <= snap_tol:
                xc, snapped = float(nearest), True
        item = _keep(val, xc, snapped)
        if item:
            snapped_pts.append(item)

    xr = (min(vals), max(vals))
    cands = []
    for cand in (plain, interior, snapped_pts):
        if cand and len(cand) >= 2:
            cands.append(cand)
    if not cands:
        return None
    return max(cands, key=lambda ticks: _tick_quality(ticks, frame, xr)[0])


def _clip_tick_overshoot(
    ticks: list[dict[str, Any]],
    xr: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    """丢掉远超出标注范围的外推次刻度（例如把 19–109 的轴描到 129°）。"""
    if not ticks or xr is None:
        return ticks
    lo, hi = xr
    span = max(hi - lo, 1.0)
    vals = [float(t["value"]) for t in ticks]
    if len(vals) >= 2:
        vstep = float(np.median(np.diff(sorted(vals)))) or 5.0
    else:
        vstep = 5.0
    # 允许比标注两端多约 1.5 个主间隔（唇膏 10–40 的 6°/45° 要保留）
    vmin = lo - max(1.6 * abs(vstep), 0.12 * span)
    vmax = hi + max(1.6 * abs(vstep), 0.12 * span)
    kept = [t for t in ticks if vmin - 1e-6 <= float(t["value"]) <= vmax + 1e-6]
    return kept if len(kept) >= 2 else ticks


def _tick_quality(
    ticks: list[dict[str, Any]] | None,
    frame: list[int],
    xr: tuple[float, float] | None,
) -> tuple[float, dict[str, Any] | None]:
    """分数越高越好。覆盖不足或残差很大的短线集（右半边误检数字）会得低分。"""
    if not ticks or len(ticks) < 2:
        return -999.0, None
    xs = [float(t["x_px"]) for t in ticks if t.get("x_px") is not None]
    vs = [float(t["value"]) for t in ticks if t.get("value") is not None]
    if len(xs) < 2 or len(vs) < 2:
        return -999.0, None
    fit = ransac_linear(np.array(xs, dtype=float), np.array(vs, dtype=float))
    resid = float(fit.get("max_residual_deg") or 99.0)
    x0, _, x1, _ = frame
    width = max(1.0, float(x1 - x0))
    n_out = sum(1 for x in xs if x < x0 - 8 or x > x1 + 8)
    span_x = min(float(x1), max(xs)) - max(float(x0), min(xs))
    coverage = max(0.0, min(1.0, span_x / width))
    overshoot = 0.0
    spine_err = 0.0
    if xr is not None:
        lo, hi = xr
        span = max(hi - lo, 1.0)
        overshoot = max(0.0, (lo - min(vs)) / span - 0.15) + max(0.0, (max(vs) - hi) / span - 0.15)
        a, bcoef = float(fit.get("a") or 0.0), float(fit.get("b") or 0.0)
        pred0 = a + bcoef * float(x0)
        pred1 = a + bcoef * float(x1)
        if pred1 < pred0:
            pred0, pred1 = pred1, pred0
        spine_err = abs(pred0 - lo) / span + abs(pred1 - hi) / span
    if resid <= 0.25:
        score = 12.0
    elif resid <= 0.6:
        score = 8.0
    elif resid <= 1.2:
        score = 3.0
    elif resid <= 3.0:
        score = -6.0
    else:
        score = -20.0
    score += 8.0 * coverage
    score -= 22.0 * overshoot
    score -= 7.0 * n_out
    score -= 10.0 * min(spine_err, 1.5)
    if not fit.get("ok"):
        score -= 8.0
    return score, fit


def _finalize_ticks(
    mark_ticks: list[dict[str, Any]] | None,
    vlm: list[dict[str, float]],
    marks: list[float],
    frame: list[int],
    xr: tuple[float, float] | None,
) -> list[dict[str, Any]] | None:
    if mark_ticks:
        mark_ticks = _clip_tick_overshoot(mark_ticks, xr)
    vlm_ticks = _snap_vlm_to_marks(vlm, marks, frame)
    mq, mfit = _tick_quality(mark_ticks, frame, xr)
    vq, vfit = _tick_quality(vlm_ticks, frame, xr)

    def _in_span(ticks: list[dict[str, Any]] | None) -> tuple[int, float]:
        if not ticks:
            return 99, 0.0
        xs = [float(t["x_px"]) for t in ticks if t.get("x_px") is not None]
        if len(xs) < 2:
            return 99, 0.0
        n_out = sum(1 for x in xs if x < frame[0] - 8 or x > frame[2] + 8)
        span_x = min(float(frame[2]), max(xs)) - max(float(frame[0]), min(xs))
        cov = max(0.0, min(1.0, span_x / max(1.0, float(frame[2] - frame[0]))))
        return n_out, cov

    # 印刷短线残差小且落在框内时优先：模型像素可自洽（残差 0）但仍整体平移十几度。
    if mark_ticks and mfit and float(mfit.get("max_residual_deg") or 99) <= 0.3:
        mnout, mcov = _in_span(mark_ticks)
        if mnout == 0 and mcov >= 0.75 and mq > -8:
            if vlm_ticks is None or vq < mq + 5:
                return mark_ticks
    if mark_ticks is not None and mq >= vq and mq > -8:
        return mark_ticks
    if vlm_ticks is not None and vq > mq:
        return vlm_ticks
    return mark_ticks or vlm_ticks


def assign_ticks(
    marks: list[float],
    calib: dict[str, Any],
    frame: list[int],
    width: int,
    detailed: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """把检测到的短线与读数对齐；若模型把刻度拉到邻图，则改用 x_range 均匀分配。"""
    xr = _x_range_of(calib)
    vlm = _vlm_ticks(calib, width)
    x0, _, x1, _ = frame
    marks_for_final = list(marks)
    # 模型刻度落在框外过多则弃用其像素位置
    usable = []
    for t in vlm:
        if "x_px" not in t:
            usable.append(t)
            continue
        if x0 - 15 <= t["x_px"] <= x1 + 15:
            usable.append(t)
    frac_inside = 0.0
    with_px = [t for t in vlm if "x_px" in t]
    if with_px:
        frac_inside = sum(1 for t in with_px if x0 - 15 <= t["x_px"] <= x1 + 15) / len(with_px)

    if marks and xr is not None and (frac_inside < 0.6 or len(marks) >= 3):
        lo, hi = xr
        marks_all = list(marks)
        marks = longest_regular_marks(marks)
        # 短线必须跨过框的大部分，否则多半是误检（不要覆盖本来可用的模型刻度）
        if marks and (marks[-1] - marks[0]) < 0.45 * max(1, x1 - x0) and len(marks) < 6:
            marks = []
        n = len(marks)
        vlm_vals = sorted({float(t["value"]) for t in vlm if t.get("value") is not None})
        # 主刻度（更长的短线）钉到标注读数，再填次刻度。这样 10° 不会错钉到轴端的 6° 短线上。
        if n >= 3 and detailed:
            majors = major_mark_xs([d for d in detailed if any(abs(float(d["x"]) - m) <= 3 for m in marks)])
            majors = longest_regular_marks(majors) or majors
            marks, majors = _extend_marks_to_spines(marks, majors, frame)
            labels = list(vlm_vals)
            if len(labels) < 2 or not _regular_series(labels):
                labels = []
            if majors and len(majors) >= 2:
                # 尝试用 x_range 两端给主刻度配 10/5° 序列
                best_vals = None
                if labels and len(labels) == len(majors) and _regular_series(labels):
                    best_vals = list(labels)
                else:
                    for step in (10.0, 5.0, 2.5, 2.0, 1.0, 20.0, 0.5, 0.2):
                        # 覆盖 lo..hi，允许两端比标注多几根（轴端未标注短线）
                        for k_start in range(0, 9):
                            start = lo - k_start * step
                            vals = [start + i * step for i in range(len(majors))]
                            if vals[0] > lo + 0.55 * step:
                                continue
                            if vals[-1] + 0.55 * step < hi:
                                continue
                            if vals[-1] > hi + 2.05 * step:
                                continue
                            if start < lo - 4.05 * step:
                                continue
                            if start < -1 and lo >= 0:
                                continue
                            # 优先：某根主刻度正好是 lo、某根正好是 hi
                            hit_lo = any(abs(v - lo) < 0.05 * step for v in vals)
                            hit_hi = any(abs(v - hi) < 0.05 * step for v in vals)
                            score = (0 if hit_lo else 2) + (0 if hit_hi else 1) + 0.05 * abs(vals[0] - lo) + 0.02 * abs(vals[-1] - hi)
                            if labels and len(labels) == len(majors):
                                score += 0.01 * sum(abs(a - b) for a, b in zip(vals, labels))
                            if best_vals is None or score < best_vals[0]:
                                best_vals = (score, vals)
                    if best_vals is not None:
                        best_vals = best_vals[1]
                if best_vals is not None:
                    filled = _ticks_filling_minors(marks, majors, best_vals)
                    if filled:
                        chosen = _finalize_ticks(filled, vlm, marks, frame, xr)
                        if chosen:
                            return chosen
        # 检测到的短线数与模型读数个数相同且都是整齐等差时，按顺序配对（用模型的读数、程序的像素）
        if n >= 4 and n == len(vlm_vals) and _regular_series(vlm_vals):
            pair = [{"value": float(v), "x_px": float(m), "snapped": True} for v, m in zip(vlm_vals, marks)]
            chosen = _finalize_ticks(pair, vlm, marks, frame, xr)
            if chosen:
                return chosen
        if n >= 2:
            vals = _values_for_marks(marks, lo, hi, frame=frame)
            if vals is not None:
                pair = [{"value": float(v), "x_px": float(m), "snapped": True} for v, m in zip(vals, marks)]
                chosen = _finalize_ticks(pair, vlm, marks, frame, xr)
                if chosen:
                    return chosen
            chosen = _finalize_ticks(None, vlm, marks, frame, xr)
            if chosen:
                return chosen
            raise ValueError(
                f"刻度短线 {n} 根与 x_range {lo:g}–{hi:g} 对不上，拒绝拉伸定标"
            )

    # 尝试把模型读数吸附到短线
    if marks and usable:
        values = [t["value"] for t in usable]
        n = len(values)
        if len(marks) >= n >= 2:
            yv = np.array(values, dtype=float)
            best = None
            for s in range(len(marks) - n + 1):
                win = np.array(marks[s : s + n], dtype=float)
                A = np.column_stack([win, np.ones(n)])
                (b, a), *_ = np.linalg.lstsq(A, yv, rcond=None)
                resid = float(np.max(np.abs(A @ np.array([b, a]) - yv)))
                if best is None or resid < best[0]:
                    best = (resid, s, float(b), float(a))
            if best and best[0] <= 0.8:
                resid, start, b, a = best
                snapped = []
                steps = np.diff(yv)
                step = float(np.median(steps)) if len(steps) else 0.0
                vs = [None] * len(marks)
                for i, val in enumerate(values):
                    vs[start + i] = val
                if abs(step) >= 0.5:
                    for i in range(start - 1, -1, -1):
                        vs[i] = vs[i + 1] - step
                    for i in range(start + n, len(marks)):
                        vs[i] = vs[i - 1] + step
                for x, v in zip(marks, vs):
                    if v is not None:
                        snapped.append({"value": float(v), "x_px": float(x), "snapped": True})
                if len(snapped) >= 2:
                    chosen = _finalize_ticks(snapped, vlm, marks, frame, xr)
                    if chosen:
                        return chosen

    # 模型像素经常整体平移却仍「线性很好」（残差≈0，主峰却偏 2°）。
    # 只要检测到短线，就必须把读数吸附到短线上，不要直接用未吸附的模型坐标。
    if marks and len(usable) >= 2:
        marks = longest_regular_marks(marks) or marks
        values = [float(t["value"]) for t in usable]
        if len(marks) >= 2:
            # 用模型读数的等差，把短线从两端对齐
            xr_lo, xr_hi = (min(values), max(values)) if values else (None, None)
            if xr is not None:
                xr_lo, xr_hi = xr
            vals = _values_for_marks(marks, float(xr_lo), float(xr_hi), frame=frame) if xr_lo is not None else None
            if vals is not None:
                pair = [{"value": float(v), "x_px": float(m), "snapped": True} for v, m in zip(vals, marks)]
                chosen = _finalize_ticks(pair, vlm, marks, frame, xr)
                if chosen:
                    return chosen
    if len(usable) >= 2 and all("x_px" in t for t in usable) and frac_inside >= 0.6 and not marks:
        pair = [{"value": t["value"], "x_px": t["x_px"], "snapped": False} for t in usable]
        chosen = _finalize_ticks(pair, vlm, marks, frame, xr)
        if chosen:
            return chosen
    chosen = _finalize_ticks(None, vlm, marks, frame, xr)
    if chosen:
        return chosen
    raise ValueError("无法把刻度短线与读数对齐")


def _is_black_name(name: str) -> bool:
    n = (name or "").strip().lower()
    return n in {"黑", "黑色", "black", "k", "grey", "gray", "灰", "灰色"}


def resolve_color(curve: dict[str, Any] | None) -> tuple[tuple[int, int, int], bool] | None:
    if not isinstance(curve, dict):
        return None
    rgb = curve.get("color_rgb")
    if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
        try:
            r, g, b = [int(float(v)) for v in rgb[:3]]
            if max(r, g, b) <= 1:
                r, g, b = int(r * 255), int(g * 255), int(b * 255)
            is_black = max(r, g, b) < 50
            return (r, g, b), is_black
        except (TypeError, ValueError):
            pass
    name = str(curve.get("color") or curve.get("name") or "")
    key = name.strip().lower()
    name_l = name.lower()
    for k, val in NAMED_RGB.items():
        kl = k.lower()
        if kl == key:
            return val, _is_black_name(k)
        # 避免把 "CR"、"H1.25-OR" 里的字母误当成 red/orange
        if len(kl) >= 2 and kl in name_l:
            # 要求左右不是字母，以免部分匹配
            import re as _re
            if _re.search(rf"(?<![a-z]){_re.escape(kl)}(?![a-z])", name_l):
                return val, _is_black_name(k)
        if k in name and not k.isascii():
            return val, _is_black_name(k)
    return None


def _rgb255_to_lab(arr: np.ndarray) -> np.ndarray:
    """Vectorized sRGB (0–255) → CIE Lab."""
    s = np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)
    lin = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float32,
    )
    xyz = lin @ m.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(np.maximum(xyz, 0.0)), (kappa * xyz + 16.0) / 116.0)
    lab = np.empty(arr.shape, dtype=np.float32)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def _lab_distance(arr: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    lab = _rgb255_to_lab(arr)
    tgt = _rgb255_to_lab(np.array(target, dtype=np.float32).reshape(1, 1, 3))[0, 0]
    return np.sqrt(((lab - tgt) ** 2).sum(axis=2))


def _morph_close_h(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """Horizontal close to reconnect compression breaks; keep height 1–2."""
    try:
        from scipy import ndimage as nd
    except Exception:
        return mask
    w = max(1, int(width))
    struct = np.ones((1, w), dtype=bool)
    return nd.binary_closing(mask, structure=struct, iterations=1)


def _morph_open(mask: np.ndarray, size: int = 2) -> np.ndarray:
    try:
        from scipy import ndimage as nd
    except Exception:
        return mask
    s = max(1, int(size))
    struct = np.ones((s, s), dtype=bool)
    return nd.binary_opening(mask, structure=struct, iterations=1)


def color_mask(
    rgb: np.ndarray,
    target: tuple[int, int, int],
    is_black: bool,
    *,
    rivals: list[tuple[tuple[int, int, int], bool]] | None = None,
    tight: bool = False,
) -> np.ndarray:
    """Pixels belonging to one legend color.

    Uses Lab distance (better near-green/near-blue separation than raw RGB)
    and, when rivals are given, winner-take-all with a margin.
    """
    arr = rgb.astype(np.float32)
    if is_black:
        mx = arr.max(axis=2)
        mn = arr.min(axis=2)
        chroma = mx - mn
        mask = (mx < (85 if tight else 100)) & (chroma < (32 if tight else 42))
        # do not let a saturated dark red/blue count as black if a rival wants it
        if rivals:
            for r_rgb, r_black in rivals:
                if r_black:
                    continue
                # high-chroma pixels closer to a named color stay with that color
                d = _lab_distance(arr, r_rgb)
                chroma_ok = chroma > 28
                mask = mask & ~(chroma_ok & (d < 35))
        return mask
    dist = _lab_distance(arr, target)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    mx = arr.max(axis=2)
    # base threshold: tighter when we must split similar hues
    thr = 28.0 if tight else 42.0
    if rivals:
        rival_ds = []
        for r_rgb, r_black in rivals:
            if r_black:
                continue
            if tuple(int(v) for v in r_rgb) == tuple(int(v) for v in target):
                continue
            rival_ds.append(_lab_distance(arr, r_rgb))
        if rival_ds:
            nearest_rival = np.min(np.stack(rival_ds, axis=0), axis=0)
            # pixel must be closer to target than to any rival, with a small margin
            margin = 4.0 if tight else 2.0
            exclusive = dist + margin <= nearest_rival
            # also shrink threshold if two legend colors are close
            pair = float(np.nanpercentile(nearest_rival[dist < 80], 5)) if np.any(dist < 80) else 40.0
            if np.isfinite(pair):
                thr = min(thr, max(16.0, 0.48 * pair))
            mask = (dist < thr) & exclusive & (chroma > 18) & (mx < 252)
            return mask
    return (dist < (thr + 6.0)) & (chroma > 18) & (mx < 252)


def _dominant_ink_color(
    rgb: np.ndarray,
    box: list[int],
    *,
    avoid: list[tuple[int, int, int]] | None = None,
) -> tuple[tuple[int, int, int], bool] | None:
    """Most common ink color in a box: black, or the median of the main chromatic cluster."""
    x0, y0, x1, y1 = [int(v) for v in box]
    H, W = rgb.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, max(x0 + 1, x1)), min(H, max(y0 + 1, y1))
    roi = rgb[y0:y1, x0:x1]
    if roi.size < 30:
        return None
    arr = roi.astype(np.float32)
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    chroma = mx - mn
    ink = (mx < 242) & ((chroma > 18) | (mx < 100))
    if int(ink.sum()) < 12:
        return None
    pix = arr[ink]
    ch = chroma[ink]
    black_frac = float(((ch < 28) & (pix.max(axis=1) < 100)).mean())
    # If mostly black ink, that's the curve color (stacked black plots, lipstick, MoS2).
    if black_frac >= 0.62 and float((ch > 28).mean()) < 0.28:
        return (25, 25, 25), True
    chrom = pix[ch > 28]
    if chrom.shape[0] < 8:
        return (25, 25, 25), True
    # k=2-ish: drop near-white leftover, take median of mid-chroma
    # snap away from already used colors if possible
    lab = _rgb255_to_lab(chrom.reshape(-1, 1, 3)).reshape(-1, 3)
    hsv_h = _hue_deg(chrom)
    sat = ch[ch > 28]
    weights = np.clip(sat, 1, None)
    hist, edges = np.histogram(hsv_h, bins=36, range=(0, 360), weights=weights)
    k = int(np.argmax(hist))
    center = 0.5 * (edges[k] + edges[k + 1])
    dh = np.abs((hsv_h - center + 180) % 360 - 180)
    pick = chrom[dh <= 20]
    if pick.shape[0] < 6:
        pick = chrom
    med = np.median(pick, axis=0)
    rgb_t = tuple(int(np.clip(v, 0, 255)) for v in med)
    if avoid:
        # if this median is extremely close to an already used chromatic color, try next hue peak
        for other in avoid:
            if _lab_distance(np.array(rgb_t, dtype=np.float32).reshape(1, 1, 3), other)[0, 0] < 14:
                hist[k] = 0
                if hist.max() > 0:
                    k2 = int(np.argmax(hist))
                    center = 0.5 * (edges[k2] + edges[k2 + 1])
                    dh = np.abs((hsv_h - center + 180) % 360 - 180)
                    pick = chrom[dh <= 20]
                    if pick.shape[0] >= 6:
                        med = np.median(pick, axis=0)
                        rgb_t = tuple(int(np.clip(v, 0, 255)) for v in med)
                break
    is_black = max(rgb_t) < 50 and (max(rgb_t) - min(rgb_t)) < 25
    return rgb_t, is_black


def _hue_deg(pix: np.ndarray) -> np.ndarray:
    """Hue in degrees [0, 360) for an (N,3) RGB array."""
    p = np.clip(pix.astype(np.float32) / 255.0, 0.0, 1.0)
    r, g, b = p[:, 0], p[:, 1], p[:, 2]
    mx = p.max(axis=1)
    mn = p.min(axis=1)
    c = mx - mn
    hue = np.zeros(len(p), dtype=np.float32)
    nz = c > 1e-6
    # red / green / blue sector
    rc = nz & (mx == r)
    gc = nz & (mx == g)
    bc = nz & (mx == b) & ~rc & ~gc
    hue[rc] = ((g[rc] - b[rc]) / c[rc]) % 6
    hue[gc] = ((b[gc] - r[gc]) / c[gc]) + 2.0
    hue[bc] = ((r[bc] - g[bc]) / c[bc]) + 4.0
    return hue * 60.0



def _column_coverage_stats(mask: np.ndarray, box: list[int]) -> tuple[float, int]:
    """Return (fraction of columns with ink, longest True run of such columns)."""
    x0, y0, x1, y1 = [int(v) for v in box]
    H, W = mask.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, max(x0 + 1, x1)), min(H, max(y0 + 1, y1))
    present = mask[y0:y1, x0:x1].any(axis=0)
    if present.size == 0:
        return 0.0, 0
    cover = float(present.mean())
    longest = 0
    for s, e in _runs_1d(present):
        longest = max(longest, e - s)
    return cover, int(longest)



def _near_gray(rgb_t: tuple[int, int, int], *, white: bool = False) -> bool:
    r, g, b = (int(v) for v in rgb_t)
    chroma = max(r, g, b) - min(r, g, b)
    if white:
        return min(r, g, b) >= 200 and chroma < 25
    return chroma < 22 and max(r, g, b) >= 80


def segment_stacked_color_bands(
    rgb: np.ndarray,
    frame: list[int],
    expected_n: int,
    ink: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Split a stacked plot into per-curve strips using ink gaps, then read each strip's color.

    Tall overlapping peaks can close gaps in a full-width profile, so we also try
    the left and right ends (usually baseline-only).
    """
    x0, y0, x1, y1 = [int(v) for v in frame]
    H, W = rgb.shape[:2]
    x0, y0 = max(0, x0 + 4), max(0, y0 + 2)
    x1, y1 = min(W, x1 - 4), min(H, y1 - 2)
    if x1 - x0 < 30 or y1 - y0 < 20 or expected_n < 2:
        return []
    if ink is None:
        ink = ink_mask(rgb)
    width = x1 - x0
    height = y1 - y0
    # Overlaid curves share one vertical span: a single long occupied block
    # on the full-width profile. Still try left/right baseline strips (Co peaks
    # overlap in y and would look "overlaid" on the full width).
    full = np.convolve(ink[y0:y1, x0:x1].mean(axis=1), np.ones(5) / 5.0, mode="same")
    occ = full > 0.02
    runs = _runs_1d(occ)
    long = [e - s for s, e in runs if (e - s) >= 0.20 * height]
    full_looks_overlaid = len(long) <= 1 and (not long or long[0] >= 0.55 * height)

    def _bands(xa: int, xb: int, floor: float, min_h: int) -> list[dict[str, Any]]:
        xa, xb = max(x0, xa), min(x1, xb)
        if xb - xa < 12:
            return []
        sl = ink[y0:y1, xa:xb].mean(axis=1)
        sm = np.convolve(sl, np.ones(5) / 5.0, mode="same")
        occupied = sm > floor
        out: list[dict[str, Any]] = []
        i = 0
        n = len(occupied)
        while i < n:
            if not occupied[i]:
                i += 1
                continue
            j = i + 1
            while j < n and occupied[j]:
                j += 1
            if (j - i) >= min_h:
                top, bot = y0 + i, y0 + j
                h = bot - top
                # color from the lower half (baseline), ignoring peak tops
                y_lo = top + int(0.50 * h)
                pix = rgb[y_lo:bot, x0:x1][ink[y_lo:bot, x0:x1]]
                if pix.shape[0] < 6:
                    pix = rgb[top:bot, x0:x1][ink[top:bot, x0:x1]]
                if pix.shape[0] < 6:
                    i = j
                    continue
                med = tuple(int(v) for v in np.median(pix, axis=0))
                if _near_gray(med, white=True):
                    i = j
                    continue
                is_black = max(med) < 80 and (max(med) - min(med)) < 30
                # y_med: lower envelope of the strip (the baseline)
                out.append({
                    "rgb": (25, 25, 25) if is_black else med,
                    "is_black": is_black or (_near_gray(med) and max(med) < 160),
                    "y_med": float(top + 0.75 * h),
                    "y_top": int(top),
                    "y_bot": int(bot),
                    "x_cover": 1.0,
                    "n": int(pix.shape[0]),
                })
            i = j
        # merge adjacent strips of essentially the same color (a curve split by a weak gap)
        merged: list[dict[str, Any]] = []
        for c in out:
            if merged:
                prev = merged[-1]
                d = float(_lab_distance(
                    np.array(prev["rgb"], dtype=np.float32).reshape(1, 1, 3), c["rgb"]
                )[0, 0])
                same = (prev["is_black"] and c["is_black"]) or d < 16
                if same and (c["y_top"] - prev["y_bot"]) <= max(8, int(0.04 * height)):
                    prev["y_bot"] = c["y_bot"]
                    prev["y_med"] = 0.5 * (prev["y_med"] + c["y_med"])
                    prev["n"] += c["n"]
                    continue
            merged.append(c)
        # drop a thin leftover gray strip at the very top (axis tick labels)
        if merged and merged[0]["y_bot"] - merged[0]["y_top"] < 0.08 * height and _near_gray(merged[0]["rgb"]):
            merged = merged[1:]
        return merged

    cands: list[tuple[float, list[dict[str, Any]]]] = []
    left_span = max(24, int(0.32 * width))
    right_span = max(24, int(0.28 * width))
    windows = [
        (x0, x0 + left_span),
        (x1 - right_span, x1),
    ]
    if not full_looks_overlaid:
        windows.append((x0, x1))
    elif expected_n <= 2:
        # truly overlaid (LDH 9b): do not invent stacked strips
        return []
    for xa, xb in windows:
        for floor in (0.015, 0.025, 0.04, 0.06):
            for min_h in (10, 14):
                bands = _bands(xa, xb, floor, min_h)
                if not bands:
                    continue
                # Drop a leftover gray/junk strip when we have one too many bands
                def _chroma(b):
                    r, g, bv = (int(v) for v in b["rgb"])
                    return max(r, g, bv) - min(r, g, bv)
                if expected_n and len(bands) == expected_n + 1:
                    # prefer dropping a near-gray end band (top labels or bottom PDF sticks)
                    ends = [0, len(bands) - 1]
                    end = min(ends, key=lambda k: (_chroma(bands[k]), -(bands[k]["y_bot"] - bands[k]["y_top"]) if k == 0 else 0))
                    if _chroma(bands[end]) < 28 or _near_gray(bands[end]["rgb"]) or bands[end]["is_black"]:
                        # only drop black/gray if another black remains or this end looks like junk
                        bands = [b for i, b in enumerate(bands) if i != end]
                n = len(bands)
                score = -abs(n - expected_n) * 8.0
                if n == expected_n:
                    score += 12.0
                # A gray strip at the TOP among saturated stacked curves is usually
                # axis/label junk (Co). A bottom black/gray curve is often real (TiO2 450).
                sat = [b for b in bands if not b["is_black"] and not _near_gray(b["rgb"])]
                if len(sat) >= 3 and bands and (_near_gray(bands[0]["rgb"]) or bands[0]["is_black"]):
                    score -= 8.0
                # distinct colors
                cols = [b["rgb"] for b in bands]
                distinct = 0
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        d = float(_lab_distance(
                            np.array(cols[i], dtype=np.float32).reshape(1, 1, 3), cols[j]
                        )[0, 0])
                        if d >= 18 or (bands[i]["is_black"] != bands[j]["is_black"]):
                            distinct += 1
                score += 0.3 * distinct
                cands.append((score, bands))
    if not cands:
        return []
    cands.sort(key=lambda z: -z[0])
    best = cands[0][1]
    # only accept if we are close to the expected count
    if expected_n and abs(len(best) - expected_n) > 1:
        return []
    return best


def discover_plot_colors(
    rgb: np.ndarray,
    frame: list[int],
    expected_n: int | None,
) -> list[dict[str, Any]]:
    """Find curve-like colors (long x-coverage), not compact PDF markers."""
    if expected_n and expected_n >= 2:
        stacked = segment_stacked_color_bands(rgb, frame, expected_n)
        if stacked and (not expected_n or abs(len(stacked) - expected_n) <= 1):
            return stacked
    x0, y0, x1, y1 = [int(v) for v in frame]
    H, W = rgb.shape[:2]
    x0i, x1i = max(0, x0 + 4), min(W, x1 - 4)
    y0i, y1i = max(0, y0 + 3), min(H, y1 - 4)
    if x1i - x0i < 20 or y1i - y0i < 16:
        return []
    arr = rgb[y0i:y1i, x0i:x1i].astype(np.float32)
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    chroma = mx - mn
    width = x1i - x0i
    clusters: list[dict[str, Any]] = []

    # black / dark-gray ink (anti-aliased black curves may have a little chroma)
    black = (mx < 115) & (chroma < 48)
    bcover, brun = _column_coverage_stats(black, [0, 0, black.shape[1], black.shape[0]])
    if bcover >= 0.30 or brun >= max(24, int(0.25 * width)):
        ys, xs = np.where(black)
        clusters.append({
            "rgb": (25, 25, 25),
            "is_black": True,
            "y_med": float(np.median(ys) + y0i) if ys.size else float((y0i + y1i) / 2),
            "x_cover": bcover,
            "n": int(black.sum()),
        })

    # chromatic hue peaks — stats are computed on a TIGHT hue window inside the ROI
    # so a loose Lab color_mask cannot pull in other curves and flatten y-centroids.
    colorful = (chroma > 30) & (mx < 250) & (mx > 25)
    if int(colorful.sum()) >= 12:
        pix = arr[colorful]
        hsv_h = _hue_deg(pix)
        # full-ROI hue map for tight masks (only where colorful)
        roi_h = np.zeros(arr.shape[:2], dtype=np.float32)
        flat = arr.reshape(-1, 3)
        roi_h.reshape(-1)[:] = _hue_deg(flat)
        weights = np.clip(chroma[colorful], 1, None)
        hist, edges = np.histogram(hsv_h, bins=36, range=(0, 360), weights=weights)
        peaks: list[tuple[float, float]] = []
        order = np.argsort(-hist)
        floor = max(1.0, float(hist.max()) * 0.008)
        for k in order:
            if hist[k] <= 0:
                continue
            center = 0.5 * (edges[k] + edges[k + 1])
            # do not merge red (~0°) with orange-brown (~20°), or navy with cyan
            if any(abs((center - p[1] + 180) % 360 - 180) < 12 for p in peaks):
                continue
            dh_bins = np.abs((np.arange(36) * 10 + 5 - center + 180) % 360 - 180)
            neigh = np.where(dh_bins <= 10)[0]
            score = float(hist[neigh].sum())
            if score < floor:
                continue
            peaks.append((score, center))
        peaks.sort(reverse=True)
        limit = max(expected_n or 1, 1) + 3
        avoid = [c["rgb"] for c in clusters if not c["is_black"]]
        for score, center in peaks[:limit]:
            tight = colorful & (np.abs((roi_h - center + 180.0) % 360.0 - 180.0) <= 14.0)
            if int(tight.sum()) < 12:
                continue
            pick = arr[tight]
            med = np.median(pick, axis=0)
            rgb_t = tuple(int(np.clip(v, 0, 255)) for v in med)
            # dark low-chroma leftovers belong to the black cluster
            if max(rgb_t) < 80 and (max(rgb_t) - min(rgb_t)) < 40:
                continue
            if any(float(_lab_distance(np.array(rgb_t, dtype=np.float32).reshape(1, 1, 3), o)[0, 0]) < 16 for o in avoid):
                continue
            cover, run = _column_coverage_stats(tight, [0, 0, tight.shape[1], tight.shape[0]])
            if not (cover >= 0.30 or (cover >= 0.18 and run >= max(24, int(0.20 * width)))):
                continue
            ys, xs = np.where(tight)
            if ys.size < 10:
                continue
            clusters.append({
                "rgb": rgb_t,
                "is_black": False,
                "y_med": float(np.median(ys) + y0i),
                "x_cover": cover,
                "n": int(ys.size),
            })
            avoid.append(rgb_t)
            if expected_n and len([c for c in clusters if not c["is_black"]]) >= expected_n + 1:
                break
    # keep the strongest expected_n clusters (prefer high coverage)
    if expected_n and len(clusters) > expected_n:
        clusters = sorted(clusters, key=lambda c: (-c["x_cover"], -c["n"]))[:expected_n]

    # Stacked plots: if a mid band's colour was missed globally, fill from that strip.
    if expected_n and expected_n >= 2 and len(clusters) < expected_n:
        height = max(1, y1i - y0i)
        used_rgb = [c["rgb"] for c in clusters]
        used_y = [c["y_med"] for c in clusters]
        for i in range(expected_n):
            top = y0i + int(height * i / expected_n)
            bot = y0i + int(height * (i + 1) / expected_n)
            # skip strips that already have a cluster
            if any(top - 4 <= yy <= bot + 4 for yy in used_y):
                continue
            if bot - top < 8:
                continue
            inf = _dominant_ink_color(rgb, [x0i, top, x1i, bot], avoid=[c for c in used_rgb if c != (25, 25, 25)])
            if inf is None:
                continue
            rgb_t, is_black = inf
            if is_black and any(c["is_black"] for c in clusters) and expected_n >= 3:
                # a second black in a chromatic stacked plot is usually axis ink; skip
                # unless this is the bottom band of an all-black stack (no chromatic yet)
                if any(not c["is_black"] for c in clusters):
                    continue
            if (not is_black) and any(float(_lab_distance(np.array(rgb_t, dtype=np.float32).reshape(1, 1, 3), o)[0, 0]) < 16 for o in used_rgb):
                continue
            # require some column coverage in this strip
            m = color_mask(rgb, rgb_t, is_black, tight=True)
            cover, run = _column_coverage_stats(m, [x0i, top, x1i, bot])
            if cover < 0.15 and run < 16:
                continue
            yy = float((top + bot) / 2)
            ys, xs = np.where(m[top:bot, x0i:x1i])
            if ys.size:
                yy = float(np.median(ys) + top)
            clusters.append({
                "rgb": tuple(int(v) for v in rgb_t),
                "is_black": bool(is_black),
                "y_med": yy,
                "x_cover": cover,
                "n": int(ys.size) if ys.size else 0,
            })
            used_rgb.append(tuple(int(v) for v in rgb_t))
            used_y.append(yy)
            if len(clusters) >= expected_n:
                break
    return clusters


def _y_centroids_stacked(clusters: list[dict[str, Any]], frame: list[int]) -> bool:
    if len(clusters) < 2:
        return False
    ys = sorted(c["y_med"] for c in clusters)
    height = max(1.0, float(frame[3] - frame[1]))
    gaps = np.diff(ys)
    # stacked: colours live in different horizontal strips
    return float(np.median(gaps)) >= max(10.0, 0.10 * height)


def refine_or_infer_curve_colors(
    rgb: np.ndarray,
    frame: list[int],
    curves: list[dict[str, Any]],
    bands: list[tuple[int, int, int]],
) -> tuple[list[dict[str, Any]], bool]:
    """Fill each curve's color from plot pixels. Returns (curves, overlaid)."""
    expected = max(len(curves), len(bands), 1)
    # Single experimental curves (lipstick, MoS2): trace ink/black after marker masks.
    # A slightly green-tinted black line must not be reduced to a chromatic remnant.
    if expected <= 1:
        out = [dict(c) if isinstance(c, dict) else {} for c in (curves or [{}])]
        if not out:
            out = [{"name": "曲线1", "stack_order": 1}]
        out[0]["color_rgb"] = [25, 25, 25]
        out[0]["color"] = "black"
        out[0].setdefault("name", "曲线1")
        out[0].setdefault("stack_order", 1)
        return out, False
    clusters = discover_plot_colors(rgb, frame, expected)
    overlaid = bool(clusters) and (not _y_centroids_stacked(clusters, frame)) and len(clusters) >= 2

    # If the VLM already named colors, snap each to the nearest discovered cluster.
    resolved_existing = [resolve_color(c) for c in curves]
    have_named = sum(1 for r in resolved_existing if r is not None)

    out: list[dict[str, Any]] = [dict(c) if isinstance(c, dict) else {} for c in curves]
    # Pad only up to the stated curve_count / named curves. Do not invent
    # extra traces just because colour discovery found leftover clusters (DEE).
    n_target = max(len(out), expected, 1)
    while len(out) < n_target:
        out.append({"name": f"曲线{len(out) + 1}", "stack_order": len(out) + 1})

    if clusters and (len(clusters) >= 2 or have_named == 0 or (len(curves) <= 1)):
        if overlaid:
            # match named/black first, then remaining chromatic clusters
            unused = list(clusters)

            def _take_black():
                for i, c in enumerate(unused):
                    if c["is_black"]:
                        return unused.pop(i)
                return None

            def _take_nearest(rgb_t, is_black):
                if is_black:
                    return _take_black()
                if not unused:
                    return None
                def d(c):
                    if c["is_black"]:
                        return 1e9
                    return float(_lab_distance(np.array(rgb_t, dtype=np.float32).reshape(1, 1, 3), c["rgb"])[0, 0])
                i = int(np.argmin([d(c) for c in unused]))
                return unused.pop(i)

            for i, curve in enumerate(out):
                res = resolve_color(curve)
                name = str(curve.get("name") or "").lower()
                picked = None
                if res is not None:
                    picked = _take_nearest(res[0], res[1])
                if picked is None and any(k in name for k in ("exp", "实验", "obs", "black", "黑")):
                    picked = _take_black()
                if picked is None and unused:
                    # leftover: prefer chromatic for sim/calc
                    if any(k in name for k in ("sim", "calc", "计算", "red", "红")):
                        chrom = [j for j, c in enumerate(unused) if not c["is_black"]]
                        picked = unused.pop(chrom[0] if chrom else 0)
                    else:
                        picked = unused.pop(0)
                if picked is not None:
                    curve["color_rgb"] = [int(v) for v in picked["rgb"]]
                    if picked["is_black"]:
                        curve["color"] = "black"
            return out, True
        else:
            # stacked: sort clusters top→bottom and assign in stack order
            ordered = sorted(clusters, key=lambda c: c["y_med"])
            # if we have more curve slots than clusters, keep even-band inference for the extras
            for i, curve in enumerate(out):
                if i < len(ordered):
                    c = ordered[i]
                    curve["color_rgb"] = [int(v) for v in c["rgb"]]
                    if c["is_black"]:
                        curve["color"] = "black"
                    if c.get("y_top") is not None:
                        curve["_band_top"] = int(c["y_top"])
                        curve["_band_bot"] = int(c.get("y_bot") or c["y_med"])
                    curve.setdefault("stack_order", i + 1)
                else:
                    # fallback: dominant color of that band
                    if i < len(bands):
                        top, base, _ = bands[i]
                        inf = _dominant_ink_color(rgb, [frame[0] + 4, int(top), frame[2] - 4, int(base)])
                        if inf is not None:
                            curve["color_rgb"] = [int(v) for v in inf[0]]
                            if inf[1]:
                                curve["color"] = "black"
                if not curve.get("name"):
                    curve["name"] = f"曲线{i + 1}"
                curve.setdefault("stack_order", i + 1)
            return out, False

    # fallback: per-band dominant color (old path), used when discovery failed
    used: list[tuple[int, int, int]] = []
    x0, y0, x1, y1 = frame
    for i, curve in enumerate(out):
        if i < len(bands):
            top, base, _ = bands[i]
            box = [x0 + 4, int(top), x1 - 4, int(base)]
        else:
            box = [x0 + 4, y0 + 4, x1 - 4, y1 - 4]
        inferred = _dominant_ink_color(rgb, box, avoid=used)
        resolved = resolve_color(curve)
        if resolved is not None and inferred is not None:
            d = float(_lab_distance(
                np.array(resolved[0], dtype=np.float32).reshape(1, 1, 3), inferred[0]
            )[0, 0])
            if inferred[1] and resolved[1]:
                snapped = inferred
            elif (not inferred[1]) and (not resolved[1]) and d < 55:
                snapped = inferred
            elif resolved[1] and not inferred[1]:
                snapped = inferred
            else:
                snapped = inferred if d < 28 else resolved
        else:
            snapped = inferred or resolved
        # Named chromatic (purple 400 °C, etc.) must not be replaced by axis-ink black
        # after text/phase masks wipe the thin anti-aliased stroke.
        if resolved is not None and not resolved[1] and snapped is not None and snapped[1]:
            snapped = resolved
        if snapped is not None:
            curve["color_rgb"] = [int(v) for v in snapped[0]]
            if snapped[1]:
                curve.setdefault("color", "black")
            if not snapped[1]:
                used.append(tuple(int(v) for v in snapped[0]))
        if not curve.get("name"):
            curve["name"] = f"曲线{i + 1}"
        curve.setdefault("stack_order", i + 1)
    return out, overlaid


def split_color_masks(
    rgb: np.ndarray,
    colors: list[tuple[tuple[int, int, int], bool] | None],
) -> list[np.ndarray | None]:
    """Per-legend-color masks, winner-take-all, with a short horizontal close."""
    masks: list[np.ndarray | None] = []
    present = [(i, c) for i, c in enumerate(colors) if c is not None]
    for i, c in enumerate(colors):
        if c is None:
            masks.append(None)
            continue
        rivals = [o for j, o in present if j != i]
        m = color_mask(rgb, c[0], c[1], rivals=rivals or None, tight=len(rivals) >= 1)
        if not c[1]:
            m = _morph_close_h(m, width=5)
        masks.append(m)
    return masks


def mask_grid_lines(rgb: np.ndarray, frame: list[int]) -> np.ndarray:
    """Wipe light-gray long grid lines; leave dark spines and curve baselines alone."""
    x0, y0, x1, y1 = [int(v) for v in frame]
    H, W = rgb.shape[:2]
    x0i, x1i = max(0, x0 + 2), min(W, x1 - 2)
    y0i, y1i = max(0, y0 + 2), min(H, y1 - 2)
    if x1i - x0i < 40 or y1i - y0i < 40:
        return rgb
    arr = rgb.astype(np.float32)
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    chroma = mx - mn
    # light gray: mid-bright, low chroma, not near-white paper and not dark ink
    light = (chroma < 18) & (mx >= 150) & (mx <= 230)
    width = x1i - x0i
    height = y1i - y0i
    out = rgb.copy()
    # long horizontal light lines (not a dark baseline)
    for y in range(y0i, y1i):
        row = light[y, x0i:x1i]
        if float(row.mean()) < 0.45:
            continue
        runs = _runs_1d(row)
        for s, e in runs:
            if (e - s) >= 0.55 * width:
                out[y, x0i + s : x0i + e] = 255
    # long vertical light lines
    for x in range(x0i, x1i):
        col = light[y0i:y1i, x]
        if float(col.mean()) < 0.45:
            continue
        runs = _runs_1d(col)
        for s, e in runs:
            if (e - s) >= 0.55 * height:
                out[y0i + s : y0i + e, x] = 255
    return out


def mask_phase_markers(rgb: np.ndarray, frame: list[int]) -> np.ndarray:
    """Erase compact chromatic PDF/JCPDS symbols (triangles, crosses, squares).

    Long chromatic strokes are real colored curves and are left intact.
    """
    try:
        from scipy import ndimage as nd
    except Exception:
        return rgb
    x0, y0, x1, y1 = [int(v) for v in frame]
    H, W = rgb.shape[:2]
    x0i, x1i = max(0, x0 + 1), min(W, x1 - 1)
    y0i, y1i = max(0, y0 + 1), min(H, y1 - 1)
    if x1i - x0i < 20 or y1i - y0i < 20:
        return rgb
    arr = rgb.astype(np.float32)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    mx = arr.max(axis=2)
    colorful = (chroma > 28) & (mx < 250)
    roi = np.zeros((H, W), dtype=bool)
    roi[y0i:y1i, x0i:x1i] = colorful[y0i:y1i, x0i:x1i]
    labeled, n = nd.label(roi)
    if n == 0:
        return rgb
    out = rgb.copy()
    width = max(1, x1i - x0i)
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if ys.size == 0:
            continue
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        area = int(ys.size)
        # long horizontal / spectral stroke: keep
        if w >= max(28, 0.06 * width) and w >= 2.2 * h:
            continue
        # compact marker (triangle/square/cross/plus). Do not treat a tall
        # thin stem as compact — that is an isolated coloured peak (lipstick
        # chlorite ~7.26°), which the vstick-comb rule below will keep.
        compact = (area <= 220) and (w <= 22) and (h <= 22) and (h <= 2.0 * w or w >= 6)
        # tall thin colourful stroke: only erase if it is part of a comb of
        # similar verticals (PDF/JCPDS card sticks). An isolated magenta/red
        # peak (lipstick chlorite ~7.26°) must be kept.
        vstick = (h >= 10) and (w <= 5) and (h >= 2.5 * w) and (area <= 0.7 * h * w + 8)
        if vstick:
            cx = float(xs.mean())
            n_nb = 0
            for j in range(1, n + 1):
                if j == i:
                    continue
                ys2, xs2 = np.where(labeled == j)
                if ys2.size == 0:
                    continue
                w2 = int(xs2.max() - xs2.min() + 1)
                h2 = int(ys2.max() - ys2.min() + 1)
                if w2 > 6 or h2 < 8:
                    continue
                if abs(float(xs2.mean()) - cx) <= 42 and abs(h2 - h) <= max(12, 0.6 * h):
                    n_nb += 1
            if n_nb < 2:
                vstick = False
        if compact or vstick:
            y0b, y1b = max(0, ys.min() - 1), min(H, ys.max() + 2)
            x0b, x1b = max(0, xs.min() - 1), min(W, xs.max() + 2)
            out[y0b:y1b, x0b:x1b][roi[y0b:y1b, x0b:x1b]] = 255
            out[ys, xs] = 255
    return out


def apply_pixel_masks(
    rgb: np.ndarray,
    boxes_1000: list[dict[str, Any]] | None,
    frame: list[int] | None = None,
) -> np.ndarray:
    out = rgb.copy()
    H, W = out.shape[:2]
    draw = Image.fromarray(out)
    painter = ImageDraw.Draw(draw)
    for rec in boxes_1000 or []:
        box = rec.get("box_1000") if isinstance(rec, dict) else rec
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        try:
            x0, y0, x1, y1 = [float(v) for v in box[:4]]
        except (TypeError, ValueError):
            continue
        if max(x0, x1) <= 1000 and max(y0, y1) <= 1000:
            x0, x1 = x0 / 1000.0 * W, x1 / 1000.0 * W
            y0, y1 = y0 / 1000.0 * H, y1 / 1000.0 * H
        bw, bh = abs(x1 - x0), abs(y1 - y0)
        # A "label" box covering ~13% of the figure (or a wide bottom/top strip)
        # will swallow a stacked curve (materials 400 °C). Keep only compact boxes.
        if (bw * bh) / max(W * H, 1) > 0.06:
            continue
        if frame and len(frame) == 4:
            fw = max(1.0, float(frame[2] - frame[0]))
            fh = max(1.0, float(frame[3] - frame[1]))
            if bw >= 0.50 * fw and bh >= 0.14 * fh:
                continue
        painter.rectangle([int(x0), int(y0), int(x1), int(y1)], fill=(255, 255, 255))
    out = np.asarray(draw)
    return out


def infer_floor_baseline(ink: np.ndarray, box: list[int]) -> int | None:
    """单谱图：用每列底轴上方最靠下的墨点估计曲线基线（允许基线是嘈杂的，不必成一整条横线）。"""
    x0, y0, x1, y1 = box
    H, W = ink.shape
    x0i, x1i = max(0, x0 + 3), min(W, x1 - 3)
    if x1i - x0i < 20:
        return None
    # 跳过底轴厚度
    axis_skip = 6
    floors: list[int] = []
    for x in range(x0i, x1i):
        for y in range(min(H - 1, y1 - axis_skip), y0 + 4, -1):
            if ink[y, x]:
                floors.append(int(y))
                break
    if len(floors) < 8:
        return None
    # 用偏下的分位，避免文字遮罩后峰列把基线抬高、矮峰落到框外
    p80 = float(np.percentile(floors, 80))
    p92 = float(np.percentile(floors, 92))
    return int(round(p92 if (p92 - p80) > 8 else p80))


def detect_baselines(ink: np.ndarray, box: list[int]) -> list[int]:
    """框内近似水平的基线（不含顶/底坐标轴）。"""
    x0, y0, x1, y1 = box
    H, W = ink.shape
    x0i, x1i = max(0, x0 + 3), min(W, x1 - 3)
    y0i, y1i = max(0, y0 + 4), min(H, y1 - 3)
    if x1i - x0i < 20 or y1i - y0i < 20:
        return []
    width = x1i - x0i
    rows = []
    for y in range(y0i, y1i):
        runs = _runs_1d(ink[y, x0i:x1i])
        if not runs:
            continue
        longest = max(e - s for s, e in runs)
        if longest >= 0.45 * width:
            rows.append(y)
    if not rows:
        return []
    # 聚成线
    groups = [[rows[0]]]
    for y in rows[1:]:
        if y - groups[-1][-1] <= 3:
            groups[-1].append(y)
        else:
            groups.append([y])
    levels = [int(round(float(np.median(g)))) for g in groups if len(g) >= 2 or (g and True)]
    # 不要把底轴和最下一条谱的基线合并
    levels = [y for y in levels if y0 + 8 <= y <= y1 - 4]
    merged: list[int] = []
    for y in levels:
        if merged and y - merged[-1] < 8:
            merged[-1] = (merged[-1] + y) // 2
        else:
            merged.append(y)
    return merged


def _curves_meta(calib: dict[str, Any]) -> list[dict[str, Any]]:
    curves = calib.get("curves") if isinstance(calib.get("curves"), list) else []
    out = [c for c in curves if isinstance(c, dict)]
    if out and all(isinstance(c.get("stack_order"), (int, float)) for c in out):
        out = sorted(out, key=lambda c: float(c.get("stack_order") or 0))
    return out


def _trace_column(
    mask: np.ndarray,
    x: int,
    y_top: int,
    y_base: int,
    prev_y: float | None,
    attach: int = 8,
    *,
    missing: float | None = None,
) -> float:
    missing = float(y_base) if missing is None else missing
    if x < 0 or x >= mask.shape[1]:
        return missing
    y0 = max(0, y_top + 1)
    y1 = min(mask.shape[0], y_base + 1)
    if y1 <= y0:
        return missing
    runs = _runs_1d(mask[y0:y1, x])
    if not runs:
        return missing
    band = max(y_base - y_top, 1)
    candidates = []
    for s, e in runs:
        top = y0 + s
        bot = y0 + e - 1
        height = e - s
        # Sharp isolated peaks may have a wiped foot (coloured stem / triangle
        # mask) a few tens of pixels above the floor; still treat them as peaks.
        near_base = bot >= y_base - attach or (height >= 15 and bot >= y_base - 40)
        # 上半部短笔画多半是文字 / hkl 残块（lipstick 10.8° 数字梯子）。
        # 真正连到基线的峰不受此限制。
        if (not near_base) and top < y_top + 0.50 * band and height < 22:
            continue
        candidates.append((top, bot, height, near_base))
    if not candidates:
        return missing
    # 几乎顶满整条带的竖块是图例框/填充，不是谱线
    slim = [c for c in candidates if not (c[2] >= 0.72 * band and c[0] <= y_top + 0.10 * band)]
    if slim:
        candidates = slim
    near = [c for c in candidates if c[3]]
    if near:
        return float(min(c[0] for c in near))
    if prev_y is not None and np.isfinite(prev_y):
        return float(min(candidates, key=lambda c: abs(c[0] - prev_y))[0])
    # 取最靠近基线的笔画顶
    return float(min(candidates, key=lambda c: y_base - c[1])[0])


def _color_track(
    mask: np.ndarray,
    frame: list[int],
    columns: np.ndarray,
    *,
    y_top: int | None = None,
    y_base: int | None = None,
) -> tuple[int, np.ndarray, np.ndarray] | None:
    """Trace one color across the frame; fill only short interior gaps that have two-sided evidence."""
    _, top_f, _, bottom_f = frame
    top = int(top_f + 2 if y_top is None else y_top)
    bottom = int(bottom_f - 2 if y_base is None else y_base)
    if bottom - top < 8 or columns.size < 16:
        return None
    xa, xb = int(columns[0]), int(columns[-1]) + 1
    region = mask[top:bottom, xa:xb]
    if not region.size:
        return None
    covered = region.any(axis=0)
    cover = float(covered.mean())
    if cover < 0.35:
        return None
    # floor from the lower envelope of color pixels
    floors = []
    sub = region[:, covered]
    if sub.size:
        # last True from the bottom
        floors = bottom - 1 - np.argmax(sub[::-1, :], axis=0)
        base = min(bottom - 1, int(np.percentile(floors, 95)) + 1)
        if np.mean(np.abs(floors - base) <= 6) < 0.30 and cover < 0.7:
            # color may be a mid-band curve whose "floor" is the stroke itself
            base = min(bottom - 1, int(np.percentile(floors, 90)) + 2)
    else:
        return None
    ys = []
    prev = None
    for x in columns:
        y = _trace_column(mask, int(x), top, base, prev, missing=np.nan)
        ys.append(y)
        if np.isfinite(y):
            prev = y
    ys = np.asarray(ys, dtype=float)
    valid = np.isfinite(ys)
    if float(valid.mean()) < 0.35:
        return None
    first, last = np.flatnonzero(valid)[[0, -1]]
    # reconnect short interior holes; leave long mid-range holes to _reconnect_color_trace
    max_gap = max(8, min(36, round(0.04 * len(columns))))
    hole = ~valid[first:last + 1]
    # interpolate only gaps <= max_gap
    out = ys.copy()
    for s, e in _runs_1d(hole):
        # s,e relative to slice [first, last]
        if e - s <= max_gap:
            continue  # interp later
        # leave long gaps as nan; caller may reconnect using mask evidence
    xs = columns[first:last + 1]
    filled = out[first:last + 1].copy()
    vloc = np.isfinite(filled)
    if int(vloc.sum()) >= 2:
        filled[~vloc] = np.interp(
            np.flatnonzero(~vloc),
            np.flatnonzero(vloc),
            filled[vloc],
        )
        # but restore long gaps to nan so we don't invent a straight baseline through a real hole
        for s, e in _runs_1d(hole):
            if e - s > max_gap:
                filled[s:e] = np.nan
    return base, xs, filled


def _is_boxy_or_holed(intens: np.ndarray) -> tuple[bool, bool]:
    y = np.asarray(intens, dtype=float)
    if y.size < 16:
        return False, False
    dy = np.abs(np.diff(y))
    boxy = float((dy < 1e-6).mean()) >= 0.82
    # long constant plateaus with abrupt jumps also count as boxy
    if not boxy:
        runs_const = 0
        i = 0
        while i < len(dy):
            if dy[i] < 1e-6:
                j = i + 1
                while j < len(dy) and dy[j] < 1e-6:
                    j += 1
                if (j - i) >= max(12, int(0.04 * len(y))):
                    runs_const += 1
                i = j
            else:
                i += 1
        boxy = runs_const >= 4 and float((dy < 1e-6).mean()) >= 0.65
    n = len(y)
    mid = y[int(0.3 * n): int(0.7 * n)]
    ends = np.concatenate([y[: int(0.15 * n)], y[int(0.85 * n) :]]) if n >= 20 else y
    hole = bool(ends.size and float(ends.max()) > 4 and mid.size and float((mid <= 1e-6).mean()) >= 0.75)
    return boxy, hole


def _reconnect_color_trace(
    mask: np.ndarray,
    columns: np.ndarray,
    y_top: int,
    y_base: int,
    ys: np.ndarray,
) -> np.ndarray:
    """If a column has color pixels but the trace sat at the floor (or NaN), pick them up.

    Mid-range holes get filled only when that color is actually present in the column.
    """
    out = np.array(ys, dtype=float)
    n = len(columns)
    if n != len(out):
        return out
    prev = None
    for i, x in enumerate(columns):
        cur = out[i]
        at_floor = (not np.isfinite(cur)) or (cur >= y_base - 1.5)
        if not at_floor:
            prev = cur
            continue
        y = _trace_column(mask, int(x), y_top, y_base, prev, missing=np.nan)
        if np.isfinite(y) and y < y_base - 1.0:
            out[i] = y
            prev = y
    # interpolate remaining short interior NaNs / floor dips that are isolated
    valid = np.isfinite(out) & (out < y_base - 0.8)
    if int(valid.sum()) >= 2:
        first, last = np.flatnonzero(valid)[[0, -1]]
        max_gap = max(8, min(40, round(0.05 * n)))
        hole = ~valid
        filled = out.copy()
        filled[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), out[valid])
        for s, e in _runs_1d(hole):
            # only fill interior short gaps (not leading/trailing empty)
            if s <= first or e - 1 >= last:
                continue
            if e - s <= max_gap:
                # confirm at least one column in the gap actually has color, or gap is very short
                has = False
                for j in range(s, e):
                    xx = int(columns[j])
                    sl = mask[y_top:y_base + 1, xx] if 0 <= xx < mask.shape[1] else None
                    if sl is not None and sl.any():
                        has = True
                        break
                if has or (e - s) <= 6:
                    out[s:e] = filled[s:e]
    return out


def _finalize_intensity(
    xs_px: np.ndarray,
    ys_px: np.ndarray,
    base: float,
    a: float,
    bcoef: float,
    name: str,
    color: tuple[int, int, int] | None,
) -> dict[str, Any]:
    finite = np.isfinite(ys_px)
    if finite.any():
        y_use = np.array(ys_px, dtype=float)
        # leading/trailing nan → baseline (do not invent an extension)
        y_use[~finite] = float(base)
    else:
        y_use = np.full_like(np.asarray(ys_px, dtype=float), float(base), dtype=float)
        xs_px = np.asarray(xs_px, dtype=float)
    y_clean = clean_y(y_use, float(base))
    intens = (float(base) - y_clean).clip(min=0)
    edge = 16
    try:
        edge = max(10, int(round(1.6 / abs(float(bcoef))))) if abs(float(bcoef)) > 1e-9 else 16
    except (TypeError, ValueError):
        edge = 16
    intens = _remove_vertical_spikes(intens, width=6, edge_only_px=edge)
    two_theta = a + bcoef * np.asarray(xs_px, dtype=float)
    xg, yg = resample_xy(two_theta, intens, GRID_DEG)
    ypx_g = np.interp(xg, two_theta, y_clean) if len(two_theta) else y_clean
    return {
        "name": name,
        "x": xg,
        "y": yg,
        "y_px": ypx_g,
        "x_px": np.interp(xg, two_theta, np.asarray(xs_px, dtype=float)) if len(two_theta) else np.asarray(xs_px),
        "color": color,
        "baseline_px": int(base),
        "peaks": find_trace_peaks(xg, yg),
    }


def _trace_one_band(
    mask: np.ndarray,
    ink: np.ndarray,
    columns: np.ndarray,
    frame: list[int],
    top: int,
    base: int,
    resolved: tuple[tuple[int, int, int], bool] | None,
    unique_color: bool,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Trace a single curve from its color (or ink) mask, reconnecting boxy/hole results."""
    track = None
    if resolved is not None and unique_color and not resolved[1]:
        track = _color_track(mask, frame, columns, y_top=top, y_base=base)
    if track is not None:
        base_t, xs, ys = track
        ys2 = _reconnect_color_trace(mask, xs, top, base_t, ys)
        intens = np.where(np.isfinite(ys2), np.clip(base_t - ys2, 0, None), 0.0)
        boxy, holed = _is_boxy_or_holed(intens)
        if boxy or holed:
            ys2 = _reconnect_color_trace(mask, xs, top, base_t, ys2)
        return int(base_t), np.asarray(xs, dtype=float), np.asarray(ys2, dtype=float)

    xs = np.asarray(columns, dtype=float)
    ys = []
    prev = None
    for x in columns:
        yp = _trace_column(mask, int(x), top, base, prev)
        yp = min(max(yp, float(top)), float(base))
        prev = yp
        ys.append(prev)
    ys = np.asarray(ys, dtype=float)
    intens = (float(base) - ys).clip(min=0)
    boxy, holed = _is_boxy_or_holed(intens)
    if (boxy or holed) and resolved is not None:
        # retry along the color band with NaN gaps + reconnect, instead of keeping a rectangle / zero hole
        ys_n = []
        prev = None
        for x in columns:
            yp = _trace_column(mask, int(x), top, base, prev, missing=np.nan)
            ys_n.append(yp)
            if np.isfinite(yp):
                prev = yp
        ys2 = _reconnect_color_trace(mask, xs, top, base, np.asarray(ys_n, dtype=float))
        intens2 = np.where(np.isfinite(ys2), np.clip(base - ys2, 0, None), 0.0)
        boxy2, holed2 = _is_boxy_or_holed(intens2)
        # adopt if we recovered mid-range signal or reduced boxiness
        mid = slice(int(0.3 * len(intens2)), int(0.7 * len(intens2)))
        recovered = float(intens2[mid].max() if intens2.size else 0) > float(intens[mid].max() if intens.size else 0) + 2
        if recovered or (boxy and not boxy2) or (holed and not holed2):
            ys = ys2
    return int(base), xs, ys


def clean_y(y: np.ndarray, y_base: float) -> np.ndarray:
    """去掉单列尖刺和纯竖线伪迹，保持 y(x) 连续。"""
    y = np.array(y, dtype=float)
    n = len(y)
    if n < 5:
        return y
    out = y.copy()
    for i in range(1, n - 1):
        # 只抹掉 1 px 宽的矮毛刺；XRD 锐峰在低分辨率下也常只有 1 px 宽，必须保留
        left = y[i - 1]
        right = y[i + 1]
        mid = y[i]
        h = y_base - mid
        if 2 <= h <= 8 and (y_base - left) < 3 and (y_base - right) < 3:
            out[i] = 0.5 * (left + right)
    # 3 点中值抑制噪声，但保留高锐峰（含单像素峰）
    med = out.copy()
    for i in range(1, n - 1):
        med[i] = float(np.median(out[i - 1 : i + 2]))
        if (y_base - out[i]) >= 8:
            med[i] = out[i]
    return med


def resample_xy(x: np.ndarray, y: np.ndarray, step: float = GRID_DEG) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < 2:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    # 去掉重复 x
    uniq_x, idx = np.unique(x, return_index=True)
    uniq_y = y[idx]
    if len(uniq_x) < 2:
        return uniq_x, uniq_y
    start = np.ceil(uniq_x[0] / step) * step
    stop = np.floor(uniq_x[-1] / step) * step
    if stop < start:
        return uniq_x, uniq_y
    grid = np.arange(start, stop + 0.5 * step, step)
    yg = np.interp(grid, uniq_x, uniq_y)
    return grid, yg


def find_trace_peaks(x: np.ndarray, y: np.ndarray, min_prom_frac: float = 0.18, min_dist_deg: float = 0.8) -> list[dict[str, float]]:
    if len(x) < 5:
        return []
    y0 = y - np.nanmin(y)
    span = float(np.nanmax(y0)) if np.isfinite(y0).any() else 0.0
    if span <= 1e-9:
        return []
    noise = float(np.percentile(y0, 50))
    thr = max(min_prom_frac * span, 3.0 * max(noise, 1.0), 4.0)
    idx = []
    for i in range(2, len(y) - 2):
        if y0[i] >= y0[i - 1] and y0[i] >= y0[i + 1] and y0[i] >= y0[i - 2] and y0[i] >= y0[i + 2] and y0[i] >= thr:
            idx.append(i)
    kept: list[int] = []
    for i in sorted(idx, key=lambda j: -y0[j]):
        if all(abs(x[i] - x[k]) >= min_dist_deg for k in kept):
            kept.append(i)
    kept = sorted(kept)
    return [{"two_theta": float(x[i]), "intensity": float(y[i]), "prominence": float(y0[i])} for i in kept]


def crosscheck_peaks(
    peaks: list[dict[str, float]],
    hints: list[dict[str, Any]] | None,
    tol: float = PEAK_MISMATCH_DEG,
) -> dict[str, Any]:
    """与提示峰位比较；系统偏差时可给出平移建议。精确峰位以描点为准。"""
    hint_vals = []
    for h in hints or []:
        if isinstance(h, dict):
            v = h.get("value", h.get("two_theta", h.get("twoTheta")))
        else:
            v = h
        try:
            hint_vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not hint_vals or not peaks:
        return {"hints": hint_vals, "matches": [], "mismatches": [], "shift_suggestion": None}
    pxs = [p["two_theta"] for p in peaks]
    matches, mismatches = [], []
    deltas = []
    for hv in hint_vals:
        nearest = min(pxs, key=lambda t: abs(t - hv))
        d = nearest - hv
        rec = {"hint": hv, "digitized": nearest, "delta": d}
        if abs(d) <= tol:
            matches.append(rec)
        else:
            mismatches.append(rec)
        deltas.append(d)
    shift = float(np.median(deltas)) if deltas else 0.0
    # 仅当多数提示峰有近似常偏移时建议平移定标
    if len(deltas) >= 3 and abs(shift) > tol and float(np.median(np.abs(np.array(deltas) - shift))) < 0.25:
        suggestion = shift
    else:
        suggestion = None
    return {
        "hints": hint_vals,
        "matches": matches,
        "mismatches": mismatches,
        "shift_suggestion": suggestion,
    }


def _hint_peaks(calib: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("peaks_hint", "peak_hints", "labeled_peaks"):
        v = calib.get(key)
        if isinstance(v, list):
            return v
    return []



def mask_text_blobs(rgb: np.ndarray, frame: list[int]) -> np.ndarray:
    """去掉数据框内短而宽的文字块和细小相标记，避免被描成假峰。"""
    try:
        from scipy import ndimage as nd
    except Exception:
        return rgb
    x0, y0, x1, y1 = frame
    ink = ink_mask(rgb)
    # 先去掉长横线（基线/坐标轴），避免化学式残片与基线连成一块而不被当成文字
    stripped = ink.copy()
    x0i, x1i = x0 + 2, max(x0 + 3, x1 - 2)
    y0i, y1i = y0 + 2, max(y0 + 3, y1 - 2)
    width = max(1, x1i - x0i)
    for y in range(y0i, y1i):
        for s, e in _runs_1d(ink[y, x0i:x1i]):
            if (e - s) >= max(24, 0.22 * width):
                stripped[y, x0i + s : x0i + e] = False
    roi = np.zeros_like(ink)
    roi[y0i:y1i, x0i:x1i] = stripped[y0i:y1i, x0i:x1i]
    labeled, n = nd.label(roi)
    out = rgb.copy()
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if ys.size == 0:
            continue
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        area = int(ys.size)
        # 文字通常比高更宽；细高块是衍射峰，不能当文字擦掉
        textish = (
            8 <= w <= 200 and 5 <= h <= 36 and w >= h + 2
            and area < 0.72 * w * h + 8
        )
        marker = 3 <= w <= 16 and 3 <= h <= 16 and area <= 90 and w <= h + 6
        if textish or marker:
            # 略外扩 1 px，吃掉抗锯齿残边
            y0b, y1b = max(0, ys.min() - 1), min(out.shape[0], ys.max() + 2)
            x0b, x1b = max(0, xs.min() - 1), min(out.shape[1], xs.max() + 2)
            out[y0b:y1b, x0b:x1b][stripped[y0b:y1b, x0b:x1b]] = 255
            out[ys, xs] = 255
            continue
        # 化学式残片贴在峰肩上：上半截明显比下半截的细“茎”更宽，只擦掉宽的头部
        if h >= 24 and w >= 16:
            y_hi, y_lo = int(ys.min()), int(ys.max())
            stem_from = y_hi + int(0.65 * h)
            row_w = []
            stem_ws = []
            for yy in range(y_hi, y_lo + 1):
                ww = int(np.count_nonzero(xs[ys == yy]))
                row_w.append((yy, ww))
                if yy >= stem_from:
                    stem_ws.append(ww)
            stem = float(np.median(stem_ws)) if stem_ws else 0.0
            max_w = max((ww for _, ww in row_w), default=0)
            if stem >= 1 and max_w >= 16 and max_w >= 3.0 * (stem + 1):
                for yy, ww in row_w:
                    if ww >= max(12, 2.5 * (stem + 1)):
                        cols = xs[ys == yy]
                        out[yy, cols] = 255
    return out


def trace_bands(
    rgb: np.ndarray,
    frame: list[int],
    calib: dict[str, Any],
    fit: dict[str, Any],
    rgb_color: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Split the plot into per-legend-color masks, trace each color, then merge.

    Falls back to ink-in-band tracing when a curve has no usable color. Frame / tick
    calibration is the caller's job and is left unchanged.

    rgb_color, if given, is the plot after only conservative masks (legend boxes,
    outside-frame, grid). Colour discovery runs on that image so text/phase/stick
    whitening cannot erase a thin bottom curve before it is found. Tracing still
    punches out pixels that later masks painted white.
    """
    x0, y0, x1, y1 = frame
    color_img = rgb if rgb_color is None else rgb_color
    x0i, x1i = x0 + 4, x1 - 4
    ink = ink_mask(rgb)
    baselines = detect_baselines(ink, frame)
    curves = _curves_meta(calib)
    expected = calib.get("curve_count")
    try:
        expected = int(expected) if expected is not None else None
    except (TypeError, ValueError):
        expected = None

    # Honor expected count (P25 / 400 °C / H1.25-OR). Also keep at least as many as named curves.
    expected_n = max(expected or 0, len(curves))
    if expected_n and len(baselines) != expected_n:
        if expected_n >= 1:
            height = y1 - y0
            baselines = [int(y0 + height * (i + 1) / expected_n) for i in range(expected_n)]
            detected = detect_baselines(ink, frame)
            used = []
            for b in baselines:
                near = [d for d in detected if abs(d - b) < height / (2 * expected_n)]
                used.append(int(np.median(near)) if near else b)
            baselines = used
            if expected_n == 1 and abs(baselines[0] - y1) <= 6:
                floor = infer_floor_baseline(ink, frame)
                if floor is not None and floor < y1 - 6:
                    baselines = [floor]
    if expected_n and baselines and len(baselines) < expected_n and (y1 - baselines[-1]) > 16:
        while len(baselines) < expected_n and (y1 - baselines[-1]) > 16:
            floor = infer_floor_baseline(ink, frame)
            baselines.append(int(floor if floor is not None else (y1 - 8)))
            break
    if not baselines:
        floor = infer_floor_baseline(ink, frame)
        baselines = [floor if floor is not None else (y1 - 8)]
    if baselines and abs(baselines[-1] - y1) <= 8:
        floor = infer_floor_baseline(ink, frame)
        if floor is not None and floor < y1 - 6:
            baselines[-1] = floor

    bands = []
    prev_top = y0 + 2
    for i, b in enumerate(baselines):
        top = prev_top
        if i > 0:
            top = baselines[i - 1] + 2
        bands.append((top, int(b), i))
        prev_top = b

    # Infer / snap colors from pixels so offline redigitize still gets a color split
    # when figures_index dropped color_rgb.
    curves, overlaid = refine_or_infer_curve_colors(rgb, frame, curves, bands)
    # stacked plots with 3+ expected curves must not be treated as overlaid
    if (expected_n or 0) >= 3:
        overlaid = False
    colors = [resolve_color(c) for c in curves]
    # Prefer the color-derived strips over even y-splits when we have them.
    if not overlaid and all(c.get("_band_top") is not None for c in curves) and len(curves) >= 2:
        bands = []
        for i, c in enumerate(curves):
            top = int(c["_band_top"])
            bot = int(c.get("_band_bot") or (top + 8))
            # include peak space above the baseline: stretch up to the previous band
            if i == 0:
                top = min(top, y0 + 2)
            else:
                top = min(top, bands[i - 1][1] + 2)
            bands.append((top, max(bot, top + 8), i))
    # If discovery produced extra stacked curves, extend bands to match.
    elif not overlaid and len(curves) > len(bands) and len(curves) >= 2:
        height = y1 - y0
        n = len(curves)
        baselines = [int(y0 + height * (i + 1) / n) for i in range(n)]
        bands = []
        prev_top = y0 + 2
        for i, b in enumerate(baselines):
            top = prev_top if i == 0 else baselines[i - 1] + 2
            bands.append((top, int(b), i))
            prev_top = b

    # Colour identity comes from color_img, but the trace mask is taken on the
    # already-whitened tracing image so peak cores are not stolen or punched out.
    masks = split_color_masks(rgb, colors)
    traces: list[dict[str, Any]] = []
    a, bcoef = fit["a"], fit["b"]
    columns = np.arange(x0i, x1i + 1)
    # unique non-black colors can be tracked independently
    color_keys = []
    for resolved in colors:
        if resolved is None:
            color_keys.append(None)
        else:
            color_keys.append((tuple(int(v) for v in resolved[0]), bool(resolved[1])))

    for top, base, i in bands:
        curve = curves[i] if i < len(curves) else {}
        name = str(curve.get("name") or f"曲线{i + 1}").strip() or f"曲线{i + 1}"
        resolved = colors[i] if i < len(colors) else None
        unique_color = bool(
            resolved is not None
            and not resolved[1]
            and color_keys.count(color_keys[i]) == 1
        )
        mask = masks[i] if i < len(masks) and masks[i] is not None else None
        use_top, use_base = int(top), int(base)
        if overlaid and unique_color:
            # same shared floor; allow the color to use the full data height
            floor = infer_floor_baseline(ink, frame)
            use_base = int(floor if floor is not None else (y1 - 8))
            use_top = y0 + 2
        if mask is not None:
            if resolved is not None and resolved[1]:
                # Coloured cores of a black curve (lipstick chlorite is magenta)
                # sit on the dark stroke. Include high-chroma ink that touches
                # the black mask; do not swallow a separate green baseline.
                try:
                    from scipy import ndimage as ndi
                    arr = rgb.astype(np.float32)
                    chroma = arr.max(axis=2) - arr.min(axis=2)
                    extra = ink & (chroma > 28)
                    dil = ndi.binary_dilation(mask, iterations=4)
                    mask = mask | (extra & dil)
                except Exception:
                    pass
                band_mask = np.zeros(mask.shape, dtype=bool)
                band_mask[use_top:use_base + 1, x0i:x1i] = True
                mask = mask & band_mask
            sl = mask[use_top:use_base + 1, x0i:x1i]
            if sl.size == 0 or float(np.mean(sl)) < 0.002:
                mask = ink
        else:
            mask = ink
        base_o, xs_px, ys_px = _trace_one_band(
            mask, ink, columns if not (unique_color and overlaid) else columns,
            [x0, use_top, x1, use_base] if overlaid and unique_color else frame,
            use_top, use_base, resolved, unique_color,
        )
        traces.append(
            _finalize_intensity(
                xs_px, ys_px, float(base_o), a, bcoef, name,
                (resolved[0] if resolved else None),
            )
        )
    return traces


def _remove_vertical_spikes(intens: np.ndarray, width: int = 2, edge_only_px: int = 0) -> np.ndarray:
    y = np.array(intens, dtype=float)
    n = len(y)
    if n < 6:
        return y
    floor = np.percentile(y, 20)
    for i in range(1, n - 1):
        if edge_only_px and not (i <= edge_only_px or i >= n - 1 - edge_only_px):
            continue
        if y[i] <= floor + 2:
            continue
        # 找这段高点的宽度
        L = i
        while L > 0 and y[L] > floor + 2:
            L -= 1
        R = i
        while R < n - 1 and y[R] > floor + 2:
            R += 1
        lim = width + 1
        if edge_only_px and (i <= edge_only_px or i >= n - 1 - edge_only_px):
            lim = max(lim, 8)
        if 0 < (R - L) <= lim and max(y[max(0, L)], y[min(n - 1, R)]) <= floor + 3:
            # Lone sharp left-edge peaks (lipstick chlorite ~7.26°) must stay;
            # PDF/JCPDS card sticks come in a comb of several similar bars.
            if edge_only_px and (i <= edge_only_px or i >= n - 1 - edge_only_px):
                neigh = False
                for j in range(max(1, L - 50), min(n - 1, R + 50)):
                    if j <= L or j >= R:
                        if y[j] > floor + 8:
                            neigh = True
                            break
                if not neigh:
                    continue
            y[L + 1 : R] = floor
    return y


def write_debug_overlay(
    rgb: np.ndarray,
    frame: list[int],
    ticks: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    dest: Path,
    flags: list[str] | None = None,
) -> Path:
    im = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(im)
    x0, y0, x1, y1 = frame
    draw.rectangle([x0, y0, x1, y1], outline=(255, 140, 0), width=2)
    for t in ticks:
        x = float(t["x_px"])
        draw.line([(x, y1 - 8), (x, y1 + 12)], fill=(220, 30, 30), width=1)
        draw.text((x + 1, y1 + 12), f"{t['value']:g}", fill=(220, 30, 30))
    palette = [(200, 30, 30), (30, 80, 200), (20, 20, 20), (30, 140, 40), (140, 40, 170), (220, 70, 180)]
    for i, tr in enumerate(traces):
        color = tr.get("color") or palette[i % len(palette)]
        xs = tr.get("x_px")
        ys = tr.get("y_px")
        if xs is None or ys is None:
            continue
        pts = [(int(round(x)), int(round(y))) for x, y in zip(xs, ys)]
        pts = [(p[0], p[1]) for p in pts if 0 <= p[0] < im.width and 0 <= p[1] < im.height]
        if len(pts) >= 2:
            draw.line(pts, fill=tuple(int(c) for c in color), width=1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest)
    return dest



def _white_column_gaps(
    ink: np.ndarray,
    box: list[int],
    min_width: int = 18,
    max_ink: float = 0.08,
) -> list[tuple[int, int]]:
    """框内竖直空白带（并排子图之间的间隙）。"""
    x0, y0, x1, y1 = [int(v) for v in box]
    H, W = ink.shape
    xa, xb = max(0, x0), min(W, x1)
    ya, yb = min(y1 - 8, y0 + 8), max(y0 + 9, y1 - 8)
    if xb - xa < 80 or yb - ya < 20:
        return []
    col = ink[ya:yb, xa:xb].mean(axis=0)
    gaps: list[tuple[int, int]] = []
    n = len(col)
    i = 0
    while i < n:
        if col[i] <= max_ink:
            j = i + 1
            while j < n and col[j] <= max_ink:
                j += 1
            if j - i >= min_width and i > 25 and j < n - 25:
                ga, gb = xa + i, xa + j
                # 真间隙会打断底轴；图内无峰区仍有连续底轴，不能拆开
                ya_ax, yb_ax = max(0, y1 - 4), min(ink.shape[0], y1 + 3)
                if ya_ax < yb_ax and float(ink[ya_ax:yb_ax, ga:gb].mean()) < 0.25:
                    gaps.append((ga, gb))
            i = j
        else:
            i += 1
    return gaps


def _subboxes_from_gaps(box: list[int], gaps: list[tuple[int, int]]) -> list[list[int]]:
    if not gaps:
        return []
    x0, y0, x1, y1 = [int(v) for v in box]
    xs: list[int] = [x0]
    for a, b in sorted(gaps):
        xs.append(int(a))
        xs.append(int(b))
    xs.append(x1)
    out: list[list[int]] = []
    for i in range(0, len(xs) - 1, 2):
        a, b = int(xs[i]), int(xs[i + 1])
        if b - a >= 80:
            out.append([a, y0, b, y1])
    return out if len(out) >= 2 else []


def list_candidate_frames(
    rgb: np.ndarray,
    plot_box_1000: list[float] | None = None,
) -> list[dict[str, Any]]:
    """返回打过分的候选数据框，分数高的在前（含内嵌子框）。"""
    ink = ink_mask(rgb)
    gray = gray_of(rgb)
    H, W = ink.shape
    hlines, vlines = find_hv_lines(ink)
    cands = frames_from_spines(ink, hlines, vlines)
    # 把被空白带连在一起的并排子图拆开（宽 2θ 主图 + 放大图）
    extras: list[dict[str, Any]] = []
    for c in list(cands):
        gaps = _white_column_gaps(ink, c["box"])
        for sb in _subboxes_from_gaps(c["box"], gaps):
            extras.append({"box": sb, "width": sb[2] - sb[0], "height": sb[3] - sb[1], "vscore": c.get("vscore", 1) + 1})
        if gaps:
            c["has_gap"] = True
    cands.extend(extras)
    scored: list[dict[str, Any]] = []
    for c in cands:
        detailed = detect_tick_marks_detailed(gray, c["box"], axis_y=c["box"][3])
        marks = [float(m["x"]) for m in detailed]
        rec = score_frame(ink, c, marks, (H, W))
        rec["marks"] = marks
        rec["marks_detailed"] = detailed
        if c.get("has_gap"):
            rec["score"] = float(rec.get("score") or 0) - 8.0
        scored.append(rec)
    if plot_box_1000 and len(plot_box_1000) == 4:
        pb = [plot_box_1000[0] / 1000.0 * W, plot_box_1000[1] / 1000.0 * H,
              plot_box_1000[2] / 1000.0 * W, plot_box_1000[3] / 1000.0 * H]
        for c in scored:
            x0, y0, x1, y1 = c["box"]
            ow = _overlap(int(x0), int(x1), int(pb[0]), int(pb[2]))
            oh = _overlap(int(y0), int(y1), int(pb[1]), int(pb[3]))
            inter = ow * oh
            union = (x1 - x0) * (y1 - y0) + (pb[2] - pb[0]) * (pb[3] - pb[1]) - inter
            iou = inter / union if union > 0 else 0.0
            c["score"] += 1.5 * iou
    scored.sort(key=lambda c: -c["score"])
    uniq: list[dict[str, Any]] = []
    for c in scored:
        if any(
            abs(c["box"][0] - u["box"][0]) < 10
            and abs(c["box"][2] - u["box"][2]) < 10
            and abs(c["box"][1] - u["box"][1]) < 10
            and abs(c["box"][3] - u["box"][3]) < 10
            for u in uniq
        ):
            continue
        uniq.append(c)
    return uniq



def shrink_frame_to_ticks(
    frame: list[int],
    ticks: list[dict[str, Any]],
    *,
    ink: np.ndarray | None = None,
    inset_boxes: list[list[int]] | None = None,
) -> list[int]:
    """只在确认外侧是邻图/inset 或空白边时才收缩；不要把第一根标注刻度左侧的数据切掉。"""
    xs = sorted(float(t["x_px"]) for t in ticks if t.get("x_px") is not None)
    x0, y0, x1, y1 = [int(v) for v in frame]
    # 若最后一根刻度已经是轴右端标注，右边界不要再吃进隔壁放大图
    if ticks:
        last = max((t for t in ticks if t.get("x_px") is not None), key=lambda t: float(t["x_px"]))
        try:
            last_v = float(last.get("value"))
            last_x = float(last["x_px"])
            vals = sorted(float(t["value"]) for t in ticks if t.get("value") is not None)
            if len(vals) >= 2:
                vstep = float(np.median(np.diff(vals))) or 5.0
                xstep = float(np.median(np.diff(xs))) if len(xs) >= 2 else 20.0
                if last_v >= vals[-1] - 0.55 * vstep and x1 > last_x + 1.2 * xstep:
                    x1 = int(round(min(x1, last_x + 0.6 * xstep)))
        except (TypeError, ValueError):
            pass
    # 先切掉已知的不同 2θ 范围 inset
    if inset_boxes:
        for ix0, iy0, ix1, iy1 in inset_boxes:
            # 右侧 inset：把主框右边界收到 inset 左侧
            if ix0 > x0 + 0.4 * (x1 - x0) and ix1 >= x1 - 8:
                x1 = min(x1, max(int(ix0) - 2, x0 + 40))
            # 左侧 inset
            if ix1 < x1 - 0.4 * (x1 - x0) and ix0 <= x0 + 8:
                x0 = max(x0, min(int(ix1) + 2, x1 - 40))
    if len(xs) < 4:
        return [x0, y0, x1, y1]
    step = float(np.median(np.diff(xs)))
    if step < 6:
        return [x0, y0, x1, y1]
    width = max(1, x1 - x0)

    def _margin_is_empty(xa: int, xb: int) -> bool:
        if ink is None or xb <= xa:
            return False
        ya, yb = y0 + 4, max(y0 + 5, y1 - 8)
        sl = ink[ya:yb, max(0, xa):max(0, xb)]
        if sl.size == 0:
            return True
        return float(sl.mean()) < 0.03

    # 仅当第一根刻度左侧几乎没有墨迹（空白边），才略收；有峰则保留到脊
    if xs[0] - x0 > max(24, 0.12 * width) and _margin_is_empty(x0 + 2, int(xs[0] - 0.3 * step)):
        x0 = int(round(max(x0, xs[0] - step * 0.85)))
    if x1 - xs[-1] > max(24, 0.12 * width) and _margin_is_empty(int(xs[-1] + 0.3 * step), x1 - 2):
        x1 = int(round(min(x1, xs[-1] + 2)))
    if x1 - x0 < 40:
        return list(frame)
    return [x0, y0, x1, y1]



def classify_frames(
    candidates: list[dict[str, Any]],
    calib: dict[str, Any] | None = None,
    width: int | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """分出主 2θ 框、不同 2θ 范围的 inset/放大图、以及同尺度并排子图。"""
    if not candidates:
        return None, [], []
    calib = calib or {}
    # 估计整图宽度（VLM 刻度像素是相对整张裁图的）
    if width is None:
        width = 0
        for c in candidates:
            box = c.get("box") or [0, 0, 0, 0]
            width = max(width, int(box[2]) + 8)
        width = max(width, 100)

    scored: list[tuple] = []
    for c in candidates:
        box = list(c["box"])
        marks = list(c.get("marks") or [])
        detailed = list(c.get("marks_detailed") or [])
        tks, ft, ok, px = [], None, False, 0.0
        try:
            tks = assign_ticks(marks, calib, box, width, detailed=detailed or None)
            ft = ransac_linear(
                np.array([t["x_px"] for t in tks], dtype=float),
                np.array([t["value"] for t in tks], dtype=float),
            )
            px = float(ft.get("px_per_deg") or 0)
            ok = bool(ft.get("ok")) and float(ft.get("max_residual_deg") or 99) <= 0.6 and px > 1.0
        except Exception:
            pass
        c = dict(c)
        c["tick_assign"] = tks
        c["tick_fit"] = ft
        c["tick_ok"] = ok
        c["px_per_deg"] = px
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        nreg = len(longest_regular_marks(marks))
        # 带空白带的并排联合框不当主框
        gap_pen = 8.0 if c.get("has_gap") else 0.0
        snapped = any(bool(tk.get("snapped")) for tk in tks)
        overshoot = 0.0
        xr = _x_range_of(calib)
        if tks and xr is not None:
            hi = float(xr[1])
            ends = [float(tk["x_px"]) for tk in tks if abs(float(tk["value"]) - hi) <= 1.01]
            if ends:
                extra = float(box[2]) - max(ends)
                gaps = np.diff(sorted(float(tk["x_px"]) for tk in tks))
                step = float(np.median(gaps)) if len(gaps) else 20.0
                if extra > 1.6 * max(step, 8.0):
                    overshoot = 12.0
        snapped = any(bool(tk.get("snapped")) for tk in tks)
        coverage = (box[2] - box[0]) / max(1.0, float(width))
        resid = float((ft or {}).get("max_residual_deg") or 99.0) if ft else 99.0
        range_cov = 0.0
        spine_pen = 0.0
        if tks and ft and xr is not None:
            lo, hi = xr
            span = max(hi - lo, 1.0)
            a, bcoef = float(ft.get("a") or 0.0), float(ft.get("b") or 0.0)
            pred0 = a + bcoef * float(box[0])
            pred1 = a + bcoef * float(box[2])
            if pred1 < pred0:
                pred0, pred1 = pred1, pred0
            range_cov = max(0.0, min(1.0, (pred1 - pred0) / span))
            spine_pen = 12.0 * min(1.5, abs(pred0 - lo) / span + abs(pred1 - hi) / span)
        # 只覆盖标注范围一小段的框（主峰附近的窄条）不能当主框
        if range_cov < 0.7:
            ok = False
            c["tick_ok"] = False
        rank = (
            (20.0 if ok else 0.0)
            + (15.0 if snapped else 0.0)
            + min(nreg, 14) * 1.5
            + float(c.get("score") or 0) * 0.3
            + 5.0 * coverage
            + 16.0 * range_cov
            - (0.0 if resid > 10 else min(8.0, resid * 3.0))
            - gap_pen
            - overshoot
            - spine_pen
        )
        scored.append((rank, ok, c))
    scored.sort(key=lambda z: -z[0])
    main = scored[0][2]
    # 若最高分是联合框，但某个子框定标更好，改用子框
    if main.get("has_gap"):
        children = [s[2] for s in scored[1:] if s[1]]
        if children:
            main = max(children, key=lambda c: (len(c.get("marks") or []), c["box"][2] - c["box"][0]))

    mx0, my0, mx1, my1 = main["box"]
    main_px = float(main.get("px_per_deg") or 0) or 1.0
    insets: list[dict[str, Any]] = []
    siblings: list[dict[str, Any]] = []
    for _, ok, c in scored:
        if c is main or c["box"] == main["box"]:
            continue
        x0, y0, x1, y1 = c["box"]
        # 被主框几乎完全包含且只是半幅：不要当 inset（同轴右半幅）
        contained = (x0 >= mx0 - 8 and x1 <= mx1 + 8 and y0 >= my0 - 8 and y1 <= my1 + 8)
        area_c = max(1.0, (x1 - x0) * (y1 - y0))
        area_m = max(1.0, (mx1 - mx0) * (my1 - my0))
        ar = area_c / area_m
        px = float(c.get("px_per_deg") or 0)
        # 放大图：每度像素数明显更大（同一段 2θ 被拉宽），或根本无法用主图 x_range 定标
        is_zoom = False
        if px > 0 and main_px > 0 and px >= 2.2 * main_px:
            is_zoom = True
        if not c.get("tick_ok") and not contained:
            # 并排、定标失败，多半是另一套 2θ 范围
            if x0 >= mx1 - 8 or mx0 >= x1 - 8 or x0 >= mx0 + 0.55 * (mx1 - mx0):
                is_zoom = True
        ox = _overlap(mx0, mx1, x0, x1)
        oy = _overlap(my0, my1, y0, y1)
        inter = ox * oy
        # 与主框大部分重叠，或从主框内部才起笔再伸出：不是干净的并排放大图
        starts_inside = (mx0 + 24) < x0 < (mx1 - 24)
        overlaps_main = inter > 0.4 * area_c or inter > 0.5 * area_m or starts_inside
        if contained and ar < 0.55 and (is_zoom or not c.get("tick_ok")) and not overlaps_main:
            insets.append(c)
            continue
        if is_zoom and not contained and not overlaps_main:
            insets.append(c)
            continue
        # 同尺度并排：吸附成功、px/deg 接近、不太重叠、不是带间隙的联合框
        similar = (px <= 0 or main_px <= 0 or (0.6 * main_px <= px <= 1.6 * main_px))
        snapped = any(bool(tk.get("snapped")) for tk in (c.get("tick_assign") or []))
        if (
            similar
            and ar >= 0.35
            and inter < 0.5 * area_c
            and c.get("tick_ok")
            and snapped
            and not c.get("has_gap")
            and len(c.get("marks") or []) >= 3
        ):
            siblings.append(c)
    return main, insets, siblings


def mask_boxes_on_rgb(rgb: np.ndarray, boxes: list[list[int]], pad: int = 2) -> np.ndarray:
    out = rgb.copy()
    H, W = out.shape[:2]
    for box in boxes:
        x0, y0, x1, y1 = [int(v) for v in box]
        x0, x1 = max(0, min(x0, x1) - pad), min(W, max(x0, x1) + pad)
        y0, y1 = max(0, min(y0, y1) - pad), min(H, max(y0, y1) + pad)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = 255
    return out


def mask_vertical_sticks(rgb: np.ndarray, frame: list[int]) -> np.ndarray:
    """抹掉标准卡片那种成排的又高又窄竖线；不要把 1–3 条尖锐数据峰当成卡片。

    Also searches horizontal sub-bands (e.g. PDF# sticks under a stacked black curve).
    """
    x0, y0, x1, y1 = [int(v) for v in frame]
    ink = ink_mask(rgb)
    H, W = ink.shape
    x0i, x1i = max(0, x0 + 2), min(W, x1 - 1)
    y0i, y1i = max(0, y0 + 2), min(H, y1 - 2)
    if x1i - x0i < 20 or y1i - y0i < 20:
        return rgb

    def _comb(ya: int, yb: int, min_frac: float = 0.55) -> list[tuple[int, int, int, int]]:
        if yb - ya < 16:
            return []
        height = yb - ya
        col_h = ink[ya:yb, x0i:x1i].sum(axis=0)
        cands: list[tuple[int, int]] = []
        i = 0
        while i < len(col_h):
            ch = col_h[i]
            if ch < min_frac * height:
                i += 1
                continue
            left = col_h[i - 1] if i > 0 else 0
            w = 1
            j = i + 1
            while j < len(col_h) and col_h[j] >= min_frac * height:
                w += 1
                j += 1
            right = col_h[j] if j < len(col_h) else 0
            if w <= 3 and left < 0.22 * height and right < 0.22 * height:
                cands.append((i, w))
            i = j
        if len(cands) < 6:
            return []
        return [(x0i + i, x0i + i + w, ya, yb) for i, w in cands]

    boxes: list[tuple[int, int, int, int]] = []
    boxes.extend(_comb(y0i, y1i, 0.55))
    # sub-bands: lower third, lower half, and a sliding 20% window (PDF cards under LFP)
    height = y1i - y0i
    for ya, yb in (
        (y0i + int(0.55 * height), y1i),
        (y0i + int(0.62 * height), y1i - 2),
        (y0i + int(0.40 * height), y0i + int(0.75 * height)),
    ):
        boxes.extend(_comb(ya, yb, 0.50))
    if not boxes:
        return rgb
    out = rgb.copy()
    for xx0, xx1, ya, yb in boxes:
        out[ya:yb, xx0:xx1] = 255
    return out


def refine_ticks_from_axis_labels(
    rgb: np.ndarray,
    frame: list[int],
    ticks: list[dict[str, Any]],
    fit: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Re-fit 2θ ticks using printed axis-label positions under the frame.

    Used when short-line mark detection misses the 10/20/30° strokes and the
    VLM x_px sits on a peak (LDH Fig 9b experimental ~11° reading as ~9.9°).
    Only applied when ≥4 evenly spaced labels are found and the new residual
    is small. Title text such as "2θ, °" is ignored.
    """
    flags: list[str] = []
    if not ticks or len(ticks) < 3:
        return ticks, fit, flags
    try:
        from scipy import ndimage as nd
    except Exception:
        return ticks, fit, flags
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in frame]
    if y1 + 6 >= H:
        return ticks, fit, flags
    ink = ink_mask(rgb)
    y_lo = min(H - 1, y1 + 3)
    y_hi = min(H, y1 + 90)
    x_lo, x_hi = max(0, x0 - 8), min(W, x1 + 24)
    roi = np.zeros_like(ink)
    roi[y_lo:y_hi, x_lo:x_hi] = ink[y_lo:y_hi, x_lo:x_hi]
    labeled, n = nd.label(roi)
    raw: list[tuple[float, float, int, int, int]] = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if ys.size < 8:
            continue
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        if h < 6 or w < 3 or h > 28 or w > 36:
            continue
        raw.append((float(np.mean(xs)), float(np.mean(ys)), w, h, int(ys.size)))
    if len(raw) < 4:
        return ticks, fit, flags
    raw.sort()
    # drop the axis title (usually lower than the number row)
    ys = [r[1] for r in raw]
    y_med = float(np.median(ys))
    raw = [r for r in raw if abs(r[1] - y_med) <= 8]
    # merge split digits of "10" (two close blobs on the same row)
    merged: list[float] = []
    i = 0
    while i < len(raw):
        cx, _cy, w, h, area = raw[i]
        if i + 1 < len(raw) and (raw[i + 1][0] - cx) <= 14 and raw[i + 1][2] <= 14 and w <= 14:
            merged.append(0.5 * (cx + raw[i + 1][0]))
            i += 2
        else:
            merged.append(cx)
            i += 1
    if len(merged) < 4:
        return ticks, fit, flags
    # keep the longest almost-arithmetic progression (major 2θ labels)
    best: list[float] = []
    for start in range(len(merged)):
        for end in range(len(merged), start + 3, -1):
            xs = np.array(merged[start:end], dtype=float)
            if xs.size < 4:
                continue
            gaps = np.diff(xs)
            med = float(np.median(gaps))
            if med < 20:
                continue
            if float(np.max(np.abs(gaps - med))) > 0.18 * med:
                continue
            if xs.size > len(best):
                best = [float(v) for v in xs]
            break
        if len(best) >= 6:
            break
    if len(best) < 4:
        return ticks, fit, flags
    # match the longest run of consecutive calib tick values
    vals = sorted({float(t["value"]) for t in ticks if t.get("value") is not None})
    if len(vals) < 3:
        return ticks, fit, flags
    nlab = len(best)
    step = float(np.median(np.diff(vals))) if len(vals) >= 2 else 10.0
    if step <= 0:
        return ticks, fit, flags
    # Current ticks may omit a label that sits on the right spine (9b: 10–70
    # in-frame, printed 80 just at x1). Extend the same major-tick grid.
    grid = list(vals)
    while len(grid) < nlab:
        grid.append(grid[-1] + step)
    while len(grid) < nlab:
        grid.insert(0, grid[0] - step)
    chosen = None
    for i in range(0, len(grid) - nlab + 1):
        sl = grid[i : i + nlab]
        vg = np.diff(sl)
        if vg.size and float(np.max(np.abs(vg - vg[0]))) > 0.15 * max(abs(vg[0]), 1.0):
            continue
        chosen = sl
        break
    if chosen is None:
        chosen = grid[:nlab]
    xs = np.array(best, dtype=float)
    ys = np.array(chosen, dtype=float)
    new_fit = ransac_linear(xs, ys)
    if not new_fit.get("ok"):
        return ticks, fit, flags
    new_res = float(new_fit.get("max_residual_deg") or 99)
    old_res = float((fit or {}).get("max_residual_deg") or 99)
    # accept if labels are consistent; allow a slightly worse numeric residual
    # than a self-consistent but shifted VLM fit
    if new_res > 0.20:
        return ticks, fit, flags
    # The new ticks must span essentially the same labeled 2θ range. A short
    # subset (e.g. 10–50 on a 10–90 axis) would shrink MoS2 / ZIF / TiO2 / LFP.
    old_vals = sorted({float(t["value"]) for t in ticks if t.get("value") is not None})
    new_span = float(ys.max() - ys.min())
    old_span = float(old_vals[-1] - old_vals[0]) if len(old_vals) >= 2 else 0.0
    if old_span >= 20 and new_span < 0.85 * old_span:
        return ticks, fit, flags
    # Reject a shrunk range; allow one extra label just past the last in-frame tick
    # (printed 80° on the right spine when in-frame ticks only go to 70°).
    if float(ys.min()) > old_vals[0] + 5 or float(ys.max()) < old_vals[-1] - 5:
        return ticks, fit, flags
    # A self-consistent but shifted VLM fit can have residual ~0; still accept
    # labels when they cover the full range and are internally consistent.
    if old_res <= 0.05 and new_res > 0.18:
        return ticks, fit, flags
    new_ticks = []
    for xpx, val in zip(xs, ys):
        rec = {"value": float(val), "x_px": float(xpx), "source": "axis_label"}
        new_ticks.append(rec)
    flags.append(
        f"用轴下印刷数字重定标（残差 {old_res:.3f}→{new_res:.3f}°，{len(new_ticks)} 个标签）"
    )
    return new_ticks, new_fit, flags


def maybe_widen_frame_to_marks(
    rgb: np.ndarray,
    frame: list[int],
    ticks: list[dict[str, Any]],
    fit: dict[str, Any],
    calib: dict[str, Any],
    width: int,
    *,
    inset_boxes: list[list[int]] | None = None,
    ink: np.ndarray | None = None,
) -> tuple[list[int], list[dict[str, Any]], dict[str, Any], list[str]]:
    """If printed tick marks continue past the chosen right edge (and that region is
    not an inset), open the frame and re-assign ticks with the existing v11 rules.

    Reverts when residual worsens. Aimed at Co p4_01 cutting 100–105°.
    """
    flags: list[str] = []
    x0, y0, x1, y1 = [int(v) for v in frame]
    if width - x1 < 16:
        return list(frame), ticks, fit, flags
    # never open into a right-hand inset / zoom
    if inset_boxes:
        for ix0, iy0, ix1, iy1 in inset_boxes:
            if ix0 >= x1 - 8 and ix1 > x1 + 10:
                return list(frame), ticks, fit, flags
            # inset starts inside the would-be extension
            if ix0 > x0 + 0.4 * (x1 - x0) and _overlap(y0, y1, iy0, iy1) > 0.4 * (y1 - y0):
                # if the inset begins shortly after current x1, don't cross it
                if ix0 <= x1 + 30:
                    return list(frame), ticks, fit, flags
    gray = gray_of(rgb)
    # probe a wider box up to the crop (minus a small pad)
    wide = [x0, y0, min(width - 2, max(x1, width - 4)), y1]
    # if a vertical spine exists between x1 and the crop edge, stop there
    ink = ink if ink is not None else ink_mask(rgb)
    H, W = ink.shape
    # look for a tall vertical ink line (right spine) to the right of x1
    ya, yb = y0 + 4, max(y0 + 5, y1 - 4)
    right_spine = None
    if yb - ya > 20:
        col = ink[ya:yb, max(0, x1 - 2):min(W, wide[2] + 1)].mean(axis=0)
        # a spine is a thin high-ink column well after x1, with emptier columns around
        for i, v in enumerate(col):
            x = (x1 - 2) + i
            if x <= x1 + 4:
                continue
            if v >= 0.45:
                left = col[i - 2] if i >= 2 else v
                right = col[i + 2] if i + 2 < len(col) else v
                if left < 0.25 or x >= width - 12:
                    right_spine = x
                    break
    if right_spine is not None:
        wide[2] = min(wide[2], int(right_spine) + 2)
    if wide[2] <= x1 + 12:
        return list(frame), ticks, fit, flags
    # require some tick marks in the extension, or a continuing baseline
    marks_wide = detect_tick_marks(gray, wide, axis_y=wide[3])
    marks_wide = longest_regular_marks(marks_wide) or marks_wide
    xs = sorted(float(t["x_px"]) for t in ticks if t.get("x_px") is not None)
    extra = [m for m in marks_wide if xs and m > xs[-1] + 6 and m <= wide[2] + 2]
    if len(extra) < 1 and not (
        ink[ya:yb, x1:wide[2]].mean() > 0.01 and (wide[2] - x1) >= 20
    ):
        return list(frame), ticks, fit, flags
    # Do not jump a wide ink-free gutter (adjacent zoom / panel (b)).
    gutter = ink[ya:yb, x1:wide[2]]
    if gutter.size:
        col = gutter.mean(axis=0)
        i = 0
        while i < len(col):
            if col[i] <= 0.03:
                j = i + 1
                while j < len(col) and col[j] <= 0.03:
                    j += 1
                if (j - i) >= 12:
                    # stop before the gutter rather than crossing into the next panel
                    wide[2] = min(wide[2], x1 + i)
                    break
                i = j
            else:
                i += 1
    if wide[2] <= x1 + 12:
        return list(frame), ticks, fit, flags
    old_resid = float(fit.get("max_residual_deg") or 99)
    try:
        detailed = detect_tick_marks_detailed(gray, wide, axis_y=wide[3])
        new_ticks = assign_ticks(
            marks_wide, calib, wide, width,
            detailed=detailed or None,
        )
        new_fit = ransac_linear(
            np.array([t["x_px"] for t in new_ticks], dtype=float),
            np.array([t["value"] for t in new_ticks], dtype=float),
        )
    except Exception:
        return list(frame), ticks, fit, flags
    new_resid = float(new_fit.get("max_residual_deg") or 99)
    # keep v11 protection: do not adopt a worse residual (lipstick / Co shift)
    if new_resid > old_resid + 0.05:
        flags.append(f"右侧仍有刻度短线，扩框后残差 {new_resid:.3f}°，未改")
        return list(frame), ticks, fit, flags
    if new_resid > 0.15 and new_resid > old_resid + 0.01:
        return list(frame), ticks, fit, flags
    if not new_fit.get("ok") and new_resid > 0.5:
        return list(frame), ticks, fit, flags
    # also reject if the widened predicted end jumps several degrees past a consistent old fit
    # (unless residual stayed tiny — then the new values are the printed axis)
    flags.append(f"框右缘扩至 {wide[2]}px 以包含印刷刻度（残差 {old_resid:.3f}→{new_resid:.3f}°）")
    return wide, new_ticks, new_fit, flags


def ink_peak_xs(rgb: np.ndarray, frame: list[int], top_n: int = 10) -> list[float]:
    """数据框内按列突出度找主要峰的像素横坐标。"""
    x0, y0, x1, y1 = [int(v) for v in frame]
    ink = ink_mask(rgb)
    H, W = ink.shape
    x0i, x1i = max(0, x0 + 2), min(W, x1 - 1)
    y0i, y1i = max(0, y0 + 2), min(H, y1 - 6)
    if x1i - x0i < 10 or y1i - y0i < 10:
        return []
    # 每列从下往上的墨迹高度（近似峰高）
    heights = []
    for x in range(x0i, x1i):
        run = 0
        for y in range(y1i - 1, y0i, -1):
            if ink[y, x]:
                run += 1
            else:
                if run:
                    break
        heights.append(float(run))
    h = np.array(heights, dtype=float)
    if h.size < 8 or float(np.max(h)) < 6:
        return []
    # 局部基线
    k = max(9, h.size // 30)
    base = np.array([float(np.min(h[max(0, i - k):i + k + 1])) for i in range(len(h))])
    prom = h - base
    pmax = float(np.max(prom)) or 1.0
    xs: list[float] = []
    for i in range(2, len(prom) - 2):
        if prom[i] >= 0.25 * pmax and prom[i] >= prom[i - 1] and prom[i] >= prom[i + 1]:
            if xs and abs((x0i + i) - xs[-1]) < 6:
                if prom[i] > prom[int(xs[-1] - x0i)]:
                    xs[-1] = float(x0i + i)
            else:
                xs.append(float(x0i + i))
    xs = sorted(xs, key=lambda x: -prom[int(x - x0i)])[:top_n]
    return sorted(xs)


def verify_and_correct_fit(
    rgb: np.ndarray,
    frame: list[int],
    ticks: list[dict[str, Any]],
    fit: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """用原图墨迹峰与刻度短线交叉校验；系统偏差约 0.5° 以上则重定标。"""
    flags: list[str] = []
    ticks = [dict(t) for t in ticks]
    a, bcoef = float(fit["a"]), float(fit["b"])
    peak_xs = ink_peak_xs(rgb, frame)
    if not peak_xs or len(ticks) < 2:
        return ticks, fit, flags

    # 若主峰几乎压在某根已标注刻度上，描点 2θ 应等于该刻度读数
    deltas: list[float] = []
    constraints: list[tuple[float, float]] = []
    for px in peak_xs:
        nearest = min(ticks, key=lambda t: abs(float(t["x_px"]) - px))
        dist = abs(float(nearest["x_px"]) - px)
        pred = a + bcoef * px
        if dist <= 5:
            constraints.append((px, float(nearest["value"])))
            deltas.append(pred - float(nearest["value"]))
    already_snapped = all(bool(t.get("snapped")) for t in ticks) and float(fit.get("max_residual_deg") or 99) <= 0.25
    if deltas:
        med = float(np.median(deltas))
        # 已钉在印刷短线上且残差很小的定标，不要因峰顶偏几个像素而改（用户要纠的是约 2° 的整体平移）
        if abs(med) >= (1.2 if already_snapped else 0.8):
            flags.append(f"主峰相对邻近刻度中位偏差 {med:+.2f}°，已按刻度短线重定标")
            xs = [float(t["x_px"]) for t in ticks] + [c[0] for c in constraints]
            ys = [float(t["value"]) for t in ticks] + [c[1] for c in constraints]
            new_fit = ransac_linear(np.array(xs, dtype=float), np.array(ys, dtype=float))
            if new_fit.get("ok") or float(new_fit.get("max_residual_deg") or 99) <= 1.0:
                fit = new_fit
                a, bcoef = float(fit["a"]), float(fit["b"])
        elif abs(med) >= 0.45:
            flags.append(f"主峰相对邻近刻度中位偏差 {med:+.2f}°（已钉短线，未改）")
    # 若模型刻度整体平移：检测短线与 tick.x_px 的中位差
    gray = gray_of(rgb)
    marks = detect_tick_marks(gray, frame, axis_y=frame[3])
    marks = longest_regular_marks(marks)
    if len(marks) >= max(3, len(ticks) - 1) and ticks:
        # 每个 tick 找最近短线
        offsets = []
        for t in ticks:
            mx = min(marks, key=lambda m: abs(m - float(t["x_px"])))
            if abs(mx - float(t["x_px"])) <= 25:
                offsets.append(mx - float(t["x_px"]))
        if len(offsets) >= 3:
            off = float(np.median(offsets))
            old_resid = float(fit.get("max_residual_deg") or 99.0)
            # 已有低残差定标时不要再整体改钉：误检短线（数字笔画、峰谷）会把 7°/80° 好轴拉到几十度残差。
            if abs(off) >= 4 and not (already_snapped and old_resid <= 0.3) and old_resid > 0.25:
                snapped = []
                used = set()
                for t in ticks:
                    cand = [m for m in marks if m not in used]
                    if not cand:
                        continue
                    mx = min(cand, key=lambda m: abs(m - (float(t["x_px"]) + off)))
                    used.add(mx)
                    snapped.append({"value": float(t["value"]), "x_px": float(mx), "snapped": True})
                if len(snapped) >= 2:
                    new_fit = ransac_linear(
                        np.array([t["x_px"] for t in snapped], dtype=float),
                        np.array([t["value"] for t in snapped], dtype=float),
                    )
                    new_resid = float(new_fit.get("max_residual_deg") or 99.0)
                    # 只有吸附后残差仍然好、或明显改善时才采用
                    improved = new_resid <= 0.5 or (new_resid + 0.2 < old_resid and new_resid <= 1.2)
                    if improved and new_fit.get("ok"):
                        flags.append(f"刻度像素相对短线整体平移 {off:+.1f}px，已吸附短线")
                        ticks = snapped
                        fit = new_fit
                    else:
                        flags.append(
                            f"刻度像素相对短线偏移 {off:+.1f}px，吸附后残差 {new_resid:.2f}°，未改"
                        )
    return ticks, fit, flags


def apply_fit_to_traces(traces: list[dict[str, Any]], fit: dict[str, Any]) -> None:
    """定标更新后，用同一套 x_px 重算 2θ。"""
    a, bcoef = float(fit["a"]), float(fit["b"])
    for tr in traces:
        xs_px = np.asarray(tr.get("x_px"), dtype=float)
        ys = np.asarray(tr.get("y"), dtype=float)
        ypx = tr.get("y_px")
        if xs_px.size < 8:
            continue
        two_theta = a + bcoef * xs_px
        xg, yg = resample_xy(two_theta, ys, GRID_DEG)
        tr["x"], tr["y"] = xg, yg
        tr["x_px"] = np.interp(xg, two_theta, xs_px) if len(two_theta) else xs_px
        if ypx is not None:
            tr["y_px"] = np.interp(xg, two_theta, np.asarray(ypx, dtype=float))
        tr["peaks"] = find_trace_peaks(tr["x"], tr["y"])


def clip_traces_to_frame(
    traces: list[dict[str, Any]],
    frame: list[int],
    ticks: list[dict[str, Any]],
    fit: dict[str, Any],
) -> None:
    """保留坐标框两端之间的 2θ（含第一根标注刻度左侧），不外推到邻图。"""
    a, bcoef = float(fit["a"]), float(fit["b"])
    x0, _, x1, _ = frame
    lo = a + bcoef * (x0 + 1)
    hi = a + bcoef * (x1 - 1)
    if hi < lo:
        lo, hi = hi, lo
    tvals = [float(t["value"]) for t in ticks if t.get("value") is not None]
    if tvals:
        # 允许比首尾标注刻度略宽，但不短于刻度范围
        lo = min(lo, min(tvals) - 0.05)
        hi = max(hi, max(tvals) + 0.05)
    for tr in traces:
        x = np.asarray(tr.get("x"))
        y = np.asarray(tr.get("y"))
        if x.size < 8:
            continue
        keep = (x >= lo) & (x <= hi)
        if int(keep.sum()) < 8:
            continue
        tr["x"], tr["y"] = x[keep], y[keep]
        for key in ("y_px", "x_px"):
            if key in tr:
                arr = np.asarray(tr[key])
                if arr.shape[0] == keep.shape[0]:
                    tr[key] = arr[keep]
        tr["peaks"] = find_trace_peaks(tr["x"], tr["y"])


def traces_quality_flags(traces: list[dict[str, Any]]) -> list[str]:
    flags = []
    for tr in traces:
        y = np.asarray(tr.get("y"), dtype=float)
        if y.size == 0 or float(np.nanmax(y) if y.size else 0) <= 1e-9:
            flags.append(f"曲线「{tr.get('name')}」强度全为零")
        elif y.size >= 8 and float(np.nanmax(y) - np.nanmin(y)) < 1e-6:
            flags.append(f"曲线「{tr.get('name')}」塌缩为一条水平线")
    return flags


def drop_or_keep_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """去掉全零迹线；若全部为零则原样返回以便上层重试。"""
    kept = []
    for tr in traces:
        y = np.asarray(tr.get("y"), dtype=float)
        if y.size and float(np.nanmax(y)) > 1e-9:
            kept.append(tr)
    return kept if kept else traces


def digitize_in_frame(
    crop_path: Path,
    calib: dict[str, Any],
    run_dir: Path | None = None,
    *,
    mask: bool = True,
) -> dict[str, Any]:
    """对一张裁图做框内描点。精确 2θ 来自刻度短线线性定标，不依赖模型读峰。"""
    crop_path = Path(crop_path)
    rgb = load_rgb(crop_path)
    H, W = rgb.shape[:2]
    plot_box = calib.get("plot_box_1000") if isinstance(calib.get("plot_box_1000"), (list, tuple)) else None
    plot_hint = list(plot_box) if plot_box and len(plot_box) == 4 else None
    candidates = list_candidate_frames(rgb, plot_box_1000=plot_hint)
    if not candidates:
        raise ValueError("未检测到可用的 XRD 坐标框")
    main_cand, inset_cands, sibling_cands = classify_frames(candidates, calib, width=W)
    inset_boxes = [list(c["box"]) for c in inset_cands]
    if inset_boxes:
        rgb_work = mask_boxes_on_rgb(rgb, inset_boxes, pad=2)
    else:
        rgb_work = rgb
    gray = gray_of(rgb_work)
    chosen = None
    ticks = None
    fit = None
    last_err = None
    ranked = []
    # 主框优先；inset 不参与主定标
    inset_ids = {id(c) for c in inset_cands}
    pool = [c for c in candidates if id(c) not in inset_ids]
    if main_cand is not None and main_cand not in pool:
        pool = [main_cand] + pool
    for cand in pool:
        frame = list(cand["box"])
        marks = list(cand.get("marks") or [])
        if len(marks) < 3:
            marks = detect_tick_marks(gray, frame, axis_y=frame[3])
        try:
            tks = assign_ticks(marks, calib, frame, W, detailed=cand.get("marks_detailed"))
            ft = ransac_linear(
                np.array([t["x_px"] for t in tks], dtype=float),
                np.array([t["value"] for t in tks], dtype=float),
            )
        except Exception as exc:
            last_err = exc
            continue
        peak_frac = float((cand.get("stats") or {}).get("peak_col_frac") or 0)
        long_frac = float((cand.get("stats") or {}).get("long_col_frac") or 0)
        nreg = len(longest_regular_marks(marks))
        resid = float(ft.get("max_residual_deg") or 99)
        fw = max(1.0, float(cand["box"][2] - cand["box"][0]))
        xr = _x_range_of(calib)
        range_cov = 0.0
        spine_pen = 0.0
        if tks and xr is not None:
            lo, hi = xr
            span = max(hi - lo, 1.0)
            a, bcoef = float(ft.get("a") or 0.0), float(ft.get("b") or 0.0)
            pred0 = a + bcoef * float(frame[0])
            pred1 = a + bcoef * float(frame[2])
            if pred1 < pred0:
                pred0, pred1 = pred1, pred0
            range_cov = max(0.0, min(1.0, (pred1 - pred0) / span))
            spine_pen = 12.0 * min(1.5, abs(pred0 - lo) / span + abs(pred1 - hi) / span)
        ok_res = bool(ft.get("ok")) and resid <= 0.5 and (range_cov >= 0.65 or xr is None)
        score = (
            (20.0 if ok_res else 0.0)
            + peak_frac * 8
            - long_frac * 10
            + min(nreg, 14) * 0.8
            + float(cand.get("score") or 0) * 0.2
            + 5.0 * (fw / max(1.0, float(W)))
            + 16.0 * range_cov
            - min(8.0, resid * 3.0)
            - spine_pen
        )
        if cand is main_cand:
            score += 3.0
        ranked.append((score, ok_res, cand, tks, ft))
    ranked.sort(key=lambda z: -z[0])
    if ranked:
        good = [z for z in ranked if z[1]]
        if good:
            def _pick_key(z):
                box = z[2]["box"]
                resid = float(z[4].get("max_residual_deg") or 99)
                tks = z[3] or []
                n_out = sum(
                    1
                    for tk in tks
                    if tk.get("x_px") is not None
                    and (float(tk["x_px"]) < box[0] - 8 or float(tk["x_px"]) > box[2] + 8)
                )
                # 不要只因为框更宽（把右上放大图框进来）就赢
                return (z[0] - 6.0 * n_out, -resid, box[2] - box[0])
            pick = max(good, key=_pick_key)
        else:
            pick = ranked[0]
        chosen, ticks, fit = pick[2], pick[3], pick[4]
    if chosen is None or ticks is None or fit is None:
        raise last_err if last_err else ValueError("未检测到可用的 XRD 坐标框")
    ink_src = ink_mask(rgb_work)
    frame = shrink_frame_to_ticks(list(chosen["box"]), ticks, ink=ink_src, inset_boxes=inset_boxes)
    frame, ticks, fit, widen_flags = maybe_widen_frame_to_marks(
        rgb_work, frame, ticks, fit, calib, W,
        inset_boxes=inset_boxes, ink=ink_src,
    )
    ticks, fit, label_flags = refine_ticks_from_axis_labels(rgb_work, frame, ticks, fit)
    marks = list(chosen.get("marks") or ticks)
    ticks, fit, align_flags = verify_and_correct_fit(rgb_work, frame, ticks, fit)
    align_flags = list(label_flags) + list(align_flags or [])
    work = rgb_work
    if mask:
        work = apply_pixel_masks(work, calib.get("mask_boxes_1000"), frame)
        masked = work.copy()
        x0, y0, x1, y1 = frame
        if y0 > 0:
            masked[:y0, :] = 255
        if y1 < H - 1:
            masked[y1 + 1 :, :] = 255
        if x0 > 0:
            masked[:, :x0] = 255
        if x1 < W - 1:
            masked[:, x1 + 1 :] = 255
        work = masked
    work = mask_grid_lines(work, frame)
    color_src = work.copy()
    work = mask_text_blobs(work, frame)
    work = mask_phase_markers(work, frame)
    work = mask_vertical_sticks(work, frame)

    def _trace_frame(img, fr, ft, name_suffix: str = "") -> list[dict[str, Any]]:
        trs = trace_bands(img, fr, calib, ft, rgb_color=color_src)
        if name_suffix:
            for tr in trs:
                tr["name"] = f"{tr['name']}{name_suffix}"
        return trs

    traces = _trace_frame(work, frame, fit)
    # 同尺度并排子图（如两个 Rietveld 面板）分别描，避免一条框跨两套曲线描成全零
    for i, sib in enumerate(sibling_cands[:2]):
        sbox = list(sib["box"])
        sbox = shrink_frame_to_ticks(sbox, ticks, ink=ink_src, inset_boxes=inset_boxes)
        # 只处理与主框不严重重叠的兄弟框
        ox = _overlap(frame[0], frame[2], sbox[0], sbox[2])
        oy = _overlap(frame[1], frame[3], sbox[1], sbox[3])
        inter = ox * oy
        if inter > 0.55 * (sbox[2] - sbox[0]) * (sbox[3] - sbox[1]):
            continue
        try:
            smarks = list(sib.get("marks") or [])
            if len(smarks) < 3:
                smarks = detect_tick_marks(gray, sbox, axis_y=sbox[3])
            stks = assign_ticks(smarks, calib, sbox, W, detailed=sib.get("marks_detailed"))
            sfit = ransac_linear(
                np.array([z["x_px"] for z in stks], dtype=float),
                np.array([z["value"] for z in stks], dtype=float),
            )
            extra = _trace_frame(work, sbox, sfit, name_suffix=f"-p{i + 2}")
            traces.extend(extra)
        except Exception:
            continue
    if not traces:
        raise ValueError("坐标框内没有描出曲线")
    # 全零/塌缩：去掉卡片竖线后再描一次
    qflags = traces_quality_flags(traces)
    if qflags:
        retry_img = mask_vertical_sticks(work, frame)
        retried = _trace_frame(retry_img, frame, fit)
        if True:
            # 若重试产生了更多非零曲线，则替换零线
            by_name = {tr["name"]: tr for tr in traces}
            for tr in retried:
                y = np.asarray(tr.get("y"), dtype=float)
                old = by_name.get(tr["name"])
                old_y = np.asarray((old or {}).get("y"), dtype=float) if old else np.array([])
                old_zero = (old_y.size == 0 or float(np.nanmax(old_y) if old_y.size else 0) <= 1e-9)
                new_ok = y.size and float(np.nanmax(y)) > 1e-9
                if new_ok and (old is None or old_zero):
                    by_name[tr["name"]] = tr
            traces = list(by_name.values())
        work = retry_img
    traces = drop_or_keep_traces(traces)
    clip_traces_to_frame(traces, frame, ticks, fit)
    apply_fit_to_traces(traces, fit)
    clip_traces_to_frame(traces, frame, ticks, fit)
    all_peaks = []
    for tr in traces:
        all_peaks.extend(tr.get("peaks") or [])
    check = crosscheck_peaks(all_peaks, _hint_peaks(calib))
    flags = list(align_flags) + list(widen_flags)
    if inset_boxes:
        flags.append(f"已屏蔽 {len(inset_boxes)} 个不同 2θ 范围的放大图/inset")
    flags.extend(traces_quality_flags(traces))
    if check.get("shift_suggestion") is not None:
        flags.append(f"提示峰相对描点中位偏差 {check['shift_suggestion']:+.2f}°（以描点为准，未改定标）")
    if check.get("mismatches"):
        flags.append(f"{len(check['mismatches'])} 个提示峰与描点相差 > {PEAK_MISMATCH_DEG}°")

    result: dict[str, Any] = {
        "status": "ok",
        "method": "frame_trace",
        "frame": frame,
        "ticks": ticks,
        "fit": fit,
        "traces": traces,
        "n_traces": len(traces),
        "trace_names": [t["name"] for t in traces],
        "peak_check": check,
        "flags": flags,
        "frame_score": chosen.get("score"),
        "n_tick_marks": len(marks),
    }
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(work).save(run_dir / "masked.png")
        write_debug_overlay(rgb, frame, ticks, traces, run_dir / "validate.png", flags)
        (run_dir / "trace_meta.json").write_text(
            __import__("json").dumps(
                {
                    "frame": frame,
                    "ticks": ticks,
                    "fit": {k: v for k, v in fit.items() if k != "residuals"} | {"residuals": fit.get("residuals")},
                    "flags": flags,
                    "peak_check": check,
                    "n_traces": len(traces),
                    "trace_names": [t["name"] for t in traces],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        result["validate"] = str(run_dir / "validate.png")
        result["run_dir"] = str(run_dir)
    return result
