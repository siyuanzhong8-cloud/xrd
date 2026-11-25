# Vendored from XRD-Workbench-Dev for the pdf_xrd pipeline. Logic unchanged.
"""Separate supported stacked color traces before running the skill digitizer."""
import sys
import csv
import json

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, find_objects, label, maximum_filter1d, percentile_filter
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks


def apply_vision_plan(source, output, skill, plan, anchors):
    """Validate visual suggestions against the source pixels and preserve calibration coordinates."""
    if str(skill) not in sys.path:
        sys.path.insert(0, str(skill))
    from xrd_auto import dx

    with Image.open(source) as image:
        rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    ink = rgb.mean(axis=2) < 155
    left, right, top, bottom = dx.auto_frame(ink)
    framed = framed_panel_geometry(rgb)
    if framed:
        left, right, dividers = framed
        top, bottom = int(dividers[0]), int(dividers[-1])
    actions = []

    def box(value):
        if not isinstance(value, list) or len(value) != 4:
            return None
        if any(type(v) not in (int, float) or not 0 <= v <= 1000 for v in value):
            return None
        a, b, c, d = [round(v * size / 1000) for v, size in zip(value, (width, height, width, height))]
        return (a, b, c, d) if a < c and b < d else None

    panel = box(plan.get("panel_box"))
    if panel and right - left >= 60 and bottom - top >= 35:
        a, b, c, d = panel
        margin = max(12, round(height * 0.035))
        if a < min(left, min(anchors)) and c > max(right, max(anchors)) and b <= top and d >= bottom + margin:
            outside = np.ones(ink.shape, dtype=bool)
            outside[b:d, a:c] = False
            if (ink & outside).any():
                rgb[outside] = 255
                actions.append("保留主图和横轴刻度")

    masks = plan.get("masks")
    if isinstance(masks, list) and masks:
        # Label actual ink rather than blanking the entire model-proposed rectangle.
        colored_ink = (rgb.mean(axis=2) < 205) | (np.ptp(rgb, axis=2) >= 42)
        components, _ = label(colored_ink, structure=np.ones((3, 3)))
        locations = find_objects(components)
        removed = set()
        for candidate in masks[:6]:
            rect = box(candidate)
            if not rect:
                continue
            a, b, c, d = rect
            if not (left + 2 <= a < c <= right - 2 and top + 2 <= b < d <= bottom - 2):
                continue
            if c - a > 0.23 * (right - left) or d - b > 0.15 * (bottom - top):
                continue
            ids = []
            for index in np.unique(components[b:d, a:c]):
                if not index or index in removed:
                    continue
                ys, xs = locations[index - 1]
                if (a <= xs.start < xs.stop <= c and b <= ys.start < ys.stop <= d
                    and xs.stop - xs.start <= max(5, width * 0.04)
                    and 4 <= ys.stop - ys.start <= height * 0.08):
                    ids.append(index)
            if len(ids) >= 2:
                removed.update(ids)
        if removed:
            rgb[np.isin(components, list(removed))] = 255
            actions.append(f"屏蔽 {len(removed)} 个独立标注笔画")

    style = None
    proposed = plan.get("style")
    if proposed in ("line", "fill") and right - left >= 60 and bottom - top >= 35:
        ink = rgb.mean(axis=2) < 155
        lengths = [end - start + 1 for column in range(left + 3, right - 2)
                   for start, end in dx.col_strokes(ink, column, top + 3, bottom - 3)]
        if lengths:
            filled = np.median(lengths) >= 6 or dx.filled_at_data_bottom(ink, bottom - 3, left + 3, right - 3)
            if proposed == ("fill" if filled else "line"):
                style = proposed
    if actions:
        Image.fromarray(rgb).save(output)
    return actions, style


def clean_connected_trace(source, output, skill):
    """Remove disconnected labels when a single stroke spans the plot."""
    if str(skill) not in sys.path:
        sys.path.insert(0, str(skill))
    from xrd_auto import dx

    with Image.open(source) as image:
        gray = np.asarray(image.convert("L"))
        rgb = np.asarray(image.convert("RGB"))
    ink = gray < 155
    left, right, top, bottom = dx.auto_frame(ink)
    data = np.s_[top + 3:bottom - 3, left + 3:right - 2]
    segments, _ = label(ink[data], structure=np.ones((3, 3)))
    counts = np.bincount(segments.ravel())
    if len(counts) < 2:
        return False
    stroke = segments == (np.argmax(counts[1:]) + 1)
    if np.median(np.ptp(rgb[data], axis=2)[stroke]) > 18:
        return False
    if stroke.any(axis=0).mean() < 0.98:
        return False
    if np.median(stroke.sum(axis=0)) > 5 or np.count_nonzero(ink[data] & ~stroke) < 40:
        return False
    remaining = segments.copy()
    remaining[stroke] = 0
    for component in np.flatnonzero(np.bincount(remaining.ravel())[1:] >= 0.3 * stroke.shape[1]) + 1:
        if (remaining == component).any(axis=0).mean() >= 0.3:
            return False
    clean = gray.copy()
    clean[data] = np.where(stroke, gray[data], 255)
    Image.fromarray(clean).save(output)
    return True


def extract_overlapping_colors(source, run, skill, slope, intercept, wavelength, grid_step, prominence):
    """Trace distinct colored ink independently when curves cross vertical bands."""
    if str(skill) not in sys.path:
        sys.path.insert(0, str(skill))
    from xrd_auto import dx

    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"))
    left, right, top, bottom = dx.auto_frame(rgb.mean(axis=2) < 155)
    crop = rgb[top + 3:bottom - 3, left + 3:right - 2]
    height, width = crop.shape[:2]
    if width < 100 or height < 80:
        return False
    chromatic = crop.max(axis=2).astype(float) - crop.min(axis=2) >= 45
    if not chromatic.any():
        return False
    palette, counts = np.unique(crop[chromatic], axis=0, return_counts=True)
    deficit = 255.0 - crop
    norm = np.linalg.norm(deficit, axis=2)
    curves = []
    bases = []
    for color in palette[np.argsort(counts)[-64:][::-1]]:
        direction = 255.0 - color
        direction /= np.linalg.norm(direction)
        ink = (deficit @ direction) > 48
        ink &= np.linalg.norm(np.cross(deficit, direction), axis=2) < np.maximum(12, 0.07 * norm)
        base = int(np.argmax(maximum_filter1d(ink, size=13, axis=0).sum(axis=1)))
        if ink[max(0, base - 6):base + 7].any(axis=0).sum() < 0.52 * width:
            continue
        if any(abs(base - known) < 12 for known in bases):
            continue
        bases.append(base)
        ink[base + 14:] = False
        connected, _ = label(ink, structure=np.ones((3, 3)))
        ids = np.unique(connected[max(0, base - 8):base + 13])
        sizes = np.bincount(connected.ravel())
        ink &= np.isin(connected, ids[(ids != 0) & (sizes[ids] >= 0.06 * width)])
        good = np.flatnonzero(ink.any(axis=0))
        if len(good) < 0.92 * width or np.diff(good).max(initial=0) > 12:
            continue
        edge = np.interp(np.arange(width), good, ink[:, good].argmax(axis=0))
        background = percentile_filter(edge, percentile=75, size=101)
        curve = np.maximum(0, background - edge)
        curves.append((base, color, curve, background, ink, 0))
    if len(curves) < 2 or len(curves) != len(bases):
        return False
    if any(np.max(item[2]) <= 8 for item in curves):
        return False
    curves.sort(key=lambda item: item[0])
    if not any(np.max(curve) > base - previous[0]
               for previous, (base, _, curve, _, _, _) in zip(curves, curves[1:])):
        return False
    write_color_traces(run, rgb, left, top, width, curves, slope, intercept,
                       wavelength, grid_step, prominence)
    return True


def framed_panel_geometry(rgb):
    dark = rgb.max(axis=2) < 105
    lines = []
    for y in np.flatnonzero(dark.sum(axis=1) >= 0.6 * dark.shape[1]):
        edges = np.flatnonzero(np.diff(np.r_[False, dark[y], False]))
        starts, stops = edges[::2], edges[1::2]
        i = int(np.argmax(stops - starts))
        if 0.6 * dark.shape[1] <= stops[i] - starts[i] <= 0.97 * dark.shape[1]:
            lines.append((int(y), starts[i], stops[i]))
    if not lines:
        return None
    aligned = max(
        ([line for line in lines if abs(line[1] - start) <= 3 and abs(line[2] - stop) <= 3]
         for _, start, stop in lines), key=len,
    )
    rows = np.array([line[0] for line in aligned])
    dividers = rows[np.r_[True, np.diff(rows) > 3]]
    if not 3 <= len(dividers) <= 16 or np.min(np.diff(dividers)) < 28:
        return None
    left = int(np.median([line[1] for line in aligned])) - 1
    right = int(np.median([line[2] for line in aligned])) - 3
    return left, right, dividers


def extract_framed_panels(source, run, slope, intercept, wavelength, grid_step, prominence):
    """Separate vertically aligned panels before tracing their experimental curves."""
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"))
    framed = framed_panel_geometry(rgb)
    if framed is None:
        return False
    left, right, dividers = framed
    top = int(dividers[0])
    width = right - left - 3
    curves = []
    for start, stop in zip(dividers, dividers[1:]):
        panel = rgb[start + 3:stop - 2, left + 3:right]
        p = panel.astype(float)
        masks = (np.ptp(p, axis=2) >= 45, p.max(axis=2) < 145)
        for chromatic, mask in zip((True, False), masks):
            joined = mask | binary_closing(mask, structure=np.ones((1, 9)))
            components, _ = label(joined, structure=np.ones((3, 3)))
            counts = np.bincount(components.ravel())
            if len(counts) < 2:
                continue
            count_ids = np.argsort(counts[1:])[-8:] + 1
            stroke = max((components == k for k in count_ids),
                         key=lambda line: line.any(axis=0).sum())
            stroke &= mask
            good = np.flatnonzero(stroke.any(axis=0))
            if len(good) < 0.92 * width or np.diff(good).max(initial=0) > 8:
                continue
            if chromatic:
                edge = np.interp(np.arange(width), good, stroke[:, good].argmax(axis=0))
                baseline = percentile_filter(edge, percentile=75, size=101)
            else:
                # Use ink-weighted subpixel centers for faint monochrome curves.
                # A fixed offset preserves broad features and the measured background.
                neighbors = maximum_filter1d(stroke, size=3, axis=0)
                weights = (255 - p.mean(axis=2)) * (stroke | (neighbors & ~mask))
                rows = np.arange(len(panel))[:, None]
                centers = (rows * weights[:, good]).sum(axis=0) / weights[:, good].sum(axis=0)
                edge = np.interp(np.arange(width), good, centers)
                baseline = float(edge.max())
            color = np.median(panel[stroke], axis=0)
            curves.append((start, color, np.maximum(0, baseline - edge),
                           start - top + baseline, stroke, start - top))
            break
        else:
            return False
    write_color_traces(run, rgb, left, top, width, curves, slope, intercept,
                       wavelength, grid_step, prominence)
    return True


def trace_quality(stroke, projected, angles, peaks, peak_prominences):
    """Compare the overlay with source ink and weak peaks with an alternate center/edge."""
    rows = np.arange(stroke.shape[0])[:, None]
    near = (stroke & (np.abs(rows - projected) <= 1.5)).any(axis=0)
    edges = np.flatnonzero(np.diff(np.r_[False, ~near, False]))
    regions = [(a, b - 1, "pixel_gap") for a, b in edges.reshape(-1, 2) if b - a >= 3]

    good = np.flatnonzero(stroke.any(axis=0))
    if not len(good):
        return regions, set(peaks)
    top = stroke[:, good].argmax(axis=0)
    center = (rows * stroke[:, good]).sum(axis=0) / stroke[:, good].sum(axis=0)
    alternate = center if np.median(np.abs(projected[good] - top)) < 0.35 else top
    alternate = -np.interp(np.arange(len(angles)), good, alternate)
    alternate_peaks, _ = find_peaks(alternate, prominence=1.25)
    tolerance = max(float(np.median(np.diff(angles))), 0.15)
    unstable = set()
    for peak, prominence in zip(peaks, peak_prominences):
        if prominence <= 6 and (not near[peak] or not np.any(
            np.abs(angles[alternate_peaks] - angles[peak]) <= tolerance
        )):
            unstable.add(peak)
            regions.append((max(0, peak - 1), min(len(angles) - 1, peak + 1), "weak_peak"))
    return regions, unstable


def write_color_traces(run, rgb, left, top, width, curves, slope, intercept,
                       wavelength, grid_step, prominence):
    from matplotlib.figure import Figure

    folder = run / "traces_xy"
    folder.mkdir()
    x = left + 3 + np.arange(width)
    angles = intercept + slope * x
    overlay = Figure(figsize=(10, 2.1 * len(curves)), layout="constrained")
    axes = overlay.subplots(len(curves), sharex=True)
    validation = Figure(figsize=(12, 7), layout="constrained")
    image_axis = validation.subplots()
    image_axis.imshow(rgb)
    image_axis.set_axis_off()
    reviews = []
    with (run / "peaks.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trace", "2theta", "d_A", "Irel", "phase", "conf"])
        for index, (_, color, curve, background, stroke, row_offset) in enumerate(curves):
            np.savetxt(folder / f"trace{index}.xy", np.column_stack([angles, curve]), fmt="%.6f")
            plot_x, plot_y = angles, curve
            if grid_step:
                plot_x = np.arange(np.ceil(angles[0] / grid_step) * grid_step, angles[-1], grid_step)
                plot_y = PchipInterpolator(angles, curve)(plot_x)
                np.savetxt(folder / f"trace{index}_grid.xy", np.column_stack([plot_x, plot_y]), fmt="%.6f")
            peaks, properties = find_peaks(curve, prominence=max(prominence, 0.025 * curve.max()),
                                           distance=max(3, round(0.4 / slope)))
            regions, unstable = trace_quality(stroke, background - curve - row_offset,
                                              angles, peaks, properties["prominences"])
            for first, last, reason in regions:
                reviews.append({"trace": index, "start": round(float(angles[first]), 2),
                                "end": round(float(angles[last]), 2), "reason": reason})
                axes[index].axvspan(angles[first], angles[last], color="#e8a545", alpha=0.23)
            for peak in peaks:
                angle = angles[peak]
                spacing = wavelength / (2 * np.sin(np.radians(angle / 2))) if wavelength else ""
                writer.writerow([index, round(angle, 4), round(spacing, 4) if spacing else "",
                                 round(curve[peak] / curve.max(), 4), "",
                                 "review" if peak in unstable else "ok"])
            axes[index].plot(plot_x, plot_y, color=color / 255, lw=0.85)
            axes[index].set_ylabel(f"#{index}")
            image_axis.plot(x, top + 3 + background - curve, color="black", lw=0.5, alpha=0.6)
    axes[-1].set_xlabel("2theta (deg)")
    overlay.savefig(run / "overlay.png", dpi=135)
    validation.savefig(run / "validate.png", dpi=135)
    detail = "".join(f"Trace {r['trace']}: {r['start']:.2f}–{r['end']:.2f}° {r['reason']}\n"
                     for r in reviews)
    (run / "qc.txt").write_text(
        f"Pixel support and weak peak stability checked on {len(curves)} independent traces.\n" + detail,
        encoding="utf-8",
    )
    (run / "pixel-qc.json").write_text(json.dumps(reviews), encoding="utf-8")


def prepare_color_image(source, output, skill):
    if str(skill) not in sys.path:
        sys.path.insert(0, str(skill))
    from xrd_auto import baseline_of, colour_groups, dx

    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    ink = rgb.mean(axis=2) < 155
    left, right, top, bottom = dx.auto_frame(ink)
    width = right - left
    if width < 100 or bottom - top < 80:
        return False
    inside = np.zeros(ink.shape, dtype=bool)
    inside[top + 4:bottom - 4, left + 4:right - 3] = True
    groups = []
    for group in colour_groups(rgb, inside):
        mask = group["mask"]
        base = baseline_of(mask, top + 4, bottom - 4)
        if base is None or mask[base - 2:base + 3].any(axis=0).sum() < 0.45 * width:
            continue
        if all(abs(base - existing) > 16 for existing, _ in groups):
            groups.append((base, mask))
    if len(groups) < 2:
        return False

    # Include an achromatic trace only if it has a distinct, long baseline.
    dark = inside & (rgb.max(axis=2) - rgb.min(axis=2) < 24) & ink
    base = baseline_of(dark, top + 4, bottom - 4)
    if base is not None and dark[base - 2:base + 3].any(axis=0).sum() >= 0.65 * width:
        if all(abs(base - existing) > 16 for existing, _ in groups):
            groups.append((base, dark))

    clean = np.zeros_like(ink)
    upper = top + 4
    for base, mask in sorted(groups, key=lambda item: item[0]):
        end = min(bottom - 3, base + 5)
        band = binary_closing(mask[upper:end], structure=np.ones((1, 5))) | mask[upper:end]
        labels, _ = label(band, structure=np.ones((3, 3)))
        connected = np.unique(labels[max(0, base - upper - 4):])
        keep = band & np.isin(labels, connected[connected != 0])
        if keep.any(axis=0).sum() < 0.80 * width or (upper != top + 4 and keep[0].any()):
            return None
        clean[upper:end] |= keep
        upper = end

    clean[top:top + 2, left:right + 1] = True
    clean[bottom - 1:bottom + 1, left:right + 1] = True
    clean[top:bottom + 1, left:left + 2] = True
    clean[top:bottom + 1, right - 1:right + 1] = True
    Image.fromarray(np.where(clean, 0, 255).astype(np.uint8)).save(output)
    return True
