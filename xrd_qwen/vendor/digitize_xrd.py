#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Vendored from XRD-Workbench-Dev for the pdf_xrd pipeline. Logic unchanged.
"""
digitize_xrd.py - digitize stacked XRD patterns from a cropped figure image.

Workflow skill: pdf-xrd-digitize (step 2-5).

Usage:
  python digitize_xrd.py --img fig.png \
      --anchors "210:20,322:30,430:40,540:50,652:60,761:70,869:80" \
      [--lamb 1.5406] [--outdir out]

  --anchors : comma list of x_px:2theta for the x-axis tick labels (use DIGIT
              CENTERS, not left edges; get them via OCR, see ocr.ps1 and
              references/example_tio2.md).

Outputs:
  peaks.csv    per-trace peak table (2theta, d, Irel, phase, confidence)
  qc.txt       acceptance audit: calibration residuals, coverage, unexplained
               strong peaks, per-phase systematic offset, drift
  overlay.png  per-trace panels with peak labels + reference lines
  validate.png digitized curves redrawn onto the original image geometry

2026-09-17 (ground-truth bench driven fixes):
  * auto_frame requires a LONG CONTINUOUS dark run (a spine line) instead of a
    mere ink extent -- rotated y-axis titles used to be mistaken for the left
    frame, so the real spine was traced as a phantom trace.
  * columns that are full-height lines (spines) are skipped while tracing.
  * detect_baselines falls back to a rolling high-percentile envelope curve when
    the flat-background histogram finds nothing (monotone sloping background);
    levels then become offsets above the curve and base_row(i,x) is used for both
    lane assignment and intensity.  Flat backgrounds keep the old path bit-for-bit.
  * peak acceptance rule is selectable (--peak-crit hmax|sigma|cwt).  The bench
    (7 variants incl. a 1.5 px-noise pattern) shows the legacy hmax rule is the
    most balanced default; sigma recovers a few weak peaks on clean data but adds
    weak false peaks on noisy data; CWT needs widths matched to the peak FWHM.
  * constants are split by category: the x-axis (2theta) ones -- peak separation
    and CWT widths -- follow the figure's sampling (px/deg, ref 7.04), while the
    noise/intensity ones (boxcar window, prominence floor, sigma window, baseline
    clustering) stay in pixels.  The boxcar is now 3 px instead of 5 px: 5 px at
    7.04 px/deg smooths 0.71 deg and used to merge weak 0.8-deg doublets.

2026-09-21:
  * new '--peak-crit deconv': a CONSERVATIVE local multi-pseudo-Voigt
    least-squares splitter (Rietveld-like).  It only ever ADDS resolution to the
    ordinary find_peaks table (never loses a peak) and is SNR-gated.
"""
import argparse, os, csv, re
import numpy as np
from PIL import Image
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

DEFAULT_REFS = {
    "A(101)": 25.28, "A(103)": 36.95, "A(004)": 37.80, "A(112)": 38.58,
    "A(200)": 48.05, "A(105)": 53.89, "A(211)": 55.06, "A(213)": 62.12,
    "A(204)": 62.69, "A(116)": 68.76, "A(220)": 70.31, "A(215)": 75.03,
    "R(110)": 27.45, "R(101)": 36.09, "R(200)": 39.19, "R(111)": 41.23,
    "R(210)": 44.05, "R(211)": 54.32, "R(220)": 56.64, "R(002)": 62.74,
    "R(310)": 64.04, "R(301)": 65.78, "R(112)": 69.01,
}

def parse_refs(spec):
    """--refs: 'name:pos,...' overrides defaults; 'none'/'-'/empty disables."""
    if spec is None:
        return dict(DEFAULT_REFS)
    if not spec.strip() or spec.strip().lower() in ("none", "-", "off"):
        return {}
    out = {}
    for tok in spec.split(","):
        name, pos = tok.rsplit(":", 1)
        out[name.strip()] = float(pos)
    return out


def load_ink(path, thresh=155):
    im = Image.open(path).convert("L")
    gray = np.asarray(im, dtype=np.float64)
    return gray < thresh, gray, gray.shape[1], gray.shape[0]


def subpixel_top(gray, x, ytop, thresh):
    """Sub-pixel top edge of the ink: where the vertical gray profile crosses
    `thresh` between row ytop-1 (light) and ytop (dark), linear-interpolated."""
    if ytop <= 0 or ytop >= gray.shape[0]:
        return float(ytop)
    g0 = float(gray[ytop - 1, x])
    g1 = float(gray[ytop, x])
    if g0 > g1 and g0 >= thresh:
        f = (thresh - g0) / (g1 - g0)
        f = min(max(f, 0.0), 1.0)
        return (ytop - 1) + f
    return float(ytop)


def long_row_runs(ink):
    """rows with a >=55% width contiguous dark run"""
    out = []
    H, W = ink.shape
    for y in range(H):
        mx = cur = 0
        for v in ink[y]:
            cur = cur + 1 if v else 0
            mx = max(mx, cur)
        if mx > 0.55 * W:
            out.append((y, mx))
    return out


def longest_run(col):
    """longest contiguous dark run in a boolean column"""
    mx = cur = 0
    for v in col:
        cur = cur + 1 if v else 0
        mx = max(mx, cur)
    return mx


def auto_frame(ink):
    """estimate plot box rows/cols.

    Columns qualify only if they carry a LONG CONTINUOUS dark run (a frame /
    spine line)."""
    H, W = ink.shape
    rows = long_row_runs(ink)
    top = min((y for y, _ in rows), default=int(0.08 * H))
    bot = max((y for y, _ in rows), default=int(0.9 * H))
    h = max(1, bot - top)
    cols = [x for x in range(W) if longest_run(ink[top:bot, x]) > 0.6 * h]
    if not cols:                                   # fallback: old extent test
        for x in range(W):
            nz = np.nonzero(ink[top:bot, x])[0]
            if len(nz) and nz[-1] - nz[0] > 0.7 * h:
                cols.append(x)
    left = cols[0] if cols else int(0.05 * W)
    right = cols[-1] if cols else int(0.95 * W)
    return left, right, top, bot


def col_strokes(ink, x, y0, y1):
    """contiguous dark y-runs in column x within [y0,y1)"""
    col = ink[y0:y1, x]
    nz = np.nonzero(col)[0]
    out = []
    if len(nz):
        s = p = int(nz[0])
        for v in nz[1:]:
            v = int(v)
            if v == p + 1:
                p = v
            else:
                out.append((y0 + s, y0 + p))
                s = p = v
        out.append((y0 + s, y0 + p))
    return out


def envelope_curve(ink, y0, y1, x0, x1, win=None):
    """Robust baseline CURVE (rolling high percentile of lowest ink row)."""
    env = np.full(x1 - x0 + 1, np.nan)
    for i, x in enumerate(range(x0, x1 + 1)):
        st = col_strokes(ink, x, y0, y1)
        if st:
            env[i] = max(b for _, b in st)
    if np.all(np.isnan(env)):
        return None
    idx = np.arange(len(env))
    good = ~np.isnan(env)
    env = np.interp(idx, idx[good], env[good])
    if win is None:
        win = max(15, len(env) // 8)
    half = win // 2
    out = np.empty_like(env)
    for i in range(len(env)):
        lo, hi = max(0, i - half), min(len(env), i + half + 1)
        out[i] = np.percentile(env[lo:hi], 90)
    return out


def cluster_1d(vals, hmin, span):
    """histogram + find_peaks clustering of 1-D values -> sorted levels"""
    hist = np.zeros(int(span) + 2, dtype=int)
    for v in vals:
        j = int(v - hmin)
        if 0 <= j < len(hist):
            hist[j] += 1
    hs = np.convolve(hist, np.ones(5) / 5, mode="same")
    pk, _ = find_peaks(hs, height=max(20, 0.05 * len(vals)), distance=20)
    out = []
    for v in sorted(hmin + float(kk) for kk in pk):
        if out and v - out[-1] < 12:
            out[-1] = (out[-1] + v) / 2.0
        else:
            out.append(v)
    return out


def filled_at_data_bottom(ink, y1, x0, x1):
    return y1 >= 3 and np.mean(np.all(ink[y1 - 3:y1, x0:x1 + 1], axis=0)) >= 0.8


def detect_baselines(ink, y0, y1, x0, x1):
    """Returns (levels, envelope).  Flat background -> absolute rows; monotone
    sloping background -> offsets above the rolling envelope curve."""
    hist = np.zeros(y1 - y0 + 1, dtype=int)
    for x in range(x0, x1 + 1):
        for (_, b) in col_strokes(ink, x, y0, y1):
            if y0 <= b <= y1:
                hist[b - y0] += 1
    hs = np.convolve(hist, np.ones(5) / 5, mode="same")
    pk, _ = find_peaks(hs, height=max(20, 0.05 * (x1 - x0)), distance=20)
    bl = sorted(y0 + int(v) for v in pk)
    out = []
    for b in bl:
        if out and b - out[-1] < 12:
            out[-1] = (out[-1] + b) // 2
        else:
            out.append(b)
    # A filled trace touching the plot floor ends at the clipping boundary.
    # find_peaks does not detect a histogram maximum at that boundary.
    if filled_at_data_bottom(ink, y1, x0, x1):
        out = [b for b in out if b <= y1 - 13] + [y1 - 1]
    if out:
        return out, None
    # --- sloping background fallback -------------------------------------
    env = envelope_curve(ink, y0, y1, x0, x1)
    if env is None:
        return [], None
    det = []
    for i, x in enumerate(range(x0, x1 + 1)):
        for (_, b) in col_strokes(ink, x, y0, y1):
            det.append(b - env[i])
    if not det:
        return [], env
    det = np.asarray(det, dtype=float)
    lo, hi = float(det.min()), float(det.max())
    out = cluster_1d(det, lo, hi - lo)
    if not out:
        out = [float(np.median(det))]
    return out, env


REF_PX_PER_DEG = 7.04          # reference sampling of the calibration case


def px_scale(px_per_deg):
    """Scale factor for the X-AXIS (2theta) constants."""
    if not px_per_deg or px_per_deg <= 0:
        return 1.0
    return float(min(max(px_per_deg / REF_PX_PER_DEG, 0.25), 4.0))


def noise_sigma(y, win=7):
    """Robust noise scale: 1.4826*MAD of the residual against a short SG fit."""
    if len(y) < 9:
        return 0.0
    w = int(win)
    if w % 2 == 0:
        w += 1
    w = max(5, min(w, (len(y) - 1) | 1))
    res = y - savgol_filter(y, w, 2)
    med = np.median(res)
    return float(1.4826 * np.median(np.abs(res - med)))


def find_peak_idx(y, hmax, mode="hmax", k=3.0, floor=2.5, dist=3, widths=None):
    """Peak search with a selectable acceptance rule (bench-calibrated)."""
    if mode == "hmax":
        pk, _ = find_peaks(y, prominence=max(floor, 0.025 * hmax), distance=dist)
        return np.asarray(pk)
    if mode == "cwt":
        try:
            from scipy.signal import find_peaks_cwt
            w = np.arange(*widths) if widths else np.arange(1, 4)
            pk = find_peaks_cwt(y, widths=w, min_snr=k, noise_perc=10)
            return np.asarray(pk)
        except Exception as exc:                      # pragma: no cover
            print("   (cwt unavailable: %s -> falling back to sigma)" % exc)
            mode = "sigma"
    sg = noise_sigma(y)
    pk, _ = find_peaks(y, prominence=max(floor, k * sg), distance=dist)
    return np.asarray(pk)


def _pvoigt(x, c, a, f, eta=0.5):
    """Unit pseudo-Voigt (eta=0.5) -- the profile used by --peak-crit deconv."""
    s = f / 2.0
    lor = 1.0 / (1.0 + ((x - c) / s) ** 2)
    gau = np.exp(-np.log(2.0) * ((x - c) / s) ** 2)
    return a * ((1 - eta) * lor + eta * gau)


def _peak_halfwidth(y, i):
    """FWHM (in samples) around peak index i, from the half-max crossings."""
    base = float(np.median(y))
    half = base + 0.5 * (float(y[i]) - base)
    l = i
    while l > 0 and y[l] > half:
        l -= 1
    r = i
    while r < len(y) - 1 and y[r] > half:
        r += 1
    return max(2.0, float(r - l))


def _deconv_segment(seg, fwhm_px, max_k=3, amp_frac=0.10):
    """Least-squares fit of linear background + K pseudo-Voigt components
    (shared FWHM); K chosen by BIC.  Returns {'best_k','comps','bic1','bic'} or None."""
    n = len(seg)
    if n < 9:
        return None
    x = np.arange(n, dtype=float)
    xmid = 0.5 * (n - 1)
    h = float(seg.max() - np.median(seg))
    if h <= 0:
        return None
    pk, _ = find_peaks(seg, distance=max(1, int(round(0.6 * fwhm_px))))
    if len(pk) == 0:
        pk = np.array([int(np.argmax(seg))])
    cand = [float(v) for v in pk[np.argsort(seg[pk])[::-1]][:max(2, max_k)]]
    while len(cand) < 2:
        cand.append(cand[0] + 2.0)

    def model(th):
        yy = th[0] + th[1] * (x - xmid)
        for c, aa in zip(th[3::2], th[4::2]):
            yy = yy + _pvoigt(x, c, aa, th[2])
        return yy

    fits = []
    for K in range(1, max_k + 1):
        cs = cand[:K]
        th0 = [float(np.median(seg)), 0.0, float(fwhm_px)] + sum([[c, h] for c in cs], [])
        lo = [-1e4, -1e4, 0.5] + sum([[0.0, 0.0] for _ in range(K)], [])
        hi = [1e6, 1e6, 40.0] + sum([[float(n), 1e7] for _ in range(K)], [])
        try:
            r = least_squares(lambda th: model(th) - seg, th0,
                              bounds=(lo, hi), max_nfev=1200, x_scale="jac")
        except Exception:
            continue
        rss = float(np.sum(r.fun ** 2))
        bic = n * np.log(max(rss, 1e-12) / n) + len(r.x) * np.log(n)
        comps = sorted(zip(r.x[3::2], r.x[4::2]), key=lambda t: t[0])
        fits.append((K, bic, comps))
    if not fits:
        return None
    bic1 = next((b for K, b, _ in fits if K == 1), None)
    K, bic, comps = min(fits, key=lambda z: z[1])
    amax = max(a for _, a in comps) if comps else 0.0
    signif = sorted(float(c) for c, a in comps if a > amp_frac * amax)
    return {"best_k": K, "comps": signif, "bic1": bic1, "bic": bic}


def deconv_peaks_profile(y, fwhm_px, prom, max_k=3, dbic=10.0):
    """Local model-fitting peak finder (--peak-crit deconv).  CONSERVATIVE: it
    only ADDS resolution to the ordinary find_peaks table (never loses a peak)."""
    pk, _ = find_peaks(y, prominence=prom, distance=1)
    if len(pk) == 0:
        return np.asarray([], dtype=int)
    pk = [int(v) for v in pk]
    seen = [False] * len(pk)
    out = []
    for a in range(len(pk)):
        if seen[a]:
            continue
        group = [pk[a]]
        seen[a] = True
        for b in range(a + 1, len(pk)):
            if not seen[b] and pk[b] - group[-1] <= fwhm_px:
                group.append(pk[b])
                seen[b] = True
        fw = max(_peak_halfwidth(y, p) for p in group)
        pad = int(round(2.0 * fw))
        lo = max(0, group[0] - pad)
        hi = min(len(y), group[-1] + pad + 1)
        seg = np.asarray(y[lo:hi], dtype=float)
        r = _deconv_segment(seg, fw, max_k)
        if r and r["best_k"] >= 2 and r["bic1"] is not None and \
                r["bic1"] - r["bic"] > dbic and len(r["comps"]) >= 2:
            cs = [lo + c for c in r["comps"]]
            sep = cs[-1] - cs[0]
            mid = 0.5 * (cs[0] + cs[-1])
            seed_mid = 0.5 * (group[0] + group[-1])
            if (0.5 * fw <= sep <= 2.0 * fw) and abs(mid - seed_mid) <= 1.0 * fw:
                out += [int(round(c)) for c in cs]
                continue
        out += group
    return np.asarray(sorted(set(min(max(int(v), 0), len(y) - 1) for v in out)), dtype=int)


def detect_ticks(ink, axis_row, x0, x1, L=10, frac=0.85):
    """x centers of short vertical tick segments adjacent to the bottom axis."""
    sets = []
    for y0, y1 in ((axis_row - L, axis_row - 1), (axis_row + 1, axis_row + L)):
        cols = [x for x in range(x0, x1 + 1)
                if y0 >= 0 and ink[y0:y1, x].mean() >= frac]
        groups = []
        for x in cols:
            if groups and x - groups[-1][-1] <= 2:
                groups[-1].append(x)
            else:
                groups.append([x])
        sets.append([(sum(gg) / len(gg), gg[0], gg[-1]) for gg in groups
                     if len(gg) <= 8])
    return sets[0] if len(sets[0]) >= len(sets[1]) else sets[1]


def parabola_refine(xs, ys, i):
    """sub-pixel peak from 3 points"""
    if i <= 0 or i >= len(xs) - 1:
        return xs[i]
    x0, x1, x2 = xs[i - 1], xs[i], xs[i + 1]
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
    den = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(den) < 1e-9:
        return x1
    A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
    B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
    if A >= 0:
        return x1
    return -B / (2 * A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--anchors", required=True,
                    help="x_px:2theta pairs, comma separated (digit CENTERS)")
    ap.add_argument("--lamb", type=float, default=1.5406)
    ap.add_argument("--data-xmax", type=int, default=None,
                    help="rightmost data column (exclude right-side labels); default auto")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--thresh", type=int, default=155)
    ap.add_argument("--style", choices=["auto", "fill", "line"], default="auto",
                    help="fill=area-filled traces; line=thin curve traces")
    ap.add_argument("--refs", default=None,
                    help="override reference lines 'name:pos,...'; empty string disables")
    ap.add_argument("--mask", default=None,
                    help="exclude ink regions 'x0,y0,x1,y1;...' (image px)")
    ap.add_argument("--smooth", type=int, default=0,
                    help="extra Savitzky-Golay window (odd px) on the intensity profile")
    ap.add_argument("--boxcar", type=int, default=3,
                    help="boxcar smoothing window in px (default 3; 1 = off)")
    ap.add_argument("--peak-dist-deg", type=float, default=None,
                    help="minimum peak separation in DEGREES for find_peaks")
    ap.add_argument("--resample", type=float, default=0.0,
                    help="resample each trace onto a uniform 2theta grid of this step (deg)")
    ap.add_argument("--ticks", default=None,
                    help="comma list of x_px tick-line centers (one per --anchors entry)")
    ap.add_argument("--auto-ticks", action="store_true",
                    help="auto-detect tick-line centers near the bottom frame for calibration")
    ap.add_argument("--peak-crit", choices=["hmax", "sigma", "cwt", "fine", "deconv"], default="hmax",
                    help="peak acceptance rule: hmax (default) | sigma | cwt | fine | deconv "
                         "(deconv = windowed least-squares pseudo-Voigt decomposition, "
                         "splits overlapping doublets; noise/SNR-gated)")
    ap.add_argument("--sigma-k", type=float, default=3.0,
                    help="k for --peak-crit sigma|cwt (default 3)")
    ap.add_argument("--prom-floor", type=float, default=2.5,
                    help="absolute prominence floor in px (default 2.5)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ink, gray, W, H = load_ink(args.img, args.thresh)
    xl, xr, yt, yb = auto_frame(ink)
    y_data_bottom = yb - 3
    xl_i, xr_i = xl + 3, xr - 3
    if args.data_xmax:
        xr_i = min(xr_i, args.data_xmax)

    pairs = [p.split(":") for p in args.anchors.split(",")]
    if args.ticks and args.ticks.strip():
        tx = [float(v) for v in args.ticks.split(",")]
        if len(tx) != len(pairs):
            raise SystemExit(f"--ticks has {len(tx)} values but --anchors has {len(pairs)}")
        pairs = [[str(xx), tt] for xx, (_, tt) in zip(tx, pairs)]
        print(f"calibration: using {len(tx)} explicit tick centers")
    elif args.auto_ticks:
        tv = [float(t) for _, t in pairs]
        ticks = [c for c, _, _ in detect_ticks(ink, yb, xl, xr)]
        best = None
        if len(ticks) >= len(tv):
            yv = np.array(tv)
            for s in range(len(ticks) - len(tv) + 1):
                win = np.array(ticks[s:s + len(tv)])
                Aw = np.column_stack([win, np.ones_like(win)])
                (bb, aa), *_ = np.linalg.lstsq(Aw, yv, rcond=None)
                r = float(np.max(np.abs(Aw @ np.array([bb, aa]) - yv)))
                if best is None or r < best[0]:
                    best = (r, win)
        if best is not None:
            print("auto-ticks: x=[" + ",".join(f"{v:.1f}" for v in best[1]) +
                  f"] (best-window max resid {best[0]:.3f} deg)")
            pairs = [[str(xx), tt] for xx, tt in zip(best[1].tolist(), tv)]
        else:
            print(f"auto-ticks: found {len(ticks)} ticks (< {len(tv)} anchors); "
                  "falling back to --anchors x")
    A = np.array([[float(x), 1.0] for x, _ in pairs], dtype=float)
    Y = np.array([float(t) for _, t in pairs], dtype=float)
    (b, a), *_ = np.linalg.lstsq(A, Y, rcond=None)
    print(f"frame x[{xl}-{xr}] y[{yt}-{yb}]  cal: 2th = {a:.3f} + {b:.5f}*x")

    # --- constants: only the x-axis (2theta) ones follow the sampling (px/deg) ---
    ppd = (1.0 / abs(b)) if b else 0.0
    S = px_scale(ppd)
    win_sm = max(1, int(args.boxcar))
    if args.peak_dist_deg is not None:
        dist_pk = max(1, int(round(args.peak_dist_deg / abs(b)))) if b else 3
    else:
        dist_pk = max(1, int(round(3 * S)))
    fine = (args.peak_crit == "fine")
    dist_pk_default = max(1, int(round(3 * S)))
    if fine:
        win_sm = 1
        dist_pk = max(1, int(round(1.0 * S)))
    if args.peak_crit == "deconv":
        win_sm = 1
    floor_px = args.prom_floor
    w_lo = max(1, int(round(0.15 * ppd)))
    w_hi = max(w_lo + 1, int(round(0.45 * ppd)))
    print("sampling %.2f px/deg (x-scale %.2f): smooth %d px, peak-dist %d px (%.3f deg), "
          "prom floor %.2f px, cwt widths %d-%d%s"
          % (ppd, S, win_sm, dist_pk, dist_pk * abs(b), floor_px, w_lo, w_hi,
             "  [FINE mode]" if fine else ""))

    # --- calibration acceptance: anchor residuals (AutoXRD-style check) ---
    pred = b * A[:, 0] + a
    resid = pred - Y
    max_res = float(np.max(np.abs(resid))) if len(resid) else 0.0
    for (xpx, tth), r in zip(pairs, resid):
        print(f"   anchor x={float(xpx):7.1f}  2th={float(tth):6.2f}  resid={r:+.3f}")
    if len(resid) and max_res > 0.3:
        print(f"   !! calibration residual up to {max_res:.2f} deg (>0.3): "
              "re-check OCR digit CENTERS / tick values before trusting phase IDs")
    elif len(resid):
        print(f"   calibration residuals OK (max |resid|={max_res:.2f} deg)")

    masks = []
    if args.mask and args.mask.strip():
        for tok in args.mask.split(";"):
            x0m, y0m, x1m, y1m = (float(v) for v in tok.split(","))
            masks.append((x0m, y0m, x1m, y1m))
        print(f"masks: {masks}")

    def masked(x, tp):
        return any(x0m <= x <= x1m and y0m <= tp <= y1m for (x0m, y0m, x1m, y1m) in masks)

    baselines, env = detect_baselines(ink, yt, y_data_bottom, xl_i, xr_i)
    if env is not None:
        print("baselines: offsets %.0f..%.0f above a sloping baseline curve (n=%d)"
              % (min(baselines), max(baselines), len(baselines)))
    else:
        print(f"baselines: {baselines} (n={len(baselines)})")
    if not baselines:
        raise SystemExit("no trace baselines found")

    def base_row(i, x):
        """baseline row of trace i at column x (envelope curve + offset)"""
        return (env[x - xl_i] if env is not None else 0.0) + baselines[i]

    # full-height columns (frame spines) are never data
    spine_cols = set(x for x in range(xl_i, xr_i + 1)
                     if longest_run(ink[yt:y_data_bottom, x]) > 0.85 * (y_data_bottom - yt))
    if spine_cols:
        print("spine columns skipped: %d" % len(spine_cols))

    style = args.style
    if style == "auto":
        lens = []
        for x in range(xl_i, xr_i + 1):
            for (tp, bt) in col_strokes(ink, x, yt, y_data_bottom):
                lens.append(bt - tp + 1)
        med = float(np.median(lens)) if lens else 0
        style = "fill" if med >= 6 or filled_at_data_bottom(ink, y_data_bottom, xl_i, xr_i) else "line"
        print(f"auto style: median stroke len={med:.1f}px -> {style}")

    def assign(x, bottom, top):
        """fill: stroke bottom near a baseline.  line: stroke TOP in a lane."""
        if style == "fill":
            bi, bd = None, 1e9
            for i in range(len(baselines)):
                d = bottom - base_row(i, x)
                if -8 <= d <= 45 and abs(d) < bd:
                    bd, bi = abs(d), i
            return bi
        if top < yt + 4 or top > base_row(len(baselines) - 1, x) + 3:
            return None
        for i in range(len(baselines)):
            lo = (base_row(i - 1, x) if i > 0 else yt + 3)
            if lo < top <= base_row(i, x) + 3:
                return i
        return None

    traces = [{"baseline": bl, "apex": {}} for bl in baselines]
    for x in range(xl_i, xr_i + 1):
        if x in spine_cols:
            continue
        for (tp, bt) in col_strokes(ink, x, yt, y_data_bottom):
            if masked(x, tp):
                continue
            i = assign(x, bt, tp)
            if i is None:
                continue
            sub = subpixel_top(gray, x, tp, args.thresh)
            if x not in traces[i]["apex"] or sub < traces[i]["apex"][x]:
                traces[i]["apex"][x] = sub

    refs = parse_refs(args.refs)
    refvals = sorted(set(refs.values()))
    refnames = {}
    for v in refvals:
        refnames[v] = [k for k, u in refs.items() if abs(u - v) < 1e-9]
    refarr = np.array(refvals)

    rows_out = []
    all_tr = []
    for i, t in enumerate(traces):
        bl = t["baseline"]
        xs = np.arange(xl_i, xr_i + 1)
        I = np.zeros_like(xs, dtype=float)
        for x, tp in t["apex"].items():
            I[x - xl_i] = max(0.0, base_row(i, x) - tp)
        sm = np.convolve(I, np.ones(win_sm) / win_sm, mode="same")
        if args.smooth and args.smooth >= 5:
            w = args.smooth if args.smooth % 2 == 1 else args.smooth + 1
            if 5 <= w <= len(sm):
                sm = savgol_filter(sm, w, 2)
        hmax = sm.max()
        lowconf = hmax < 15
        sg = noise_sigma(sm)
        crit = args.peak_crit
        dd = dist_pk
        if crit == "fine":
            if hmax > 0 and 3.0 * sg > 0.008 * hmax:
                sm = np.convolve(I, np.ones(3) / 3, mode="same")
                if args.smooth and args.smooth >= 5:
                    w = args.smooth if args.smooth % 2 == 1 else args.smooth + 1
                    if 5 <= w <= len(sm):
                        sm = savgol_filter(sm, w, 2)
                hmax = sm.max(); lowconf = hmax < 15; sg = noise_sigma(sm)
                crit = "hmax"; dd = dist_pk_default
                print("   [fine] SNR gate: 3*sigma=%.1fpx > 0.8%%*hmax -> "
                      "fall back to default (hmax) rule" % (3 * sg))
            else:
                crit = "sigma"
        if crit == "deconv" and hmax > 0 and 3.0 * sg > 0.008 * hmax:
            crit = "hmax"; dd = dist_pk_default
            print("   [deconv] SNR gate: 3*sigma=%.1fpx > 0.8%%*hmax -> "
                  "fall back to default (hmax) rule" % (3 * sg))
        peaks = []
        if hmax >= 4:
            if crit == "deconv":
                cfw = max(1.5, 0.3 * ppd)
                cf = max(floor_px, 0.01 * hmax)
                pk = deconv_peaks_profile(sm, cfw, cf)
                print("   peak-crit deconv: fwhm-guess %.1fpx floor %.2f -> %d candidates"
                      % (cfw, cf, len(pk)))
            else:
                pk = find_peak_idx(sm, hmax, crit, args.sigma_k, floor_px,
                                   dd, (w_lo, w_hi + 1))
                print("   peak-crit %s: sigma=%.2fpx floor=%.2f -> %d candidates"
                      % (crit, sg,
                         max(floor_px, args.sigma_k * sg), len(pk)))
            cand = []
            for j in pk:
                xc = parabola_refine(xs, sm, int(j))
                cand.append([xc, a + b * xc, sm[int(j)]])
            for c in cand:
                dm = np.abs(refarr - c[1])
                tag = ""
                if refarr.size:
                    k = int(dm.argmin())
                    amb = np.abs(refarr - c[1]) < 0.25
                    if dm[k] < 0.35:
                        tag = "/".join(refnames[refarr[k]])
                    if amb.sum() > 1:
                        tag = "/".join(refnames[refarr[j]][0]
                                        for j in np.nonzero(amb)[0])
                peaks.append({"xc": c[0], "th2": c[1], "I": c[2], "tag": tag})
            peaks.sort(key=lambda q: q["th2"])
            merged = []
            for q in peaks:
                if merged and q["tag"] and q["tag"] == merged[-1]["tag"] and \
                        q["th2"] - merged[-1]["th2"] < 0.6:
                    if q["I"] > merged[-1]["I"]:
                        merged[-1] = q
                else:
                    merged.append(q)
            peaks = merged
            conf_fams = {p["tag"][0] for p in peaks
                         if p["tag"] and "/" not in p["tag"]}
            for p in peaks:
                if "/" in p["tag"]:
                    fams = set(p["tag"].split("/")) & conf_fams
                    if len(fams) == 1:
                        p["tag"] = next(iter(fams)) + "?"
        hmax_safe = hmax if hmax > 0 else 1
        tr = {"baseline": bl, "hmax": float(hmax), "lowconf": lowconf,
              "xs": xs, "I": I, "peaks": peaks, "apex": dict(t["apex"])}
        all_tr.append(tr)
        print(f"trace {i}: bl={bl} hmax={hmax:.0f}px n_peaks={len(peaks)}" +
              (" LOW-CONFIDENCE" if lowconf else ""))
        for p in peaks:
            dsp = args.lamb / (2 * np.sin(np.radians(p["th2"]) / 2))
            rows_out.append({"trace": i, "2theta": round(p["th2"], 2),
                             "d_A": round(dsp, 3),
                             "Irel": round(p["I"] / hmax_safe, 3),
                             "phase": p["tag"],
                             "conf": "low" if lowconf else "ok"})
            print(f"   2th={p['th2']:6.2f} d={dsp:.3f} Irel={p['I']/hmax_safe:.2f} {p['tag']}")

    with open(os.path.join(args.outdir, "peaks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trace", "2theta", "d_A", "Irel", "phase", "conf"])
        w.writeheader()
        w.writerows(rows_out)

    # --- acceptance QC: coverage / unexplained strong peaks / systematic shift ---
    QC_STRONG = 0.15
    QC_COV_MIN = 0.80
    QC_OFF_MAX = 0.15
    QC_DRIFT = 0.35
    fam_re = re.compile(r"^[A-Za-z]+")

    qc_lines = []
    if not refs:
        qc_lines.append("QC: no reference lines (--refs none): coverage / offset checks skipped.")
        qc_lines.append("    Cross-match peaks manually (e.g. vs an in-figure reference trace).")
    else:
        resid_pool = []
        for i, t in enumerate(all_tr):
            hmax_safe = t["hmax"] if t["hmax"] > 0 else 1.0
            peaks = t["peaks"]
            Itot = float(sum(p["I"] for p in peaks))
            firm = [p for p in peaks if p["tag"] and "/" not in p["tag"]]
            amb = [p for p in peaks if "/" in p["tag"]]
            unm = [p for p in peaks if not p["tag"]]
            Ifirm = float(sum(p["I"] for p in firm))
            cov = Ifirm / Itot if Itot > 0 else (1.0 if not peaks else 0.0)
            strong_unm = [p for p in unm if p["I"] / hmax_safe >= QC_STRONG]
            qc_lines.append(
                f"trace {i}: n_peaks={len(peaks)} firm={len(firm)} ambiguous={len(amb)} "
                f"unexplained={len(unm)} coverage(firm)={cov:.2f}")
            if strong_unm:
                lst = ", ".join(f"{p['th2']:.2f}deg(I={p['I']/hmax_safe:.2f})" for p in strong_unm)
                qc_lines.append(f"   !! strong unexplained peaks (Irel>={QC_STRONG}): {lst}")
            if amb:
                qc_lines.append(f"   ambiguous tags (A/R-type, unresolved): " +
                                ", ".join(f"{p['th2']:.2f}" for p in amb))
            if Itot and cov < QC_COV_MIN:
                qc_lines.append(f"   !! coverage {cov:.2f} < {QC_COV_MIN}: missing refs? "
                                "extra phase? wrong calibration?")
            for p in firm:
                if p["tag"] in refs:
                    fam = fam_re.match(p["tag"])
                    fam = fam.group(0) if fam else p["tag"]
                    resid_pool.append((refs[p["tag"]], p["th2"] - refs[p["tag"]], fam))
        fams = {}
        for rpos, d, fam in resid_pool:
            fams.setdefault(fam, []).append(d)
        for fam in sorted(fams):
            ds = fams[fam]
            mean_d = sum(ds) / len(ds)
            flag = " !! systematic shift" if abs(mean_d) > QC_OFF_MAX else ""
            qc_lines.append(f"phase {fam}: n={len(ds)} mean_offset={mean_d:+.2f}deg "
                            f"max|off|={max(abs(d) for d in ds):.2f}{flag}")
        if len(resid_pool) >= 3:
            rp = np.array([r for r, _, _ in resid_pool])
            rd = np.array([d for _, d, _ in resid_pool])
            if rp.std() > 1e-9:
                slope, off = np.polyfit(rp, rd, 1)
                drift_span = abs(slope) * (rp.max() - rp.min())
                mean_d = float(rd.mean())
                msg = (f"all phases: n={len(rd)} mean_offset={mean_d:+.2f}deg "
                       f"drift={slope:+.4f}deg/deg (span {drift_span:.2f}deg)")
                if abs(mean_d) > QC_OFF_MAX:
                    msg += "  !! uniform offset -> calibration zero/sample shift?"
                if drift_span > QC_DRIFT:
                    msg += "  !! drift -> check anchors (nonlinear/scale error?)"
                qc_lines.append(msg)
        else:
            qc_lines.append("all phases: too few exact matches (<3) for drift fit")

    qc_text = "\n".join(qc_lines)
    print("--- acceptance QC ---")
    print(qc_text)
    with open(os.path.join(args.outdir, "qc.txt"), "w", encoding="utf-8") as f:
        f.write(qc_text + "\n")

    # --- export full digitized curves per trace as 2-col .xy ---
    xydir = os.path.join(args.outdir, "traces_xy")
    os.makedirs(xydir, exist_ok=True)
    for i, t in enumerate(all_tr):
        pts = [(a + b * int(x), float(t["I"][int(x) - xl_i]))
               for x in sorted(t["apex"])
               if 0 <= int(x) - xl_i < len(t["I"]) and t["I"][int(x) - xl_i] > 0]
        if not pts:
            continue
        with open(os.path.join(xydir, f"trace{i}.xy"), "w") as f:
            f.write(f"# trace {i} digitized curve (Cu Ka {args.lamb} A); ")
            f.write(f"2theta Intensity, from {args.img}\n")
            for th2, iv in pts:
                f.write(f"{th2:.4f} {iv:.1f}\n")
        if args.resample and args.resample > 0 and len(pts) >= 3:
            th = np.array([p[0] for p in pts])
            Iv = np.array([p[1] for p in pts])
            step = args.resample
            g0 = np.ceil(th.min() / step) * step
            g = np.arange(g0, th.max() + 1e-9, step)
            if g.size >= 3:
                gi = np.clip(PchipInterpolator(th, Iv)(g), 0, None)
                with open(os.path.join(xydir, f"trace{i}_grid.xy"), "w") as f:
                    f.write(f"# trace {i} resampled to {step} deg (Cu Ka {args.lamb} A); "
                            f"PCHIP from {args.img}\n")
                    for tt, vv in zip(g, gi):
                        f.write(f"{tt:.4f} {vv:.1f}\n")
    print(f"trace curves (.xy) -> {xydir}" + (" (+ *_grid.xy)" if args.resample else ""))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(all_tr)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.0 * n + 0.5), sharex=True)
    if n == 1:
        axes = [axes]
    for i, t in enumerate(all_tr):
        ax = axes[i]
        th = a + b * t["xs"]
        ax.plot(th, t["I"], color="k", lw=0.7)
        for p in t["peaks"]:
            ax.annotate(f"{p['th2']:.1f}", (p["th2"], p["I"]), fontsize=6.5,
                        rotation=90, ha="right", va="bottom", color="b")
        for v in refarr:
            ax.axvline(v, color="g", lw=0.4, alpha=0.35)
        ax.set_ylabel(f"#{i}", fontsize=8)
        ax.set_ylim(0, max(t["hmax"] * 1.15, 1))
    axes[-1].set_xlabel("2theta (deg)")
    fig.suptitle(os.path.basename(args.img) + (" (green=ref lines)" if refarr.size else ""))
    fig.savefig(os.path.join(args.outdir, "overlay.png"), dpi=110)
    plt.close(fig)

    colors = plt.cm.tab10(np.linspace(0, 1, n))
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(np.asarray(Image.open(args.img).convert("RGB")))
    for i, t in enumerate(all_tr):
        xs = t["xs"]
        ys = np.array([t["apex"].get(x, np.nan) for x in xs])
        ax.plot(xs, ys, color=colors[i], lw=1.0, alpha=0.8)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title("validation: digitized traces on original geometry")
    fig.savefig(os.path.join(args.outdir, "validate.png"), dpi=110)
    plt.close(fig)
    print("outputs: peaks.csv qc.txt overlay.png validate.png in", args.outdir)


if __name__ == "__main__":
    main()
