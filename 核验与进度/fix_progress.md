# v11 progress (Asia/Seoul)

Updated: 2026-11-25 12:59 Asia/Seoul (UTC+9)

## Job
Fix v9 regressions (materials p5_02 calib, Co p4_01 all-zero crop), make phase fallback always produce a phase and include real phase time in walls, improve multi-curve traces / spine spikes / LDH Fig 4 axis. Then re-run all 13 papers no-fit and --fit, audit vs v9/v10, update 使用说明 and zip.

## Do not
- Overwrite v8/v9 or bench10/results/ (before) or results_v10/ (after-v10 baseline)
- Print or write DASHSCOPE_API_KEY

## Code status (done, compiled)
In `/workspace/xrd_qwen/trace_xrd.py` and `xrd_vl.py`:
- Prefer unsnapped VLM ticks when snapping to marks raises residual; do not clamp slightly-out-of-frame endpoints onto spines (that stretched 20–90 into a 2° residual).
- `verify_and_correct_fit` no longer adsors a systematic mark offset if residual would worsen (this had created 26°/33° residuals on zeolite and LDH Fig 1).
- `_tick_quality` penalizes out-of-frame ticks and spine mismatch vs labeled x-range; `_finalize_ticks` prefers in-frame mark ticks when residual ≤ 0.3° (VLM can be self-consistent yet shifted).
- Frame ranking: modest width bonus; require predicted 2θ span to cover ≥70% of the labeled range so a mid-peak strip cannot beat the full frame; cap nreg so dense false marks do not win; zeolite right zoom treated as inset.
- Trace: honor expected curve_count (TiO2 P25); skip band-filling boxes; spine inset and wider edge-spike window.
- Phase: any omni overdue/fail/error/empty → qwen3.8-flash; 3.8 403 → qwen3.6-flash; JSON salvage so phase is not empty.

## Offline redigitize (v10 calib+crops, new code) — /tmp/xrd_probe/v11_redig7
- materials p5_02: FIXED. frame [34,31,631,480], resid 0.001, range ~20.4–87.7° (ticks 30–80 on a 20–90 axis). No more −49–117 / 30° residual.
- Co p4_01: FIXED. frame [75,29,1238,624], resid 0.032, ticks 30–105, 6 non-zero traces.
- LDH Fig 4 (p7_02): FIXED. 10–70, resid 0.000, 3 traces.
- LDH Fig 9b (p11_02): protected, resid 0.001, 10–80.
- oxide: 20–70 resid 0.039, 5 traces (P25 back). Still boxy.
- zeolite: insets excluded, ticks 10–90 resid 0.043, 7.2° still inside the frame (x0=262, 7.2°≈272).
- Controls look preserved: dee, lfp 10–70, lipstick 7–43, BSM 10–70 resid 0.049 (30° tick path intact), mos2, mof, LSCO 5 traces, mat p5_01 ticks 20–90 resid 0.103.
- Remaining offline: HEA 曲线2 collapsed; oxide/HEA/LSCO traces still boxy; Co p4_02 still 1 trace (range 19–119); LDH Fig 1 only one of two same-scale panels (v10 also had only the left panel); mat p5_02 far-right ~2° short of printed 90 because VLM 90 sits past the spine.

## Official runs
Runner: `/tmp/xrd_probe/bench10/run_v11.py` (skips a paper if its v11 figures_index already exists).
Logs: `/tmp/xrd_probe/bench10_v11/`
Times: `/workspace/xrd_user/bench10/run_times_v11.json`
Out:
- `/workspace/xrd_user/results_{dee,mat,co}_v11` and `*_v11_fit`
- `/workspace/xrd_user/bench10/results_v11/<id>`, `<id>_fit`, `_batch_all`

Do not start audit / 使用说明 / zip until ALL_V11_DONE is in the runner log.

## Next if interrupted
1. If runner not finished: `python3 /tmp/xrd_probe/bench10/run_v11.py` (resumes).
2. Then strict per-figure audit vs v9 and v10, write `summary_v11.md/.csv`.
3. Update `使用说明.md`, rebuild `/workspace/xrd_qwen.zip` (no key, venv, results, caches).


## Official runner status (2026-11-25 13:09 Asia/Seoul)

Runner PID 1207081 still alive. Do not start a second runner.

All 13 no-fit official runs finished (rc=0) before BATCH_ALL started at 13:06:43. Current child is pdf_xrd writing `results_v11/_batch_all`. After that the same process will run all 13 `--fit`, then print ALL_V11_DONE.

`phase_empty == crop count` in run_times is likely an inspect_out false positive: figures_index.json often lacks `phase` keys, but `*.png.json` / report.md for dee already showed a non-empty phase after omni overdue → 3.8-flash, with 识图 26.81 s included in the 42.2 s wall. Verify this for all papers after ALL_V11_DONE; do not treat the official walls as missing phase time.

If the runner dies: resume only with `python3 /tmp/xrd_probe/bench10/run_v11.py` (skips existing figures_index). Never overwrite bench10/results/, results_v10/, or v9 dirs.


## Official runner status (2026-11-25 13:16 Asia/Seoul)

Still the same PID 1207081. Do not start a second runner.

No-fit 13/13 and BATCH_ALL (rc=2, 141.45 s, 18 crops) are done. --fit finished dee/mat/co/oxide/hea; ldh-mgga_fit started 13:14:11 and is still running. After that: mof, zeolite, LFP, MoS2, LSCO, lipstick, BSM20.

Phase: inspect_out flagged phase_empty because figures_index lacks phase keys. Official `*.png.json` / report.md already contain non-empty phases after omni overdue → 3.8-flash, and 识图 time is in the walls (dee 26.8 s phase inside 42.2 s). Confirm for all papers after ALL_V11_DONE.

Audit notes so far (no-fit, not final until visual pass is complete):
- mat p5_02 axis 20–90 fixed (no −49–117); 400 °C missing from xy (4/5 traces).
- Co p4_01 no longer all-zero; full-height stack is traced, but the crop cuts ~100–105° (103 peak).
- LDH Fig 4 axis 10–70 fixed (no −15–96); traces look usable — candidate new accurate.
- LDH 9b still looks good; MoS2, MOF, BSM20 30° look held.
- lipstick: v11 lost the ~7.26° clinochlore peak and picked up 10–13° card-stick bars (v10 replot had 7.26 and no sticks) — likely loses accurate.
- zeolite 7.2° recovered, inset excluded.
- Fig 4c neutron still absent (FP 0). LDH Fig 6 only panel (a); Fig 6b not a separate crop (v10 had a split).


## Official runs complete (2026-11-25 13:24 Asia/Seoul)

ALL_V11_DONE. Did not start a second runner. Audit written:

- `/workspace/xrd_user/bench10/summary_v11.md`
- `/workspace/xrd_user/bench10/summary_v11.csv`

Headline: bench10 accurate 2/18 → 5/18 → 5/18 (lost lipstick, gained LDH Fig 4). Combined with 6 original figures: 4/24 → 5/24 → 5/24. Materials p5_02 axis and Co p4_01 all-zero regressions are fixed but those two figures are still not fully accurate. Phase was not actually empty: omni overdue → 3.8 inside the official walls; text is in png.json. figures_index now (code) copies phase for future runs.

使用说明.md updated. Zip rebuilt at `/workspace/xrd_qwen.zip` (24 files, no key/venv/results/caches).


---

# v12 progress (Asia/Seoul)

Updated: 2026-11-25 14:16 Asia/Seoul (UTC+9)

## Job
Improve digitize accuracy by color-first tracing (split per legend color, near-color separation, mask grid/text/card sticks, reconnect boxy/holes). Keep v11 frame/tick constraints. Do not overwrite v8/v9, bench10/results/, results_v10/, results_v11/, or external_patch/. Never print or write DASHSCOPE_API_KEY.

## Status
- Inspected v11 hard cases and external_patch review. Will merge only color tracking / gap reconnect / persist curves; will NOT wholesale-replace trace_xrd.py.
- Hard-case notes (v11 official):
  - TiO2 Fig 3: 5 names including P25, axis 20–70, but P25 is almost empty/boxy (nz=0.03); other traces also boxy. Colors are distinct (black/red/blue/magenta/green stacked).
  - HEA / LSCO: boxy, some collapsed.
  - LFP: 2a frame kept (10–70); PDF card sticks sit under the black LFP curve — must mask, not trace.
  - lipstick: 104 still at 38.70, but 7.26° lost (y=0) and 9.9–13° card-stick bars appeared. Curve is black; colored symbols are JCPDS markers.
  - Co p4_01: 5/6 traces (H1.25-OR missing), frame cuts near ~80–83°; printed axis goes to 105° with (110)/(103).
  - materials p5_02: axis 20–90 OK, missing 400 °C (bottom purple).
  - LDH 9b: axis OK but official experimental (black) trace is nearly empty; simulation (red) present. Need both.
- figures_index currently drops `curves` / `color_rgb` / `plot_box_1000` / `x_range`; official runs have colors in memory only. Will persist them, and infer/refine colors from the crop for offline redigitize.
- Color-first tracing is in `trace_xrd.py` (v11 tick/frame logic kept; no wholesale overwrite from external_patch). `pdf_xrd.py` persists curves / color_rgb / plot_box_1000 / x_range / mask_boxes / phase into figures_index.
- Useful patch ideas taken carefully: short-gap `_color_track`, honor curve_count / don't shrink away sparse peaks, reject widen if residual rises by ≥0.05°. Empty-frame retry that swallows insets was NOT taken. Widen stops on a ≥12 px low-ink gutter (LFP 2b).
- Also: skip oversized VLM "label" boxes (area>6% or a wide strip ≥14% of frame height) so they cannot paint out a stacked bottom curve; do not replace a named chromatic color with axis-ink black; empty color-band slices fall back to ink.

## Official runs
Started 2026-11-25 14:06 Asia/Seoul (UTC+9). Runner: `/tmp/xrd_probe/bench10/run_v12.py`.
Logs: `/tmp/xrd_probe/bench10_v12/`. Times: `/workspace/xrd_user/bench10/run_times_v12.json`.
Out (new dirs only; v11/v10/v9 untouched):
- `/workspace/xrd_user/results_{dee,mat,co}_v12` and `*_v12_fit`
- `/workspace/xrd_user/bench10/results_v12/<id>`, `<id>_fit`, `_batch_all`

Sequential no-fit (13 papers) finished ~14:14. Batch started 14:14:12. Fit still pending at last update.

### Hard cases (official no-fit, first pass)
- TiO2 Fig 3: 5 traces incl. P25, axis 20.44–69.50°. P25 has real peaks (25.3/27.4/37.8/48) but still gappy (hole=0.87). Not yet complete/accurate.
- materials p5_02: axis labeled 20–90 but xy 20.66–92.94° (scale slightly long, 33° reads ~33.9°). Official first pass still 4 traces (800/700/600/500); 400 °C was dropped because a 13% VLM label box covered the bottom purple strip. A later code guard recovers 400 °C on this crop; official first-pass dir was not overwritten mid-run.
- Co p4_01: **6 named non-zero traces** (H1.25-OR through H0.0-OR), axis 29.98–107.68°, 100–105° no longer cut. Main (101) ~55.5° on all six. (103) still weak. Not yet whole-figure accurate, but the v10 all-zero / v11 right-cut issues are gone.
- Co p8_01: still ends ~97.8° vs printed 105°.
- LFP Fig 2a: 2a/2b split kept (10.40–69.76°, no 70–77° inset). 4 traces, olivine peaks present. Black LFP still has some mid-range card-stick intensity; improved vs v11 square waves, not fully clean.
- lipstick Fig 3: 104 at **38.70** (ymax=272), 31.04 quartz held. ~7.26 still 0 (same as v11). 10.8–11.2 stick bars remain; 9.9/12.5 cleaned vs v11.
- HEA / LSCO: still holed/boxy, not accurate.

### Controls
- MoS2 Fig 2a: preserved (14.20 / 32.64°).
- ZIF Fig 1: 6 traces (was 5); ZIF-8 7.30 / 12.66 / 18.00 held. Weak extra ZIF-90 calculated.
- BSM20 Fig 3: **possible regression** — tick fit residual still 0.049°, but digitized main peak tops ~30.54° (flag: +0.56° vs neighbouring tick; not applied). v11 sat on 30.02°. Range also slightly short of 70°.
- LDH Fig 9b: **regression** — experimental lost the ~11° peak (tops sit at 64–70°); simulation still has ~11° and ~35–37°.
- LDH Fig 4: axis 10.30–70.38° held; experimental 11.08 present but 12.5/14.7 dominate (boxy). Borderline vs v11.

## Comparison PNGs
`/workspace/xrd_user/bench10/compare_v12/*_official_v11_v12.png`

## Docs / zip
使用说明.md updated for color-first tracing. Zip rebuilt `/workspace/xrd_qwen.zip` (24 files, no key/venv/results/caches) at 2026-11-25 14:15 Asia/Seoul.

## Do not
- Overwrite v8/v9, bench10/results/, results_v10/, results_v11/, external_patch/
- Print or write DASHSCOPE_API_KEY


## Official v12 completed (2026-11-25 14:33:20 Asia/Seoul)

ALL_V12_DONE. 13/13 `--fit` rc=0. See `summary_v12.md` / `summary_v12.csv` / `run_times_v12.json`.
Window: no-fit 14:06:18–14:14:12 (median 32.11 s, sum 474.51 s); batch 14:14:12–14:16:31 (138.13 s); `--fit` 14:16:31–14:33:20 (sum 1009.12 s, median 65.59 s, max 172.26 s ldh).

**BSM20 audit correction:** the “main peak +0.56° vs v11 30.02°” flag also exists on official v11 xy (tops ~30.50–30.54°). Official v12 curve a already reaches ymax at 30.10°. BSM is kept accurate. Official no-fit bench10 is **4/18** (not 3/18): kept MoS2, ZIF, LDH Fig 4, BSM20; lost only LDH Fig 9b. Combined with 6 originals: **4/24**.

Official first-pass dirs were not overwritten. `results_mat_v12` still has 4 traces; `results_mat_v12_fit` recovered 400 °C (5 traces) after the oversized-box guard.

## v13 accuracy follow-up (after official v12; new dirs only)

Priority: recover LDH Fig 9b experimental ~11°, and lipstick ~7.26° chlorite without card-stick bars, without losing MoS2, ZIF, LDH Fig 4, Co p4_01 six traces to ~107°, TiO2 P25, LFP 2a/2b, or lipstick 104 at 38.70°.

Code (in `/workspace/xrd_qwen/trace_xrd.py`; official v12 result dirs untouched):

1. `refine_ticks_from_axis_labels` after `maybe_widen`, before `verify_and_correct_fit`. Accept only when residual ≤0.20°, new span ≥85% of the current tick span, and the new range does not shrink past the old first/last by >5°. In-frame ticks may omit a label that sits on the right spine (9b: 10–70 in-frame, printed 80 at x1); the major-tick grid is extended so that case is accepted.
2. Colourful `vstick` markers are erased only when they form a comb of ≥2 similar verticals; isolated magenta peak stems are kept. Compact-marker rule no longer treats a tall thin stem (h > 2w, w < 6) as a triangle.
3. Black/ink curves OR in high-chroma ink that touches the black mask (coloured cores of a black peak, e.g. lipstick chlorite), without swallowing a separate green baseline.
4. `_trace_column` allows a wiped peak foot up to 40 px above the floor, and skips short mid-air letter remnants in the upper half of the band (lipstick 10.8–11.2 number-ladder leftovers).
5. Edge `_remove_vertical_spikes` keeps a lone sharp peak (chlorite); only flattens edge spikes that have a neighbouring comb.

### After-fix redig (offline, existing v12 crops + calib)

Outputs: `/workspace/xrd_user/bench10/compare_v13/redig/<tag>/` (not official v12). Timing ~42 s for 9 control figures, no API.

| figure | result |
| --- | --- |
| LDH Fig 9b | **Recovered.** Label retick residual 0.000→0.091°, 8 labels. Experimental low-angle peak at **11.34°** (11.0/11.4 = 96; 9.9 = 0). Simulation stays distinct (22, 34, 37, **38.3** main, 60). The sim main moved from v12 35.4° / v11 37.1° because the whole axis was shifted with the 10° label; visually the tall peak sits just left of printed 40°. |
| lipstick Fig 3 | **7.26 chlorite recovered** (y=57.5). 9.9 / 10.8 / 11.2 / 12.5 card-stick bars = 0. Quartz 31.04 held (163). **104 still present at 38.70°** (y=176, within 0.3° of argmax 38.42). Caveat: official v12 had a sharp tip 38.70/272 on top of the 38.44 shoulder; v13 follows the shoulder (38.42/176) and the sharp 262 tip is weaker (~158 at 38.68–38.72). |
| BSM20 | Unchanged vs official v12: ymax already on 30.02–30.54, +0.56° flag kept (same as v11 xy). Not a real regression. |
| MoS2 | Held: 14.20 / 32.64 / 58.40, range 10.50–89.50. |
| ZIF | Held: 6 traces, 7.30 / 12.66 / 18.00, range 5.22–49.76. |
| LDH Fig 4 | Held: axis 10.44–69.62, experimental 11.08 present; main experimental tops ~12.52 / 14.46 (same two peaks as official v12). Label retick 0.000→0.035, 7 labels. |
| LFP | 2a/2b split held, 4 traces, olivine 17.16 / 25.60 / 29.70 / 35.60 present on black LFP (17.16 y=52 vs ~65 before, still in window). |
| TiO2 | P25 present (25.3 y=59), 5 traces, range 20.44–69.50. 500 °C still weak. |
| Co p4_01 | 6 named traces, 29.98–107.68°, main ~55.5° held. |

### Accurate counts

- v11 official no-fit: bench10 **5/18**, combined **5/24**.
- official v12 no-fit (after BSM correction): bench10 **4/18**, combined **4/24** (lost only 9b).
- v13 after-fix (redig of affected + controls, not a new 13-paper official run): bench10 **5/18**, combined **5/24**. LDH 9b recovered. Lipstick is improved (7.26 + clean 10.8–11.2) but not flipped to accurate because the 104 sharp tip is weaker than v12 (argmax 38.42 vs 38.70).

Comparison PNGs: `/workspace/xrd_user/bench10/compare_v13/*_v11_v12_v13_*.png`.

使用说明.md updated for axis-label retick and isolated coloured peak cores. Zip rebuilt after this section (no key/venv/results/caches).
