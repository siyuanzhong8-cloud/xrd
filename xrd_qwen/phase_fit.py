#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 3 步（可选）：根据图注/元素线索提出候选物相，下载 CIF，用 DARA+BGMN 拟合 .xy。

本模块在未安装 dara-xrd / BGMN 时不会报死，只返回 status=unavailable 和安装说明。
不要把 dara-xrd 装进本工具的主虚拟环境（依赖很重，且需钉死 pydantic 版本）。
请使用单独的 dara-venv，并用环境变量 XRD_DARA_PYTHON 指向其中的 Python。
脚本也会自动探测本目录下的 dara-venv/bin/python 或 dara-venv/Scripts/python.exe。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

VENDOR = Path(__file__).resolve().parent / "vendor"

INSTALL_HINT = """DARA 拟合需要单独的环境（不要装进本工具的 .venv）：

  python -m venv dara-venv
  dara-venv\\Scripts\\python -m pip install "dara-xrd==1.3.0" "pydantic<2.14"   # Windows
  dara-venv/bin/python -m pip install "dara-xrd==1.3.0" "pydantic<2.14"          # Linux / macOS

然后设置（也可不设，脚本会自动探测 ./dara-venv）：

  setx XRD_DARA_PYTHON "%CD%\\dara-venv\\Scripts\\python.exe"   # Windows，重开终端
  export XRD_DARA_PYTHON="$PWD/dara-venv/bin/python"          # Linux / macOS

首次拟合时，DARA 会按操作系统自动下载 BGMN：
  https://cedergrouphub.github.io/dara/_static/bgmnwin_Linux.zip
  https://cedergrouphub.github.io/dara/_static/bgmnwin_Windows.zip
  https://cedergrouphub.github.io/dara/_static/bgmnwin_Darwin.zip

也可预先下载：

  dara-venv/bin/python -c "from dara.bgmn.download_bgmn import download_bgmn; download_bgmn()"

说明：
- Linux 官方二进制（BGMN 4.2.23）可直接运行，一般不需要 Wine。若出现 GLIBC_2.29 not found，请按 DARA 文档用 patchelf 处理。
- 较新的 scipy 下，dara/search/peak_matcher.py 的 cdist 需要标量距离函数。phase_fit.py 探测环境时会自动打补丁。
- 拟合可能要数分钟，请用 --fit 显式打开；默认关闭。数字化曲线没有真实计数统计，Rwp/GOF 只宜作相对参考。
"""

# 亚化学计量的 K-β-ferrite 在 COD 中不以 K2Fe22O34 登录
KNOWN_COD = {
    "K2Fe22O34": [("1529668", "K-β-ferrite / K2Fe22O34（COD 1529668，Fe10.92K1.55O17）")],
    "Fe2O3": [("1011240", "Hematite α-Fe2O3（COD 1011240，R-3c）")],
    "K2O": [("1010878", "K2O（COD 1010878）")],
    "Co": [("9008492", "hcp Co（COD 9008492，P63/mmc）")],
    "Zn": [("9008522", "hcp Zn（COD 9008522，P63/mmc）")],
}

PREFERRED_GROUPS = {
    "Fe2O3": ("R -3 c", "R-3c", "R -3 c :R", "167"),
    "Co": ("P 63/m m c", "P63/mmc", "P 63/mmc", "194"),
    "Zn": ("P 63/m m c", "P63/mmc", "P 63/mmc", "194"),
}

# 简单无机化学式：可从「金属锌(Zn)」这类中文名里抽出
_ELEMENT_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?)(?:[A-Z][a-z]?|\d+){0,12}$")
_SKIP_FORMULA_TOKENS = {
    "OH", "H2O", "H2", "O2", "N2", "CO2", "NH3", "XH2O", "NH2O", "AQ", "OTF", "TFSI", "OH2",
}

_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉＋－²₃", "0123456789+-23")
_FORMULA_RE = re.compile(
    r"(?:α-|β-|γ-|δ-|α|β|γ|δ|a-|b-|g-)?[A-Z][a-z]?(?:[A-Z][a-z]?|\d+|₂|₃|₄|₅|₆|₇|₈|₉)+"
)


def _candidate_pythons() -> list[Path]:
    found: list[Path] = []
    env = os.environ.get("XRD_DARA_PYTHON", "").strip()
    if env:
        found.append(Path(env))
    here = Path(__file__).resolve().parent
    found.extend(
        [
            here / "dara-venv" / "bin" / "python",
            here / "dara-venv" / "bin" / "python3",
            here / "dara-venv" / "Scripts" / "python.exe",
            here / "dara-venv" / "Scripts" / "python",
            Path.home() / ".openclaw/workspace/work/dara-venv/Scripts/python.exe",
            Path.home() / "dara-venv" / "bin" / "python",
            Path.home() / "dara-venv" / "Scripts" / "python.exe",
        ]
    )
    found.append(Path(sys.executable))
    out: list[Path] = []
    seen = set()
    for p in found:
        try:
            rp = p.resolve() if p.exists() else p
        except OSError:
            rp = p
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def apply_peak_matcher_patch(python_exe: Path) -> str:
    """修补 dara.search.peak_matcher：新版 scipy 的 cdist 要求标量 metric。"""
    code = r"""
import sys
from pathlib import Path
import dara.search.peak_matcher as pm
p = Path(pm.__file__)
text = p.read_text(encoding="utf-8")
old = "metric=absolute_log_error,"
new = "metric=lambda u, v: float(np.asarray(absolute_log_error(u, v)).ravel()[0]),"
if "lambda u, v: float(np.asarray(absolute_log_error" in text:
    print("already")
elif old in text:
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("patched")
else:
    print("missing")
"""
    try:
        proc = subprocess.run(
            [str(python_exe), "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=40,
        )
    except Exception as exc:
        return f"error:{exc}"
    return (proc.stdout or proc.stderr or "").strip() or f"exit:{proc.returncode}"


def probe_dara(python_exe: Path | None = None) -> dict[str, Any]:
    """探测给定（或候选）Python 是否装有 dara，以及 BGMN 是否可用。"""
    infos = []
    targets = [python_exe] if python_exe else _candidate_pythons()
    for exe in targets:
        if not exe or not Path(exe).exists():
            continue
        code = r"""
import json, sys
from pathlib import Path
info = {"python": sys.executable, "dara": False, "dara_version": None, "bgmn": False, "bgmn_path": None, "error": ""}
try:
    import dara
    info["dara"] = True
    info["dara_version"] = getattr(dara, "__version__", "unknown")
except Exception as e:
    info["error"] = "import dara failed: %s" % e
    print(json.dumps(info, ensure_ascii=False))
    sys.exit(0)
try:
    from dara.bgmn.download_bgmn import download_bgmn
    bgmn_dir = Path(dara.__file__).resolve().parent / "bgmn" / "BGMNwin"
    bgmn = bgmn_dir / "bgmn"
    bgmn_exe = bgmn_dir / "bgmn.exe"
    if not bgmn.exists() and not bgmn_exe.exists():
        download_bgmn()
    path = bgmn if bgmn.exists() else bgmn_exe
    info["bgmn"] = path.exists()
    info["bgmn_path"] = str(path) if path.exists() else None
except Exception as e:
    info["error"] = (info.get("error") or "") + "; bgmn probe: %s" % e
print(json.dumps(info, ensure_ascii=False))
"""
        try:
            proc = subprocess.run(
                [str(exe), "-c", code],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
        except Exception as exc:
            infos.append({"python": str(exe), "dara": False, "error": str(exc)})
            continue
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            infos.append(
                {
                    "python": str(exe),
                    "dara": False,
                    "error": (proc.stderr or proc.stdout or "probe failed")[:500],
                }
            )
            continue
        infos.append(payload)
        if payload.get("dara"):
            patch_state = apply_peak_matcher_patch(Path(payload.get("python") or exe))
            payload["peak_matcher_patch"] = patch_state
            return {
                "available": True,
                "ready": bool(payload.get("dara")),
                "bgmn": bool(payload.get("bgmn")),
                "python": payload.get("python") or str(exe),
                "version": payload.get("dara_version"),
                "bgmn_path": payload.get("bgmn_path"),
                "error": payload.get("error") or "",
                "peak_matcher_patch": patch_state,
                "hint": INSTALL_HINT,
                "probed": infos,
            }
    return {
        "available": False,
        "ready": False,
        "bgmn": False,
        "python": None,
        "version": None,
        "error": "未找到已安装 dara-xrd 的 Python",
        "hint": INSTALL_HINT,
        "probed": infos,
    }


def _ascii_chem(text: str) -> str:
    return (text or "").translate(_SUB).replace("α", "a").replace("β", "b").replace("γ", "g").replace("δ", "d")


def _alias_formulas(text: str) -> list[str]:
    key = re.sub(r"\s+", " ", _ascii_chem(text).lower())
    rules = [
        (r"k2fe22o34|铁酸二钾|k-?b-?ferrite|k-?beta-?ferrite|potassium beta ?ferrite", "K2Fe22O34"),
        (r"hematite|赤铁矿|a-?fe2o3|fe2o3", "Fe2O3"),
        (r"k2o|氧化钾|dipotassium oxide", "K2O"),
        (r"hcp[-\s]?co|六方密堆积钴|金属钴|\bcobalt\b|\bhcp\b.*\bco\b", "Co"),
        (r"金属锌|metallic zinc|zinc metal|hcp[-\s]?zn|(?<![A-Za-z])zinc(?![A-Za-z])", "Zn"),
    ]
    found: list[str] = []
    for pat, formula in rules:
        if re.search(pat, key) and formula not in found:
            found.append(formula)
    return found


def _alias_formula(text: str) -> str | None:
    found = _alias_formulas(text)
    return found[0] if found else None


def formulas_from_candidates(candidates: list[Any], caption: str = "") -> list[dict[str, str]]:
    """从模型物相候选和图注里收集要去 COD 查的化学式/名称。"""
    out: list[dict[str, str]] = []
    seen = set()

    def _ok_token(token: str) -> bool:
        if not token:
            return False
        key = token.upper().replace("·", "").replace(".", "")
        if key in _SKIP_FORMULA_TOKENS or "OTF" in key or "TFSI" in key:
            return False
        if token in KNOWN_COD or re.fullmatch(r"[A-Z][a-z]?", token):
            return True
        # 真正的无机式通常带数字（Fe2O3、TiO2）；不要把 OH、OTf 当成相
        return bool(_ELEMENT_FORMULA_RE.fullmatch(token) and re.search(r"\d", token) and re.search(r"[A-Z]", token))

    def _extract_formula(name: str, formula: str = "") -> str:
        formula = _ascii_chem(formula or "").strip()
        blob = " ".join(x for x in (name, formula) if x)
        alias = _alias_formula(blob) or _alias_formula(formula) or _alias_formula(name)
        if alias:
            return alias
        # 先在名称里找完整无机式（LaCoO3、(LiFePO4)），不要先退化成单个元素 Co/La
        ascii_blob = _ascii_chem(blob)
        candidates: list[str] = []
        if formula:
            candidates.append(re.sub(r"\s+", "", formula))
        for chunk in re.findall(r"\(([^()]{1,40})\)", ascii_blob):
            candidates.append(re.sub(r"\s+", "", chunk).replace("·", "").replace(".", ""))
        for m in _FORMULA_RE.finditer(ascii_blob):
            candidates.append(m.group(0).replace("·", "").replace(".", "").replace(" ", ""))
        cleaned = []
        for token in candidates:
            token = token.replace("a-", "").replace("b-", "").replace("g-", "").replace("d-", "")
            token = token.replace("α-", "").replace("β-", "").replace("γ-", "").replace("δ-", "")
            aliased = _alias_formula(token)
            if aliased:
                cleaned.append(aliased)
                continue
            if _ok_token(token):
                cleaned.append(token)
        if cleaned:
            # 优先更长的化学式
            cleaned.sort(key=lambda s: (s not in KNOWN_COD, -len(s)))
            return cleaned[0]
        for m in re.finditer(r"(?<![A-Za-z])([A-Z][a-z]?)(?![A-Za-z])", ascii_blob):
            el = m.group(1)
            if el in KNOWN_COD or el in {"Zn", "Co", "Fe", "Ni", "Cu", "Al", "Ti", "Si", "Mn", "Mg", "Sn", "Ag", "Au", "Pt", "Pd", "W", "Mo", "V", "Cr"}:
                return el
        return formula if formula and _ok_token(formula) else ""

    def add(name: str, formula: str = "", source: str = "") -> None:
        formula = _extract_formula(name, formula)
        alias = _alias_formula(formula or name)
        if alias:
            formula = alias
        # 没有可检索化学式时不要把中文整名交给 COD
        if not formula:
            return
        key = formula.strip()
        if not key or key.lower() in seen:
            return
        seen.add(key.lower())
        out.append({"name": name or formula, "formula": formula, "source": source})

    for c in candidates or []:
        if isinstance(c, dict):
            add(str(c.get("name") or ""), str(c.get("formula") or ""), "model")
        elif isinstance(c, str):
            add(c, "", "model")
    text = caption or ""
    for alias in _alias_formulas(text):
        add(alias, alias, "caption")
    for m in _FORMULA_RE.finditer(_ascii_chem(text)):
        token = m.group(0)
        token = token.replace("a-", "").replace("b-", "").replace("g-", "").replace("d-", "")
        if re.search(r"\d", token) and len(token) >= 3:
            add(token, token, "caption")
    # 图注里单独出现的金属 Co / Zn（无下标）也收进来
    ascii_text = _ascii_chem(text)
    if re.search(r"(?i)(?<![A-Za-z])Co(?![A-Za-z])", ascii_text):
        add("Co", "Co", "caption")
    if re.search(r"(?i)(?<![A-Za-z])Zn(?![A-Za-z])", ascii_text) or re.search(r"金属锌|zinc", ascii_text, re.I):
        add("Zn", "Zn", "caption")
    return out[:12]


def default_cif_cache() -> Path:
    """全局 CIF 缓存（可用 XRD_COD_CACHE 覆盖）。新的 out_dir 不再各自下一遍。"""
    env = os.environ.get("XRD_COD_CACHE", "").strip()
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent / "cod_cache"


_RIETVELD_RE = re.compile(
    r"rietveld|y\s*obs|y\s*calc|obs\s*[-–]\s*calc|observed\s+and\s+calculated|"
    r"difference\s+curve|差谱|精修（结果|图）|精修结果|calculated\s+pattern",
    re.I,
)


def is_published_rietveld(*texts: Any) -> bool:
    """判断是否为论文里已经发表的 Rietveld 精修图（再拟合会退化甚至 ZeroDivisionError）。"""
    blob = " ".join(str(t or "") for t in texts)
    if not blob.strip():
        return False
    return bool(_RIETVELD_RE.search(blob))


def _download_cif_direct(entry_id: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{entry_id}.cif"
    if dest.is_file() and dest.stat().st_size > 200:
        return dest
    last_err = None
    for site in ("https://www.crystallography.net/cod", "https://qiserver.ugr.es/cod"):
        try:
            req = Request(f"{site}/{entry_id}.cif", headers={"User-Agent": "XRD-Workbench/1.0"})
            with urlopen(req, timeout=30) as resp:
                payload = resp.read(2 * 1024 * 1024)
            if b"_atom_site_" not in payload.lower():
                raise ValueError("not a cif")
            dest.write_bytes(payload)
            return dest
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"未能下载 COD {entry_id}: {last_err}")


def fetch_cifs(items: list[dict[str, str]], cache_dir: Path, limit_per_formula: int = 2) -> tuple[list[dict[str, Any]], list[str]]:
    """用 vendor/cod_refs 按化学式检索并下载 CIF。失败不抛到外层。"""
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import cod_refs

    cache_dir = Path(cache_dir).resolve()
    cod_refs.CACHE = cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    refs: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    def add_ref(item: dict[str, str], row_id: str, name: str, formula: str, group: str = "") -> None:
        if row_id in seen_ids:
            return
        try:
            try:
                path = cod_refs.cif(row_id)
            except Exception:
                path = _download_cif_direct(row_id, cache_dir)
        except Exception as exc:
            errors.append(f"COD {row_id}: {exc}")
            return
        seen_ids.add(row_id)
        refs.append(
            {
                "name": name or item.get("name") or formula,
                "formula": formula,
                "cod_id": row_id,
                "path": str(Path(path).resolve()),
                "source": f"COD {row_id}",
                "group": group,
                "uploaded": False,
            }
        )

    for item in items:
        formula = (item.get("formula") or "").strip()
        if not formula:
            errors.append(f"{item.get('name') or '（未命名）'}: 无法解析为可检索化学式，已跳过")
            continue
        formula = _alias_formula(formula) or _ascii_chem(formula).replace(" ", "")
        if not formula or not re.search(r"[A-Z]", formula) or not _ELEMENT_FORMULA_RE.fullmatch(formula):
            # 含配体的复杂式（OTf 等）COD 检索几乎必失败，记一条错误后跳过
            errors.append(f"{item.get('name') or formula}: 请输入化学式，例如 TiO2、ZnO 或 NaCl。复杂组成也可直接填写 COD 编号。")
            continue
        for cid, label in KNOWN_COD.get(formula, []):
            add_ref(item, cid, label, formula)
        try:
            result = cod_refs.search(formula)
        except (cod_refs.CodUnavailable, ValueError) as exc:
            if formula not in KNOWN_COD:
                errors.append(f"{formula}: {exc}")
            continue
        rows = list(result.get("results") or [])
        preferred = PREFERRED_GROUPS.get(formula, ())

        def rank(row: dict[str, Any]) -> tuple[int, int]:
            name = (row.get("name") or "") + " " + (row.get("title") or "")
            group = str(row.get("group") or "")
            pref_name = 0
            if formula == "Fe2O3" and "hematite" in name.lower():
                pref_name = -2
            if formula == "Co" and "cobalt" in name.lower():
                pref_name = -1
            pref_sg = -3 if any(p.lower().replace(" ", "") in group.lower().replace(" ", "") for p in preferred) else 0
            # 避开 P1 这种可能被还原坏的钴结构
            if formula == "Co" and group.replace(" ", "") in {"P1", "P 1"}:
                pref_sg = 9
            return (pref_sg + pref_name, int(row.get("year") or 0) * -1)

        rows.sort(key=rank)
        already = sum(
            1
            for r in refs
            if (r.get("formula") or "").replace(" ", "") in {formula, (item.get("formula") or "").replace(" ", "")}
            or _alias_formula(r.get("formula") or "") == formula
        )
        # 已有首选 COD 条目时，Fe2O3/K2Fe22O34 不再追加同类结构，避免把同一物相当成两相
        if formula in KNOWN_COD and formula != "Co":
            extra_limit = 0
        else:
            extra_limit = max(0, limit_per_formula - already)
        for row in rows[:extra_limit]:
            add_ref(
                item,
                row["id"],
                item.get("name") or row.get("name") or formula,
                result.get("formula") or formula,
                row.get("group") or "",
            )
    return refs, errors


WORKER_SOURCE = r'''# -*- coding: utf-8 -*-
"""Standalone DARA worker used by phase_fit.py (runs inside the dara-venv)."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator

def prepare_pattern(source, step, destination):
    points = np.loadtxt(source, ndmin=2)
    points = points[(points[:, 0] > 0) & (points[:, 0] < 180)]
    angles, intensity = points.T
    if len(angles) < 3 or not np.all(np.diff(angles) > 0) or np.ptp(intensity) <= 0:
        raise ValueError("Invalid diffraction trace")
    # 描点有时会外推到第一根刻度左侧（例如 2°）；BGMN 默认仪器函数大约从 ~10° 起
    lo = max(float(angles[0]), 5.0)
    hi = min(float(angles[-1]), 140.0)
    if hi - lo >= 5:
        keep = (angles >= lo) & (angles <= hi)
        angles, intensity = angles[keep], intensity[keep]
    grid = angles[0] + np.arange(1 + int((angles[-1] - angles[0]) / step)) * step
    curve = PchipInterpolator(angles, intensity)(grid)
    # 数字化曲线没有真实计数，只把强度抬到正的相对计数，便于 BGMN 跑起来
    counts = 100 + 900 * (curve - curve.min()) / np.ptp(curve)
    np.savetxt(destination, np.column_stack((grid, counts)), fmt="%.6f")
    return [float(grid[0]), float(grid[-1])]

def save_plot(result, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot = result.refinement_result.plot_data
    x, measured, calculated, background = map(np.asarray, (plot.x, plot.y_obs, plot.y_calc, plot.y_bkg))
    fig, (main, residual) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 4.8), gridspec_kw={"height_ratios": [3, 1]}, layout="constrained",
    )
    main.plot(x, measured, color="#24549a", lw=1, label="Observed")
    main.plot(x, calculated, color="#1b7758", lw=1, label="Calculated")
    main.plot(x, background, color="#9b6a45", lw=0.9, label="Background")
    for peaks, color, symbol, label in (
        (getattr(result, "missing_peaks", None), "#c55436", "v", "Unexplained observed peaks"),
        (getattr(result, "extra_peaks", None), "#744fb0", "^", "Calculated peaks not observed"),
    ):
        if peaks is not None and len(peaks):
            locations = np.asarray(peaks).reshape(-1, 2)[:, 0]
            main.scatter(locations, np.full_like(locations, measured.max()), s=19, color=color, marker=symbol, label=label)
    main.set_ylabel("Relative intensity")
    main.legend(fontsize=8, loc="best", frameon=False)
    residual.axhline(0, color="#a1a8a2", lw=0.7)
    residual.plot(x, measured - calculated, color="#59636b", lw=0.8)
    residual.set_xlabel(r"$2\theta$ (degrees)")
    residual.set_ylabel("Obs. - calc.")
    metrics = []
    try:
        lst = result.refinement_result.lst_data
        metrics.append(f"Rwp={lst.rwp:.2f}%")
        if getattr(lst, "rexp", None):
            metrics.append(f"Rexp={lst.rexp:.2f}%")
            if lst.rexp:
                metrics.append(f"GOF={lst.rwp/lst.rexp:.2f}")
    except Exception:
        pass
    if metrics:
        main.set_title("  ".join(metrics))
    fig.savefig(destination, dpi=135)
    plt.close(fig)

def pack_candidate(item, names):
    fit = item.refinement_result
    lst = fit.lst_data
    try:
        raw_w = fit.get_phase_weights() or {}
        weights = {names.get(k, k): round(float(v) * 100.0, 2) for k, v in raw_w.items()}
    except Exception:
        weights = {}
    rwp = float(lst.rwp)
    rexp = float(getattr(lst, "rexp", 0) or 0)
    gof = round(rwp / rexp, 3) if rexp else None
    return {
        "phases": [names.get(name, name) for name in fit.plot_data.structs],
        "phase_ids": list(fit.plot_data.structs),
        "weights_wt_pct": weights,
        "rwp": round(rwp, 3),
        "rexp": round(rexp, 3) if rexp else None,
        "gof": gof,
        "rp": round(float(lst.rp), 3) if getattr(lst, "rp", None) is not None else None,
        "missing": 0 if item.missing_peaks is None else len(item.missing_peaks),
        "extra": 0 if item.extra_peaks is None else len(item.extra_peaks),
    }

def main():
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path.cwd()
    pattern = output / "pattern.xy"
    fit_range = prepare_pattern(config["pattern"], config.get("source_step", 0.05), pattern)
    cifs = [Path(r["path"]) for r in config["references"]]
    import ray
    from dara.search import search_phases
    from dara.utils import process_phase_name
    # 把同名 .ger/.sav 一并拷进 BGMN 工作目录，否则角范围超出 *.geq 表（文献图常到 80–110°）会直接失败
    import shutil as _shutil
    import dara.generate_control_file as _gcf
    import dara.eflech_worker as _efw
    _orig_copy = _gcf.copy_instrument_files
    def _copy_instrument_with_ger(instrument_profile, working_dir):
        name = _orig_copy(instrument_profile, working_dir)
        devices = Path(_gcf.__file__).resolve().parent / "data" / "BGMN-Templates" / "Devices"
        for ext in (".ger", ".sav"):
            src = devices / f"{name}{ext}"
            if src.exists():
                _shutil.copy(src, Path(working_dir) / src.name)
        return name
    _gcf.copy_instrument_files = _copy_instrument_with_ger
    _efw.copy_instrument_files = _copy_instrument_with_ger
    ray.init(num_cpus=2, include_dashboard=False, logging_level="ERROR")
    try:
        # enable_angular_cut 默认会把精修切到约 70% 峰（常见结果是只到 ~50°），这里关闭以便拟合整段描点范围
        wmin, wmax = float(fit_range[0]), float(fit_range[1])
        results = search_phases(
            pattern_path=pattern, phases=cifs,
            max_phases=min(int(config.get("max_phases") or 3), len(cifs)),
            wavelength=config["dara_wavelength"],
            refinement_params={"n_threads": 1, "wmin": wmin, "wmax": wmax},
            express_mode=False,
            enable_angular_cut=False,
        )
        names = {}
        for path, reference in zip(cifs, config["references"]):
            names[path.stem] = reference["name"]
            names[process_phase_name(path.stem)] = reference["name"]
        results = list(results or [])
        # 组合搜索若没留下候选，再对每个 CIF 单独搜一次（避免“已精修却 candidates 为空”）
        if not results and cifs:
            for cif in cifs:
                try:
                    extra = search_phases(
                        pattern_path=pattern, phases=[cif],
                        max_phases=1,
                        wavelength=config["dara_wavelength"],
                        refinement_params={"n_threads": 1, "wmin": wmin, "wmax": wmax},
                        express_mode=False,
                        enable_angular_cut=False,
                    )
                    results.extend(list(extra or []))
                except Exception:
                    continue
        candidates = []
        first_ok = None
        for item in results[:8]:
            try:
                packed = pack_candidate(item, names)
                candidates.append(packed)
                if first_ok is None:
                    first_ok = item
            except Exception:
                continue
        if candidates and first_ok is not None:
            try:
                save_plot(first_ok, output / "fit.png")
            except Exception:
                pass
        report = {
            "trace": config.get("trace"),
            "wavelength_angstrom": config.get("wavelength_angstrom"),
            "dara_wavelength": config.get("dara_wavelength"),
            "fit_range": fit_range,
            "references": [{"name": r["name"], "source": r.get("source"), "cod_id": r.get("cod_id"), "formula": r.get("formula")} for r in config["references"]],
            "candidates": candidates,
        }
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        ray.shutdown()
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        Path("report.json").write_text(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
'''


def write_worker_script(dest: Path) -> Path:
    """写一份不依赖工作台路径的精简拟合脚本，供 dara-venv 调用。"""
    dest.write_text(WORKER_SOURCE, encoding="utf-8")
    return dest


def dara_wavelength_token(wavelength_a: float, radiation: str | None) -> str:
    """DARA 接受 Cu/Co/Mo... 或数值。优先用元素名。"""
    text = (radiation or "").lower()
    for name in ("cu", "co", "mo", "fe", "cr"):
        if re.search(rf"\b{name}\b|{name} ?k|{name}靶", text):
            return name.capitalize() if name != "cu" else "Cu"
    table = {"Cu": 1.5406, "Co": 1.78897, "Mo": 0.71073, "Fe": 1.9373, "Cr": 2.2909}
    best = min(table, key=lambda k: abs(table[k] - float(wavelength_a or 1.5406)))
    if abs(table[best] - float(wavelength_a or 1.5406)) < 0.05:
        return best
    return str(wavelength_a)


def _format_summary(top: dict[str, Any]) -> str:
    weights = top.get("weights_wt_pct") or {}
    parts = []
    for name in top.get("phases") or []:
        if name in weights:
            parts.append(f"{name} {weights[name]:.1f} wt%")
        else:
            parts.append(str(name))
    if not parts and weights:
        parts = [f"{k} {v:.1f} wt%" for k, v in weights.items()]
    text = " + ".join(parts) if parts else "无相名"
    extra = f"Rwp={top.get('rwp')}%"
    if top.get("gof") is not None:
        extra += f", GOF={top.get('gof')}"
    if top.get("rexp") is not None:
        extra += f", Rexp={top.get('rexp')}%"
    return f"{text}（{extra}）"


def fit_xy(
    xy_path: Path,
    references: list[dict[str, Any]],
    out_dir: Path,
    *,
    wavelength_a: float,
    radiation: str | None,
    probe: dict[str, Any] | None = None,
    timeout: int = 900,
) -> dict[str, Any]:
    """对一条 .xy 调用 DARA。环境不可用时返回 unavailable，不抛异常。"""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = probe or probe_dara()
    if not probe.get("available"):
        (out_dir / "status.json").write_text(
            json.dumps({"state": "unavailable", "hint": probe.get("hint"), "probe": probe}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {"status": "unavailable", "message": probe.get("error") or "dara 不可用", "hint": probe.get("hint")}
    if not references:
        return {"status": "skipped", "message": "没有可用的 CIF 参考相"}

    worker = write_worker_script(out_dir / "dara_worker_run.py").resolve()
    abs_refs = []
    for ref in references:
        item = dict(ref)
        item["path"] = str(Path(item["path"]).resolve())
        abs_refs.append(item)
    config = {
        "pattern": str(Path(xy_path).resolve()),
        "source_step": 0.05,
        "trace": Path(xy_path).stem,
        "wavelength_angstrom": wavelength_a,
        "dara_wavelength": dara_wavelength_token(wavelength_a, radiation),
        "max_phases": min(3, len(abs_refs)),
        "references": abs_refs,
    }
    spec = out_dir / "input.json"
    spec.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.update(MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log_path = out_dir / "dara.log"
    report_path = out_dir / "report.json"
    python_exe = probe["python"]

    def _load_finished_report() -> dict[str, Any] | None:
        if not report_path.is_file():
            return None
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if report.get("error") or not (report.get("candidates") or []):
            return None
        fit_png = out_dir / "fit.png"
        return {
            "status": "ok",
            "summary": _format_summary((report.get("candidates") or [])[0]),
            "report": report,
            "fit_png": str(fit_png) if fit_png.is_file() else "",
            "log": str(log_path),
        }

    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                # 不要 resolve() Python 解释器：venv 里的 python 是指向系统 Python 的符号链接，
                # resolve 后会丢掉 venv 的 site-packages（例如 ray）。
                [str(Path(python_exe).expanduser().absolute()), "-X", "utf8", str(worker), str(Path(spec).resolve())],
                cwd=str(out_dir),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        code = proc.returncode
    except subprocess.TimeoutExpired:
        # 报告和 fit.png 有时已写完，只是 ray.shutdown 收尾超时
        finished = _load_finished_report()
        if finished:
            finished["message"] = f"DARA 在 {timeout} 秒超时前已写出报告，按成功结果保留"
            return finished
        return {"status": "timeout", "message": f"DARA 拟合超过 {timeout} 秒，已放弃", "log": str(log_path)}
    except Exception as exc:
        return {"status": "failed", "message": str(exc), "log": str(log_path)}

    if code != 0 or not report_path.is_file():
        tail = ""
        if log_path.is_file():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        return {
            "status": "failed",
            "message": f"DARA 退出码 {code}，或未写出 report.json",
            "log": str(log_path),
            "log_tail": tail,
        }
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "failed", "message": f"无法读取 report.json：{exc}"}
    if report.get("error"):
        return {"status": "failed", "message": report["error"], "report": report}
    fit_png = out_dir / "fit.png"
    cands = report.get("candidates") or []
    summary = "无候选"
    if cands:
        summary = _format_summary(cands[0])
    return {
        "status": "ok",
        "summary": summary,
        "report": report,
        "fit_png": str(fit_png) if fit_png.is_file() else "",
        "log": str(log_path),
    }


def interpret_prompt(report: dict[str, Any], caption: str, peak_table: str) -> str:
    return "\n".join(
        [
            "下面是对数字化 XRD 曲线做 DARA/BGMN 全谱拟合的结果。请用中文解读：",
            "拟合是否支持图注/峰表中的物相，Rwp 是否可接受，漏峰/多峰可能意味着什么，不确定性在哪里。",
            "务必说明：数字化曲线没有真实计数统计，Rwp/GOF 只能作相对参考，不能当成仪器数据的精修质量。",
            "只输出 JSON：",
            json.dumps(
                {
                    "supported": True,
                    "summary": "一句话结论",
                    "details": ["要点"],
                    "uncertainty": "必须填写",
                },
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "图注：",
            caption or "（无）",
            "",
            "峰表：",
            peak_table or "（无）",
            "",
            "拟合报告：",
            json.dumps(report, ensure_ascii=False, indent=2)[:8000],
        ]
    )
