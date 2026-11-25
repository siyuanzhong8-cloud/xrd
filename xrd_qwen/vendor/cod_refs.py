# Vendored from XRD-Workbench-Dev for the pdf_xrd pipeline. Logic unchanged.
import json
import os
import re
from decimal import Decimal
from pathlib import Path
from hashlib import sha256
from threading import BoundedSemaphore
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

COD_SITES = ("https://qiserver.ugr.es/cod", "https://www.crystallography.net/cod")
COD_LINK = "https://www.crystallography.net/cod/"
CACHE = Path(__file__).resolve().parent / "reference-cache"
SEARCH_SLOTS = BoundedSemaphore(2)
TOKEN = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)?")


class CodUnavailable(RuntimeError):
    pass


def hill_formula(value):
    compact = "".join(value.split())
    tokens = list(TOKEN.finditer(compact))
    if not compact or len(compact) > 64 or "".join(match[0] for match in tokens) != compact:
        raise ValueError("请输入化学式，例如 TiO2、ZnO 或 NaCl。复杂组成也可直接填写 COD 编号。")
    counts = {}
    for token in tokens:
        element, text = token.groups()
        amount = Decimal(text or "1")
        if not 0 < amount <= 10000:
            raise ValueError("请检查化学式中的原子数。")
        counts[element] = counts.get(element, Decimal(0)) + amount
    priority = [key for key in ("C", "H") if key in counts] if "C" in counts else []
    ordered = priority + sorted(set(counts) - set(priority))
    return " ".join(
        element + (format(counts[element].normalize(), "f") if counts[element] != 1 else "")
        for element in ordered
    )


def fetch(route, limit, validate):
    for site in COD_SITES:
        try:
            request = Request(site + route, headers={"User-Agent": "XRD-Workbench/1.0"})
            with urlopen(request, timeout=6) as response:
                payload = response.read(limit + 1)
            if len(payload) > limit:
                raise ValueError("COD 返回结果过多，请换成更具体的化学式或使用已知编号。")
            return validate(payload)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, UnicodeError):
            continue
    raise CodUnavailable("暂时无法读取 COD。可以改用已下载的 CIF，或稍后再试。")


def search(formula):
    normalized = hill_formula(formula)
    if not SEARCH_SLOTS.acquire(blocking=False):
        raise CodUnavailable("COD 查询繁忙，请稍后重试。")
    try:
        route = "/result?" + urlencode({"formula": normalized, "format": "json"})
        cache = CACHE / f"query-{sha256(normalized.encode()).hexdigest()}.json"
        try:
            rows = fetch(route, 4 * 1024 * 1024, json.loads)
        except CodUnavailable:
            if not cache.is_file():
                raise
            rows = json.loads(cache.read_text(encoding="utf-8"))
        else:
            if isinstance(rows, list) and rows:
                CACHE.mkdir(exist_ok=True)
                temporary = cache.with_name(f".{uuid4().hex}.tmp")
                temporary.write_text(json.dumps(rows), encoding="utf-8")
                os.replace(temporary, cache)
    finally:
        SEARCH_SLOTS.release()
    if not isinstance(rows, list):
        raise CodUnavailable("COD 暂时没有返回可读取的查询结果。")
    by_group = {}
    others = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        entry_id = str(row.get("file", ""))
        if not re.fullmatch(r"[0-9]{7}", entry_id):
            continue
        group = str(row.get("sg") or "未标注")
        entry = {
            "id": entry_id, "formula": str(row.get("calcformula") or row.get("formula") or normalized).strip(" -"),
            "name": str(row.get("commonname") or row.get("chemname") or "").strip()[:100],
            "title": str(row.get("title") or "").strip()[:210],
            "group": group, "year": str(row.get("year") or ""),
        }
        if group in by_group:
            others.append(entry)
        else:
            by_group[group] = entry
    entries = [*by_group.values(), *others]
    return {"formula": normalized, "total": len(entries), "group_count": len(by_group), "results": entries[:24]}


def cif(entry_id):
    if not re.fullmatch(r"[0-9]{7}", entry_id):
        raise ValueError("COD 编号应是 7 位数字。")
    path = CACHE / f"{entry_id}.cif"
    if path.is_file():
        return path

    def validate(payload):
        if not re.search(rb"(?im)^\s*data_\S+", payload) or b"_atom_site_" not in payload.lower():
            raise UnicodeError("Not a CIF with coordinates")
        return payload

    try:
        payload = fetch(f"/{entry_id}.cif", 512 * 1024, validate)
    except CodUnavailable as exc:
        raise CodUnavailable(f"未能下载 COD {entry_id}。请核对编号，或改用自己下载的 CIF。") from exc
    CACHE.mkdir(exist_ok=True)
    temporary = CACHE / f".{entry_id}-{uuid4().hex}.tmp"
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return path
