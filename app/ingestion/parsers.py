"""多源知识解析器：

支持 —— Markdown/TXT 政策文档（含 Front-Matter 元数据）、
CSV 时效表、CSV/Excel 网点表、JSONL 结构化规则。

统一输出： (base_meta: dict, full_text: str)
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

from app.ingestion.metadata import ALL_ROLES, SOURCE_TYPES

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _normalize_fm_value(key: str, value) -> object:
    value = str(value).strip()
    if key == "roles":
        roles = [r.strip() for r in value.replace("，", ",").split(",") if r.strip()]
        valid = [r for r in roles if r in ALL_ROLES]
        return valid or ["customer"]
    return value


def parse_markdown(path: Path) -> tuple[dict, str]:
    """解析带 Front-Matter 的 Markdown 政策/SOP 文档。"""
    raw = path.read_text(encoding="utf-8")
    base: dict = {
        "doc_id": path.stem,
        "source": "policy",
        "title": path.stem,
        "category": "规则",
        "carrier": "ALL",
        "version": "v1.0",
        "effective_date": "",
        "expire_date": "",
        "roles": ["customer"],
        "keywords": [],
    }
    m = _FRONT_MATTER_RE.match(raw)
    if m:
        fm_text, body = m.group(1), raw[m.end():]
        for line in fm_text.splitlines():
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key, val = key.strip().lower(), val.strip().strip("\"'")
            if key in base:
                base[key] = _normalize_fm_value(key, val)
    else:
        body = raw
        # 无 Front-Matter 时从文件名推断来源
        name = path.stem.lower()
        for st in SOURCE_TYPES:
            if st in name:
                base["source"] = st
                break
    if base["source"] not in SOURCE_TYPES:
        base["source"] = "policy"
    return base, body.strip()


def _csv_rows(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8-sig")
    sample = content[:2048]
    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t") if "," in sample or "\t" in sample else csv.excel
    reader = csv.DictReader(io.StringIO(content), dialect=dialect)
    return [dict(r) for r in reader]


def parse_timeliness_table(path: Path) -> tuple[dict, str]:
    """解析时效表 CSV：线路,承运商,标准时效(天),快运时效(天),备注..."""
    rows = _csv_rows(path)
    lines = ["# 时效表", ""]
    for r in rows:
        lane = r.get("线路") or r.get("lane") or ""
        lines.append(f"## {lane}")
        parts = [f"{k}: {v.strip()}" for k, v in r.items() if v and str(v).strip()]
        lines.append("；".join(parts))
        lines.append("")
    base = {
        "doc_id": path.stem, "source": "timeliness", "title": "时效表",
        "category": "时效", "carrier": "ALL", "version": "v1.0",
        "effective_date": "", "expire_date": "",
        "roles": ["customer", "customer_service", "operations", "admin"],
        "keywords": ["时效", "运输时间"],
    }
    return base, "\n".join(lines)


def parse_network_table(path: Path) -> tuple[dict, str]:
    """解析网点表 CSV：网点名称,省市,地址,电话,营业时间,是否支持揽收..."""
    rows = _csv_rows(path)
    lines = ["# 网点表", ""]
    for r in rows:
        name = r.get("网点名称") or r.get("name") or ""
        lines.append(f"## {name}")
        parts = [f"{k}: {v.strip()}" for k, v in r.items() if v and str(v).strip()]
        lines.append("；".join(parts))
        lines.append("")
    base = {
        "doc_id": path.stem, "source": "network", "title": "网点表",
        "category": "网点", "carrier": "ALL", "version": "v1.0",
        "effective_date": "", "expire_date": "",
        "roles": ["customer", "customer_service", "operations", "admin"],
        "keywords": ["网点", "营业部", "地址"],
    }
    return base, "\n".join(lines)


def parse_jsonl_rules(path: Path) -> list[tuple[dict, str]]:
    """解析 JSONL 结构化规则（如承运商禁运品/赔偿规则），每行一条独立规则文档。"""
    out: list[tuple[dict, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        base = {
            "doc_id": obj.get("doc_id") or f"{path.stem}-{len(out)}",
            "source": obj.get("source", "policy"),
            "title": obj.get("title", ""),
            "category": obj.get("category", "规则"),
            "carrier": obj.get("carrier", "ALL"),
            "version": obj.get("version", "v1.0"),
            "effective_date": obj.get("effective_date", ""),
            "expire_date": obj.get("expire_date", ""),
            "roles": obj.get("roles", ["customer"]),
            "keywords": obj.get("keywords", []),
        }
        out.append((base, obj.get("text", "")))
    return out


def parse_any(path: Path) -> list[tuple[dict, str]]:
    """按扩展名/文件名自动路由到对应解析器。"""
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in (".md", ".markdown", ".txt"):
        return [parse_markdown(path)]
    if suffix == ".jsonl":
        return parse_jsonl_rules(path)
    if suffix == ".csv":
        if "timeliness" in name or "时效" in name:
            return [parse_timeliness_table(path)]
        if "network" in name or "网点" in name:
            return [parse_network_table(path)]
        return [parse_timeliness_table(path)]  # 默认按表处理
    if suffix in (".xlsx", ".xls"):
        return _parse_excel(path)
    return []


def _parse_excel(path: Path) -> list[tuple[dict, str]]:
    try:
        import openpyxl
    except ImportError:
        return []
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    results = []
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        header = [str(c or "").strip() for c in rows[0]]
        lines = [f"# {ws.title}", ""]
        for row in rows[1:]:
            parts = [f"{h}: {v}" for h, v in zip(header, row) if v is not None and str(v).strip()]
            if parts:
                lines.append("；".join(parts))
        base = {
            "doc_id": f"{path.stem}-{ws.title}", "source": "timeliness",
            "title": f"{path.stem}-{ws.title}", "category": "表格", "carrier": "ALL",
            "version": "v1.0", "effective_date": "", "expire_date": "",
            "roles": ["customer", "customer_service", "operations", "admin"],
            "keywords": [],
        }
        results.append((base, "\n".join(lines)))
    return results
