"""元数据规范设计：所有切片必须携带的标准化字段。

职责覆盖：多源知识的元数据规范设计 + 承运商规则版本化字段。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import date

# 支持的来源类型
SOURCE_TYPES = {"policy", "sop", "claim", "sla", "network", "timeliness"}

# 角色标签：文档可被哪些角色检索
ALL_ROLES = ["customer", "customer_service", "operations", "admin"]


def stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


@dataclass
class ChunkMeta:
    """一个切片的标准元数据（元数据规范）。"""

    doc_id: str                 # 源文档 ID
    chunk_id: str               # 切片唯一 ID
    source: str                 # policy/sop/claim/sla/network/timeliness
    title: str                  # 文档标题
    category: str               # 业务类目：时效/规则/SOP/理赔/网点/费用
    carrier: str = "ALL"        # 承运商：SF/JD/ZTO/DEBANG/ALL（通用）
    version: str = "v1.0"       # 版本号（承运商规则版本化）
    effective_date: str = ""    # 生效日期 YYYY-MM-DD
    expire_date: str = ""       # 失效日期，空为长期有效
    roles: list[str] = field(default_factory=lambda: ["customer"])  # 可见角色
    keywords: list[str] = field(default_factory=list)
    level: str = "leaf"         # 层级切片：h1/h2/leaf
    parent_id: str = ""         # 父级摘要切片 ID
    order: int = 0              # 在文档内的序号

    def to_dict(self) -> dict:
        return asdict(self)


def parse_date(s: str) -> str:
    """宽松解析日期，返回 YYYY-MM-DD 或空串。"""
    s = (s or "").strip()
    m = re.search(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            return ""
    return ""


def is_effective(meta: ChunkMeta, today: date | None = None) -> bool:
    """版本有效性判断：生效日期已到且未过期。"""
    today = today or date.today()
    eff = parse_date(meta.effective_date)
    exp = parse_date(meta.expire_date)
    if eff and date.fromisoformat(eff) > today:
        return False
    if exp and date.fromisoformat(exp) < today:
        return False
    return True
