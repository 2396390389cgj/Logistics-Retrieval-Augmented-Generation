"""混合检索：BM25 + 向量双路召回 → RRF 融合 → 权限/版本/时间衰减过滤 → 重排。"""
from __future__ import annotations

import logging
from datetime import date

from app.ingestion.metadata import is_effective, parse_date
from app.retrieval.bm25 import BM25Retriever, Hit
from app.retrieval.vector import VectorRetriever

logger = logging.getLogger(__name__)


def rrf_fuse(bm25_hits: list[Hit], vector_hits: list[Hit], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion：按排名倒数融合两路召回。"""
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for hits in (bm25_hits, vector_hits):
        for rank, h in enumerate(hits):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            if h.chunk_id not in best:
                best[h.chunk_id] = h
    fused = []
    for cid, s in scores.items():
        h = best[cid]
        h.score = s
        fused.append(h)
    return sorted(fused, key=lambda x: -x.score)


def time_decay_boost(hit: Hit, half_life_days: float, today: date | None = None) -> float:
    """时间衰减：版本越新权重越高，衰减系数 ∈ (0.5, 1]。

    承运商规则更新频繁，旧版本知识应自然下沉。
    """
    eff = parse_date(hit.meta.get("effective_date", ""))
    if not eff:
        return 1.0
    today = today or date.today()
    days = max((today - date.fromisoformat(eff)).days, 0)
    factor = math_exp2(-days / max(half_life_days, 1.0))
    # 衰减下限 0.5，避免完全排除仍有效的旧规则
    return max(0.5 + 0.5 * factor, 0.5)


def math_exp2(x: float) -> float:
    return 2.0 ** x


def role_allowed(hit_meta: dict, role: str, role_levels: dict[str, int]) -> bool:
    """按角色的检索权限过滤：用户角色等级 ≥ 文档要求的最低可见角色。"""
    doc_roles = hit_meta.get("roles") or ["customer"]
    if role in doc_roles:
        return True
    # 等级制：admin 可见全部；operations 可见 customer_service 及以下文档
    user_level = role_levels.get(role, 0)
    allowed_levels = {r: role_levels.get(r, 0) for r in doc_roles}
    return any(user_level >= lv for lv in allowed_levels.values())


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever,
                 rrf_k: int = 60, half_life_days: float = 180.0,
                 role_levels: dict[str, int] | None = None) -> None:
        self.bm25 = bm25
        self.vector = vector
        self.rrf_k = rrf_k
        self.half_life_days = half_life_days
        self.role_levels = role_levels or {"customer": 0, "customer_service": 1,
                                           "operations": 2, "admin": 3}

    def retrieve(self, query: str, role: str, topk: int = 50) -> list[Hit]:
        bm25_hits = self.bm25.search(query, topk)
        vec_hits = self.vector.search(query, topk)
        fused = rrf_fuse(bm25_hits, vec_hits, k=self.rrf_k)

        # 权限过滤 + 版本有效性过滤 + 时间衰减加权
        today = date.today()
        filtered: list[Hit] = []
        for h in fused:
            if not role_allowed(h.meta, role, self.role_levels):
                continue
            if not is_effective(_SimpleMeta(h.meta), today):
                continue
            h.score *= time_decay_boost(h, self.half_life_days, today)
            filtered.append(h)
        return filtered


class _SimpleMeta:
    """把 dict 元数据适配为 is_effective 需要的属性访问对象。"""

    def __init__(self, d: dict) -> None:
        self.effective_date = d.get("effective_date", "")
        self.expire_date = d.get("expire_date", "")
