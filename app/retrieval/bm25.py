"""BM25 稀疏召回：优先 Elasticsearch，缺失时回退内置 rank_bm25。"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from app.retrieval.tokenize import tokenize

logger = logging.getLogger(__name__)


@dataclass
class Hit:
    chunk_id: str
    score: float
    source: str = "bm25"  # bm25 / vector / tms / oms
    text: str = ""
    meta: dict = field(default_factory=dict)


class BM25Retriever:
    """内存 BM25（rank_bm25 回退实现），与 ES 检索器接口一致。"""

    def __init__(self) -> None:
        self._chunks: list[dict] = []
        self._bm25 = None

    def build(self, chunks: list[dict]) -> None:
        """chunks: [{chunk_id, text, meta}]"""
        self._chunks = chunks
        try:
            from rank_bm25 import BM25Okapi
            corpus = [tokenize(c["text"]) for c in chunks]
            self._bm25 = BM25Okapi(corpus)
        except ImportError:
            self._bm25 = _SimpleBM25(corpus)

    def search(self, query: str, topk: int = 50) -> list[Hit]:
        if not self._chunks or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:topk]
        hits = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            c = self._chunks[i]
            hits.append(Hit(chunk_id=c["chunk_id"], score=float(scores[i]),
                            text=c["text"], meta=c.get("meta", {})))
        return hits


class _SimpleBM25:
    """rank_bm25 未安装时的最小 BM25 实现（k1=1.5, b=0.75）。"""

    def __init__(self, corpus: list[list[str]]) -> None:
        self.k1, self.b = 1.5, 0.75
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.N, 1)
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        df: dict[str, int] = {}
        for doc in corpus:
            freq: dict[str, int] = {}
            for t in doc:
                freq[t] = freq.get(t, 0) + 1
            self.doc_freqs.append(freq)
            for t in freq:
                df[t] = df.get(t, 0) + 1
        for t, n in df.items():
            self.idf[t] = math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.N
        for i, freq in enumerate(self.doc_freqs):
            dl = sum(freq.values())
            for t in query:
                if t not in freq:
                    continue
                tf = freq[t]
                scores[i] += self.idf.get(t, 0.0) * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return scores
