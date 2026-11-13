"""重排序：bge-reranker 交叉编码精排；未安装/无模型时回退 embedding 余弦 + 词面匹配。"""
from __future__ import annotations

import logging

from app.retrieval.bm25 import Hit
from app.retrieval.tokenize import tokenize

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large",
                 fallback_encoder=None) -> None:
        self._model = None
        self._fallback = fallback_encoder  # 提供 encode(texts)->np.ndarray 的对象
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name, max_length=512)
            logger.info("已加载 reranker %s", model_name)
        except Exception as e:  # noqa: BLE001
            logger.warning("bge-reranker 不可用，回退词面+向量精排：%s", e)

    def rerank(self, query: str, hits: list[Hit], topk: int = 6) -> list[Hit]:
        if not hits:
            return []
        if self._model is not None:
            pairs = [(query, h.text) for h in hits]
            scores = self._model.predict(pairs)
            for h, s in zip(hits, scores):
                h.score = float(s)
        else:
            hits = self._lexicon_rerank(query, hits)
        return sorted(hits, key=lambda x: -x.score)[:topk]

    def _lexicon_rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """回退精排：向量相似度为主 + 查询词覆盖率加成。"""
        q_tokens = set(tokenize(query))
        q_vec = None
        if self._fallback is not None:
            try:
                q_vec = self._fallback.encode([query])[0]
            except Exception:  # noqa: BLE001
                q_vec = None
        for h in hits:
            h_tokens = set(tokenize(h.text))
            cover = len(q_tokens & h_tokens) / max(len(q_tokens), 1)
            score = 0.6 * cover
            if q_vec is not None:
                try:
                    import numpy as np
                    hv = self._fallback.encode([h.text])[0]
                    score += 0.4 * float(np.dot(q_vec, hv))
                except Exception:  # noqa: BLE001
                    pass
            h.score = score
        return hits
