"""RAG 服务全局单例：索引加载、检索器、生成器装配。"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from config.settings import settings
from app.core.llm import LLMClient
from app.generation.answer import AnswerGenerator
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.vector import VectorRetriever
from app.storage.stores import CacheStore, RelationalStore

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self) -> None:
        self.llm = LLMClient()
        self.bm25 = BM25Retriever()
        self.vector = VectorRetriever(
            settings.INDEX_DIR, use_local=settings.USE_LOCAL_EMBEDDING,
            model_name=settings.EMBEDDING_MODEL)
        self.reranker = Reranker(settings.RERANKER_MODEL, fallback_encoder=self.vector)
        self.retriever = HybridRetriever(
            self.bm25, self.vector, rrf_k=settings.RRF_K,
            half_life_days=settings.TIME_DECAY_HALFLIFE_DAYS)
        self.generator = AnswerGenerator(self.llm)
        self.db = RelationalStore()
        self.cache = CacheStore()
        self.ready = False

    def load_index(self) -> bool:
        """加载已构建的索引（由 scripts/build_index.py 生成）。"""
        meta_path = Path(settings.INDEX_DIR) / "chunks.meta"
        if not meta_path.exists():
            logger.warning("索引不存在，请先运行 scripts/build_index.py")
            return False
        with open(meta_path, encoding="utf-8") as f:
            chunks = json.load(f)
        self.bm25.build(chunks)
        if not self.vector.load():
            self.vector.build(chunks)
        self.ready = True
        logger.info("索引加载完成，共 %d 个切片", len(chunks))
        return True

    def build_index_from_raw(self) -> int:
        from app.ingestion.pipeline import load_chunks
        chunks = load_chunks(settings.RAW_DIR)
        payload = [{"chunk_id": c.meta.chunk_id, "text": c.text,
                    "meta": c.meta.to_dict()} for c in chunks]
        self.bm25.build(payload)
        self.vector.build(payload)
        self.ready = True
        return len(payload)


service = RAGService()
