"""摄取管线：data/raw → 解析 → 层级切片 → 索引存储（BM25/FAISS/ES/MySQL）。"""
from __future__ import annotations

import logging
from pathlib import Path

from app.ingestion.chunker import Chunk, hierarchical_chunk
from app.ingestion.parsers import parse_any

logger = logging.getLogger(__name__)


def load_chunks(raw_dir: Path) -> list[Chunk]:
    """解析 raw 目录全部文档并切片。"""
    all_chunks: list[Chunk] = []
    files = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in
                   {".md", ".markdown", ".txt", ".csv", ".jsonl", ".xlsx", ".xls"})
    for path in files:
        try:
            for base_meta, full_text in parse_any(path):
                if not full_text.strip():
                    continue
                chunks = hierarchical_chunk(base_meta, full_text, base_meta["doc_id"])
                all_chunks.extend(chunks)
                logger.info("解析 %s → %d 个切片", path.name, len(chunks))
        except Exception:  # noqa: BLE001
            logger.exception("解析失败：%s", path)
    return all_chunks
