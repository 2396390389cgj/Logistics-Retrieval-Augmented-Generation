"""索引构建脚本：解析 data/raw → 层级切片 → BM25 + FAISS 索引落盘。

用法：python scripts/build_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from app.ingestion.pipeline import load_chunks  # noqa: E402
from app.retrieval.bm25 import BM25Retriever  # noqa: E402
from app.retrieval.vector import VectorRetriever  # noqa: E402


def main() -> None:
    chunks = load_chunks(settings.RAW_DIR)
    if not chunks:
        print("!! 未解析到任何切片，请检查 data/raw 目录")
        sys.exit(1)

    payload = [{"chunk_id": c.meta.chunk_id, "text": c.text,
                "meta": c.meta.to_dict()} for c in chunks]

    bm25 = BM25Retriever()
    vector = VectorRetriever(settings.INDEX_DIR,
                             use_local=settings.USE_LOCAL_EMBEDDING,
                             model_name=settings.EMBEDDING_MODEL)
    bm25.build(payload)
    vector.build(payload)

    # 落盘切片元信息（服务启动时加载重建 BM25 + 加载 FAISS）
    settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(settings.INDEX_DIR / "chunks.meta", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    # 简单统计
    by_level: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for c in chunks:
        by_level[c.meta.level] = by_level.get(c.meta.level, 0) + 1
        by_source[c.meta.source] = by_source.get(c.meta.source, 0) + 1
    print(f"索引构建完成：共 {len(payload)} 个切片 → {settings.INDEX_DIR}")
    print(f"  层级分布: {by_level}")
    print(f"  来源分布: {by_source}")


if __name__ == "__main__":
    main()
