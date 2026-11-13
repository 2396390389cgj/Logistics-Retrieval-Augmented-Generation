"""问答评测：Hit@5、MRR、回答关键词命中、澄清率。

用法：python -m app.eval.evaluate [--limit 50]
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from config.settings import settings
from app.retrieval.rewrite import rewrite_rule_based
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector import VectorRetriever
from app.generation.answer import AnswerGenerator

EVAL_PATH = Path(__file__).parent / "evalset.jsonl"


async def run(limit: int | None = None) -> dict:
    # 装配离线评测环境
    import sys
    sys.path.insert(0, str(settings.BASE_DIR))
    meta_path = settings.INDEX_DIR / "chunks.meta"
    if not meta_path.exists():
        raise SystemExit("索引不存在，请先运行 scripts/build_index.py")
    with open(meta_path, encoding="utf-8") as f:
        chunks = json.load(f)
    bm25 = BM25Retriever()
    bm25.build(chunks)
    vector = VectorRetriever(settings.INDEX_DIR, use_local=settings.USE_LOCAL_EMBEDDING,
                             model_name=settings.EMBEDDING_MODEL)
    if not vector.load():
        vector.build(chunks)
    retriever = HybridRetriever(bm25, vector, rrf_k=settings.RRF_K,
                                half_life_days=settings.TIME_DECAY_HALFLIFE_DAYS)
    reranker = Reranker(settings.RERANKER_MODEL, fallback_encoder=vector)
    generator = AnswerGenerator()

    cases = [json.loads(l) for l in EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        cases = cases[:limit]

    hit5 = mrr = kw_hits = 0
    clarified = 0
    details = []
    for case in cases:
        query = rewrite_rule_based(case["query"], case.get("history", []))
        hits = retriever.retrieve(query, role=case.get("role", "customer_service"),
                                  topk=settings.BM25_TOPK)
        hits = reranker.rerank(query, hits, topk=settings.RERANK_TOPK)
        top5_ids = {h.chunk_id for h in hits[:5]}
        top5_titles = {h.meta.get("title", "") for h in hits[:5]}

        # 期望命中：chunk_id 或文档标题
        expected = set(case.get("expected_chunks", [])) | set(case.get("expected_titles", []))
        hit_rank = 0
        for i, h in enumerate(hits[:5], 1):
            if h.chunk_id in expected or any(t in h.meta.get("title", "") for t in expected):
                hit_rank = i
                break
        if hit_rank:
            hit5 += 1
            mrr += 1.0 / hit_rank

        result = await generator.generate(query, hits, [])
        answer_kw_ok = all(kw in result.answer for kw in case.get("answer_keywords", []))
        kw_hits += answer_kw_ok
        clarified += result.clarified
        details.append({"query": case["query"], "hit_rank": hit_rank,
                        "answer_keywords_ok": answer_kw_ok,
                        "clarified": result.clarified,
                        "top_titles": list(top5_titles)[:3]})

    n = len(cases)
    report = {
        "cases": n,
        "hit_at_5": round(hit5 / n, 3) if n else 0,
        "mrr@5": round(mrr / n, 3) if n else 0,
        "answer_keyword_accuracy": round(kw_hits / n, 3) if n else 0,
        "clarify_rate": round(clarified / n, 3) if n else 0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = settings.BASE_DIR / "data" / "eval_report.json"
    out.write_text(json.dumps({"report": report, "details": details},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写入 {out}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    asyncio.run(run(parser.parse_args().limit))
