"""API 路由：问答（JSON / SSE 流式）、运单查询、健康检查。"""
from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import settings
from app.service import service
from app.tools.agent import LogisticsAgent
from app.tools.tms_client import extract_waybill_no
from app.retrieval.rewrite import rewrite_query

router = APIRouter(prefix="/api")
agent = LogisticsAgent()


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    role: str = Field("customer_service",
                      description="customer/customer_service/operations/admin")
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    history: list[dict] = Field(default_factory=list,
                                description="[{'role':'user','content':'...'}, ...]")


def _validate_role(role: str) -> str:
    if role not in settings.ROLE_LEVELS:
        raise HTTPException(400, f"非法角色：{role}")
    return role


async def _pipeline(req: ChatRequest):
    """完整 RAG + Agent 双通道管线，返回 (query, hits, evidences, result)。"""
    role = _validate_role(req.role)
    if not service.ready and not service.load_index():
        raise HTTPException(503, "索引未就绪，请先运行 scripts/build_index.py")

    t0 = time.time()
    # 1) 多轮查询改写
    query = await rewrite_query(req.query, req.history, llm=service.llm)
    # 2) 实时通道
    evidences = await agent.detect_and_execute(query)
    # 3) 知识通道（命中运单号时知识通道仍兜底运费/时效规则）
    hits = service.retriever.retrieve(query, role=role, topk=settings.BM25_TOPK)
    hits = service.reranker.rerank(query, hits, topk=settings.RERANK_TOPK)
    # 4) 融合生成
    result = await service.generator.generate(query, hits, evidences)
    result.latency_ms = int((time.time() - t0) * 1000)  # type: ignore[attr-defined]
    return query, hits, evidences, result


@router.post("/chat")
async def chat(req: ChatRequest):
    """JSON 问答。"""
    query, hits, evidences, result = await _pipeline(req)
    service.db.log_chat(req.session_id, req.role, req.query, result.answer,
                        result.references, result.confidence, result.latency_ms)
    return {
        "session_id": req.session_id,
        "rewritten_query": query,
        "answer": result.answer,
        "references": result.references,
        "realtime_evidences": [e["summary"] for e in evidences],
        "confidence": round(result.confidence, 3),
        "clarified": result.clarified,
        "latency_ms": result.latency_ms,
    }


@router.get("/chat/stream")
async def chat_stream(q: str, role: str = "customer_service",
                      history: str = "[]"):
    """SSE 流式问答：event=data 增量文本，event=meta 收尾元信息。"""
    import ast

    req = ChatRequest(query=q, role=role,
                      history=ast.literal_eval(history) if history else [])
    _validate_role(req.role)
    if not service.ready and not service.load_index():
        raise HTTPException(503, "索引未就绪，请先运行 scripts/build_index.py")

    t0 = time.time()
    query = await rewrite_query(req.query, req.history, llm=service.llm)
    evidences = await agent.detect_and_execute(query)
    hits = service.retriever.retrieve(query, role=req.role, topk=settings.BM25_TOPK)
    hits = service.reranker.rerank(query, hits, topk=settings.RERANK_TOPK)

    async def sse():
        yield f"event: rewritten\ndata: {json.dumps({'query': query}, ensure_ascii=False)}\n\n"
        for e in evidences:
            yield (f"event: evidence\ndata: "
                   f"{json.dumps({'summary': e['summary']}, ensure_ascii=False)}\n\n")
        full = []
        async for delta in service.generator.generate_stream(query, hits, evidences):
            full.append(delta)
            yield f"event: data\ndata: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
        answer = "".join(full)
        latency = int((time.time() - t0) * 1000)
        service.db.log_chat(req.session_id, req.role, req.query, answer,
                            [{"chunk_id": h.chunk_id} for h in hits[:5]], 0.0, latency)
        meta = {"references": [{"title": h.meta.get("title", ""),
                                "chunk_id": h.chunk_id} for h in hits[:5]],
                "latency_ms": latency}
        yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@router.get("/track/{waybill_no}")
async def track(waybill_no: str):
    """实时运单查询（TMS/OMS 双接口）。"""
    if not extract_waybill_no(waybill_no):
        raise HTTPException(400, "运单号格式非法（示例：SF1234567890）")
    info = await agent.tms.get_waybill(waybill_no)
    if not info:
        raise HTTPException(404, "未找到该运单")
    order = await agent.oms.get_order(waybill_no)
    return {"tms": info, "oms": order}
