"""FastAPI 入口。启动：uvicorn app.main:app --reload --port 8000"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from config.settings import settings
from app.api.routes import router
from app.service import service

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="物流客服 RAG 问答系统",
    description="知识问答（RAG）+ 实时业务查询（TMS/OMS）双通道融合生成",
    version="1.0.0",
)
app.include_router(router)


@app.on_event("startup")
async def startup() -> None:
    service.load_index()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_ready": service.ready,
        "llm": "enabled" if service.llm.available else "fallback(抽取式兜底)",
        "embedding": "local" if settings.USE_LOCAL_EMBEDDING else "hash(离线)",
    }
