"""全局配置：所有参数集中管理，支持 .env 覆盖与无依赖降级。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


class Settings:
    # 项目路径
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    INDEX_DIR: Path = DATA_DIR / "index"

    # LLM
    OPENAI_API_KEY: str = _get("OPENAI_API_KEY")
    OPENAI_BASE_URL: str = _get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL: str = _get("LLM_MODEL", "gpt-4o-mini")

    # Embedding / Rerank
    EMBEDDING_MODEL: str = _get("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    RERANKER_MODEL: str = _get("RERANKER_MODEL", "BAAI/bge-reranker-large")
    USE_LOCAL_EMBEDDING: bool = _get("USE_LOCAL_EMBEDDING", "false").lower() == "true"

    # 存储
    MYSQL_URL: str = _get("MYSQL_URL")
    REDIS_URL: str = _get("REDIS_URL")
    ES_URL: str = _get("ES_URL")
    ES_INDEX: str = _get("ES_INDEX", "logistics_rag_chunks")

    # TMS/OMS
    TMS_BASE_URL: str = _get("TMS_BASE_URL")
    OMS_BASE_URL: str = _get("OMS_BASE_URL")

    # 检索参数
    BM25_TOPK: int = int(_get("BM25_TOPK", "50") or 50)
    VECTOR_TOPK: int = int(_get("VECTOR_TOPK", "50") or 50)
    RRF_K: int = int(_get("RRF_K", "60") or 60)
    RERANK_TOPK: int = int(_get("RERANK_TOPK", "6") or 6)
    TIME_DECAY_HALFLIFE_DAYS: float = float(_get("TIME_DECAY_HALFLIFE_DAYS", "180") or 180)
    CLARIFY_CONFIDENCE: float = 0.30  # 证据得分低于该阈值触发澄清/拒答

    # 角色 -> 可见文档角色标签
    ROLE_LEVELS: dict[str, int] = {
        "customer": 0,        # 客户：仅公开文档
        "customer_service": 1,  # 客服：公开 + 内部 SOP
        "operations": 2,      # 运营：公开 + 内部 + 承运商商务规则
        "admin": 3,           # 管理员：全部
    }


settings = Settings()
