"""存储层封装：MySQL（会话/反馈，降级 SQLite）、Redis（缓存，降级内存）。"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------- 关系库 ----------------
class RelationalStore:
    """会话与查询日志。配置 MySQL 时用 MySQL，否则降级 SQLite。"""

    def __init__(self) -> None:
        url = settings.MYSQL_URL or "sqlite:///data/app.db"
        try:
            from sqlalchemy import (Column, DateTime, Float, Integer, MetaData,
                                    String, Table, Text, create_engine)
            self.engine = create_engine(url, pool_pre_ping=True, future=True)
            meta = MetaData()
            self.chat_logs = Table(
                "chat_logs", meta,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("session_id", String(64), index=True),
                Column("role", String(32)),
                Column("query", Text),
                Column("answer", Text),
                Column("references_json", Text),
                Column("confidence", Float, default=0.0),
                Column("latency_ms", Integer, default=0),
                Column("created_at", DateTime, server_default="CURRENT_TIMESTAMP"),
            )
            meta.create_all(self.engine)
            logger.info("关系库就绪：%s", url.split("@")[-1] if "@" in url else url)
        except Exception as e:  # noqa: BLE001
            logger.warning("关系库不可用，查询日志仅内存记录：%s", e)
            self.engine = None

    def log_chat(self, session_id: str, role: str, query: str, answer: str,
                 references: list[dict], confidence: float, latency_ms: int) -> None:
        if self.engine is None:
            return
        try:
            from sqlalchemy import insert
            with self.engine.begin() as conn:
                conn.execute(insert(self.chat_logs).values(
                    session_id=session_id, role=role, query=query, answer=answer,
                    references_json=json.dumps(references, ensure_ascii=False),
                    confidence=confidence, latency_ms=latency_ms))
        except Exception as e:  # noqa: BLE001
            logger.warning("会话日志写入失败：%s", e)


# ---------------- 缓存 ----------------
class CacheStore:
    """查询结果缓存：Redis 优先，降级进程内 TTL 缓存。"""

    def __init__(self) -> None:
        self._mem: dict[str, tuple[float, Any]] = {}
        self._r = None
        if settings.REDIS_URL:
            try:
                import redis.asyncio as aioredis
                self._r = aioredis.from_url(settings.REDIS_URL,
                                            decode_responses=True)
                logger.info("Redis 缓存已启用")
            except Exception as e:  # noqa: BLE001
                logger.warning("Redis 不可用，降级内存缓存：%s", e)
        self.ttl = 300

    async def get(self, key: str) -> Any | None:
        if self._r is not None:
            try:
                raw = await self._r.get(key)
                return json.loads(raw) if raw else None
            except Exception:  # noqa: BLE001
                return None
        item = self._mem.get(key)
        if item and time.time() - item[0] < self.ttl:
            return item[1]
        self._mem.pop(key, None)
        return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ttl = ttl or self.ttl
        if self._r is not None:
            try:
                await self._r.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
            except Exception:  # noqa: BLE001
                pass
        self._mem[key] = (time.time(), value)
