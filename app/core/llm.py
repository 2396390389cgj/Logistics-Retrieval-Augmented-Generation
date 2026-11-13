"""LLM 客户端：OpenAI 兼容接口；无 Key 时返回 None，由上层走兜底生成器。"""
from __future__ import annotations

import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self) -> None:
        self._client = None
        if settings.OPENAI_API_KEY:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    base_url=settings.OPENAI_BASE_URL,
                    timeout=30.0,
                    max_retries=1,
                )
                logger.info("LLM 已启用：%s", settings.LLM_MODEL)
            except Exception as e:  # noqa: BLE001
                logger.warning("OpenAI 客户端初始化失败：%s", e)

    @property
    def available(self) -> bool:
        return self._client is not None

    async def chat(self, prompt: str, max_tokens: int = 512,
                   temperature: float = 0.3) -> str:
        if self._client is None:
            raise RuntimeError("LLM 未配置")
        resp = await self._client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    async def chat_stream(self, messages: list[dict], max_tokens: int = 1024):
        """流式输出，yield 文本增量。"""
        if self._client is None:
            raise RuntimeError("LLM 未配置")
        stream = await self._client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
