"""融合生成：知识证据 + 实时证据 → 答案（SSE 流式 / JSON）。

策略：
- LLM 可用：Prompt 融合双通道证据流式生成，带引用标注；
- LLM 不可用：抽取式兜底（按证据相关度拼接，保留引用），保证离线可用；
- 证据置信度不足 → 澄清/拒答，绝不编造。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from config.settings import settings
from app.core.llm import LLMClient
from app.retrieval.bm25 import Hit
from app.tools.agent import LogisticsAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是物流公司的智能客服。回答规则：
1. 只依据提供的【知识证据】和【实时业务证据】回答，禁止编造；
2. 涉及承运商规则时注明版本与生效日期；实时状态以实时证据为准；
3. 回答末尾用 [来源: 文档标题 v版本] 标注引用；知识证据与实时证据冲突时以实时为准并说明；
4. 证据不足时明确告知无法确认，引导用户补充信息或转人工；
5. 简洁、分点、面向客服话术。"""

CLARIFY_REPLY = (
    "抱歉，我暂时没有找到足够依据回答这个问题。您可以：\n"
    "1. 补充承运商（如顺丰/京东/德邦）或具体线路信息后重新提问；\n"
    "2. 提供运单号让我查询实时物流状态；\n"
    "3. 回复“转人工”接入人工客服。"
)


@dataclass
class AnswerResult:
    answer: str
    references: list[dict]
    evidences: list[dict]
    confidence: float
    clarified: bool


def build_context(knowledge_hits: list[Hit], evidences: list[dict]) -> str:
    parts = []
    if knowledge_hits:
        parts.append("【知识证据】")
        for i, h in enumerate(knowledge_hits, 1):
            m = h.meta
            parts.append(f"[知识{i}]（{m.get('title', '')} {m.get('version', '')} "
                         f"生效:{m.get('effective_date', '-')}）\n{h.text}")
    if evidences:
        parts.append("【实时业务证据】")
        for i, e in enumerate(evidences, 1):
            parts.append(f"[实时{i}] {e['summary']}")
    return "\n\n".join(parts)


# 覆盖率计算时忽略的功能词（不参与相关性判断）
_COVERAGE_STOPWORDS = {"要", "的", "吗", "呢", "吧", "了", "是", "有", "在", "能", "可以",
                       "怎么", "多少", "请问", "一下", "什么", "我", "你", "他", "这", "那",
                       "该", "想", "查", "告诉我", "帮忙", "需要", "知道", "看看", "附近", "周边", "附近有",
                       "到", "去", "从", "还", "都", "很", "挺"}


def _content_tokens(text: str) -> set[str]:
    """查询内容词：精确模式分词 + 停用词/纯符号过滤。"""
    import re as _re
    try:
        import jieba
        tokens = jieba.lcut(text)
    except ImportError:
        tokens = tokenize(text)
    out = set()
    for t in tokens:
        t = t.strip()
        if not t or t in _COVERAGE_STOPWORDS:
            continue
        if not _re.search(r"[\u4e00-\u9fffA-Za-z0-9]", t):
            continue
        out.add(t)
    return out


# 分词黏连时剥离的功能字后缀/前缀（如"顺丰能"→"顺丰"）
_FUNC_CHARS = set("的了吗呢吧啊呀哦能否可以要会想需应该")


def _strip_func_affixes(token: str) -> str:
    while len(token) > 1 and token[0] in _FUNC_CHARS:
        token = token[1:]
    while len(token) > 1 and token[-1] in _FUNC_CHARS:
        token = token[:-1]
    return token


def _coverage(query_tokens: set[str], hit_text: str) -> float:
    """子串匹配覆盖率：规避查询/文档分词不一致的问题。"""
    if not query_tokens:
        return 0.0
    matched = 0
    for t in query_tokens:
        t2 = _strip_func_affixes(t)
        if t in hit_text or (t2 and t2 in hit_text):
            matched += 1
    return matched / len(query_tokens)


def _max_confidence(query: str, hits: list[Hit], evidences: list[dict]) -> float:
    if evidences and any(e["type"] == "tms_waybill" for e in evidences):
        return 1.0
    if not hits:
        return 0.0
    top = hits[0].score
    # 归一化 RRF 分数（单通道命中约 0.016，双通道约 0.033）
    rrf_norm = min(max(top / 0.05, 0.0), 1.0)
    # 词面覆盖率门控：Top-3 证据需真实覆盖查询核心词，否则视为低相关
    q_tokens = _content_tokens(query)
    if not q_tokens:
        return rrf_norm
    coverage = max(_coverage(q_tokens, h.text) for h in hits[:3])
    return rrf_norm * coverage


def extractive_fallback(query: str, hits: list[Hit], evidences: list[dict]) -> str:
    """兜底生成器：抽取式拼接 + 引用标注。"""
    lines: list[str] = []
    has_track = any(e["type"] == "tms_waybill" for e in evidences)
    for e in evidences:
        if e["type"] == "need_waybill_no":
            lines.append("请问您要查询的运单号是多少？（如 SF1234567890）")
        else:
            lines.append(e["summary"])
    # 已有实时运单证据时，实时状态优先，知识证据只保留最多一条做补充
    knowledge_hits = [] if has_track else hits
    seen: set[str] = set()
    for h in knowledge_hits[:3]:
        body = h.text.replace("【", "（").replace("】", "）").strip()
        if body in seen:
            continue
        seen.add(body)
        lines.append(f"{body} [来源: {h.meta.get('title', '')} {h.meta.get('version', '')}]")
    return "\n\n".join(lines) if lines else CLARIFY_REPLY


class AnswerGenerator:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()
        self.agent = LogisticsAgent()

    async def generate(self, query: str, knowledge_hits: list[Hit],
                       evidences: list[dict] | None = None) -> AnswerResult:
        evidences = evidences or []
        confidence = _max_confidence(query, knowledge_hits, evidences)
        references = [{"title": h.meta.get("title", ""), "version": h.meta.get("version", ""),
                       "chunk_id": h.chunk_id, "score": round(h.score, 4)}
                      for h in knowledge_hits]

        # 证据不足 → 澄清/拒答
        if confidence < settings.CLARIFY_CONFIDENCE and not evidences:
            return AnswerResult(answer=CLARIFY_REPLY, references=references,
                                evidences=evidences, confidence=confidence, clarified=True)

        context = build_context(knowledge_hits, evidences)

        if self.llm.available:
            prompt = f"{SYSTEM_PROMPT}\n\n{context}\n\n用户问题：{query}\n回答："
            try:
                answer = await self.llm.chat(prompt, max_tokens=800)
                return AnswerResult(answer=answer, references=references,
                                    evidences=evidences, confidence=confidence,
                                    clarified=False)
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM 生成失败，使用兜底生成：%s", e)

        answer = extractive_fallback(query, knowledge_hits, evidences)
        return AnswerResult(answer=answer, references=references,
                            evidences=evidences, confidence=confidence, clarified=False)

    async def generate_stream(self, query: str, knowledge_hits: list[Hit],
                              evidences: list[dict] | None = None):
        """SSE 流式生成：yield 文本增量字符串。"""
        evidences = evidences or []
        confidence = _max_confidence(query, knowledge_hits, evidences)

        if confidence < settings.CLARIFY_CONFIDENCE and not evidences:
            yield CLARIFY_REPLY
            return

        if self.llm.available:
            context = build_context(knowledge_hits, evidences)
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"{context}\n\n用户问题：{query}"}]
            try:
                async for delta in self.llm.chat_stream(messages):
                    yield delta
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM 流式失败，回退兜底：%s", e)
        yield extractive_fallback(query, knowledge_hits, evidences)
