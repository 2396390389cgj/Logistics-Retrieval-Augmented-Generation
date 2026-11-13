"""多轮查询改写：

将多轮对话中的口语化问题改写为自包含检索查询：
- 指代消解（"它/这个/上面说的"）
- 承运商上下文继承（上一轮提到顺丰，本轮"多久到"→"顺丰多久到"）
- 主题延续（运输/理赔/时效场景词补全）
- LLM 可用时由 LLM 改写，不可用时规则改写（离线可用）
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

CARRIERS = ["顺丰", "京东", "德邦", "中通", "圆通", "韵达", "EMS"]
CARRIER_ALIAS = {"sf": "顺丰", "jd": "京东", "zto": "中通", "debang": "德邦"}

# 主题词 → 补全
TOPIC_HINTS = [
    (re.compile(r"多久|几天|几天到|多快|时效"), "运输时效"),
    (re.compile(r"坏了|丢了|破损|赔偿|赔"), "理赔赔偿"),
    (re.compile(r"禁运|不能寄|可以寄"), "禁运品规则"),
    (re.compile(r"网点|地址|在哪|营业"), "网点信息"),
    (re.compile(r"多少钱|费用|价格|收费"), "运费标准"),
]

PRONOUNS = ["它", "这个", "那个", "上面说的", "这个件", "该件", "此单"]


def rewrite_rule_based(query: str, history: list[dict]) -> str:
    """规则改写：history = [{"role": "user"/"assistant", "content": "..."}]"""
    if not history:
        return query.strip()
    last_user = next((m["content"] for m in reversed(history)
                      if m.get("role") == "user"), "")
    last_answer = next((m["content"] for m in reversed(history)
                        if m.get("role") == "assistant"), "")

    rewritten = query.strip()

    # 1) 指代消解：问题极短且含指代 → 拼接上一轮用户问题主体
    if len(rewritten) <= 12 and any(p in rewritten for p in PRONOUNS):
        rewritten = f"{last_user} {rewritten}"

    # 2) 承运商继承：本轮未提承运商且上一轮提到 → 继承
    if not any(c in rewritten for c in CARRIERS):
        for c in CARRIERS:
            if c in last_user or c in last_answer:
                rewritten = f"{c} {rewritten}"
                break

    # 3) 主题补全：本轮问题过于口语化时补主题词
    if len(rewritten) <= 8:
        for pat, topic in TOPIC_HINTS:
            if pat.search(rewritten):
                rewritten = f"{topic} {rewritten}"
                break
    return rewritten


async def rewrite_query(query: str, history: list[dict], llm=None) -> str:
    """LLM 改写优先，失败/不可用回退规则改写。"""
    if llm is not None and history:
        prompt = (
            "你是搜索查询改写器。根据对话历史把用户最新问题改写为一句自包含的检索查询，"
            "补全指代与省略的承运商/主题，不要解释，直接输出改写后的查询。\n"
            f"对话历史：{history[-4:]}\n最新问题：{query}\n改写后查询："
        )
        try:
            out = await llm.chat(prompt, max_tokens=64, temperature=0.0)
            if out and out.strip():
                return out.strip().strip("\"「」")
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM 改写失败，使用规则改写：%s", e)
    return rewrite_rule_based(query, history)
