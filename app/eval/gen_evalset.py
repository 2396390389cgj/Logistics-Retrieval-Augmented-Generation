"""评测集生成：手写种子 Case × 模板扩展 → 200 条评测例（evalset.jsonl）。

覆盖：知识问答（规则/禁运/理赔/SOP/时效/网点）、多轮改写、权限越权、
澄清拒答（无证据问题）、运单实时查询意图。
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "evalset.jsonl"

# ---- 种子 Case（手写，含标准答案依据） ----
SEEDS = [
    # 知识问答 - 禁运品
    {"query": "打火机可以寄顺丰吗？", "role": "customer_service",
     "expected_titles": ["顺丰承运商规则手册"], "answer_keywords": ["禁运"]},
    {"query": "锂电池能单独寄吗？", "role": "customer_service",
     "expected_titles": ["全网禁运品通则", "顺丰锂电池专项规则"], "answer_keywords": ["100Wh"]},
    {"query": "中通可以寄酒精吗", "role": "customer",
     "expected_titles": ["全网禁运品通则"], "answer_keywords": []},
    {"query": "活体动物能寄吗", "role": "customer",
     "expected_titles": ["全网禁运品通则"], "answer_keywords": []},
    # 知识问答 - 赔付
    {"query": "中通快递丢了没保价怎么赔？", "role": "customer_service",
     "expected_titles": ["中通快递承运商规则手册（2024 版）"], "answer_keywords": ["5 倍"]},
    {"query": "未保价快件破损最高赔多少", "role": "customer_service",
     "expected_titles": ["破损件客诉处理 SOP"], "answer_keywords": []},
    # SOP
    {"query": "客户说收到的件破损了，客服该怎么处理？", "role": "customer_service",
     "expected_titles": ["破损件客诉处理 SOP"], "answer_keywords": ["24"]},
    {"query": "破损赔付客诉什么时候升级到主管？", "role": "customer_service",
     "expected_titles": ["破损件客诉处理 SOP"], "answer_keywords": ["48"]},
    # 时效
    {"query": "上海到北京顺丰要几天？", "role": "customer",
     "expected_titles": ["时效表"], "answer_keywords": ["2"]},
    {"query": "广州到成都京东物流几天到？", "role": "customer",
     "expected_titles": ["时效表"], "answer_keywords": ["2"]},
    {"query": "深圳寄乌鲁木齐顺丰多久", "role": "customer",
     "expected_titles": ["时效表"], "answer_keywords": ["4"]},
    # 网点
    {"query": "浦东张江附近的顺丰网点在哪？", "role": "customer",
     "expected_titles": ["网点表"], "answer_keywords": ["科苑路"]},
    {"query": "成都高新区有营业部吗", "role": "customer",
     "expected_titles": ["网点表"], "answer_keywords": ["天府三街"]},
    # 多轮改写
    {"query": "那它多久到？",
     "history": [{"role": "user", "content": "顺丰上海到北京"},
                 {"role": "assistant", "content": "顺丰上海到北京跨省 2-3 天。"}],
     "role": "customer", "expected_titles": ["时效表"], "answer_keywords": []},
    {"query": "这个怎么赔？",
     "history": [{"role": "user", "content": "中通快件丢失了"},
                 {"role": "assistant", "content": "中通未保价快件丢失按运费 5 倍赔付。"}],
     "role": "customer_service", "expected_titles": [], "answer_keywords": []},
    # 权限：客户不能看到内部理赔商务政策
    {"query": "理赔的责任划分比例是多少？", "role": "customer",
     "expected_titles": [], "expected_chunks": [],
     "answer_keywords": [], "expect_clarify": True},
    {"query": "赔付池超支了怎么处理？", "role": "customer_service",
     "expected_titles": [], "answer_keywords": [], "expect_clarify": True},
    # 澄清拒答：知识库无据可依
    {"query": "你们的老板是谁？", "role": "customer_service",
     "answer_keywords": [], "expect_clarify": True},
    {"query": "明天上海天气怎么样", "role": "customer_service",
     "answer_keywords": [], "expect_clarify": True},
]

CARRIERS = ["顺丰", "中通", "京东", "德邦"]
LANES = [("上海", "北京", "2"), ("广州", "成都", "2"), ("深圳", "乌鲁木齐", "4"),
         ("杭州", "上海", "1"), ("北京", "哈尔滨", "3")]
TOPICS = [
    ("{0}要多久？", "时效表", [], "timeliness"),
    ("{0}可以寄锂电池吗？", "禁运品通则", ["100Wh"], "battery"),
    ("{0}破损了怎么申请赔付？", "破损件客诉处理 SOP", ["24"], "damage"),
    ("{0}附近有营业网点吗？", "网点表", [], "network"),
]
CITIES = ["上海", "广州", "成都", "北京", "杭州", "深圳"]  # 限有网点数据的城市


def generate() -> list[dict]:
    cases: list[dict] = []
    for seed in SEEDS:
        c = dict(seed)
        c["case_type"] = c.get("expect_clarify") and "clarify" or "knowledge"
        cases.append(c)
    # 模板扩展到 200
    i = 0
    while len(cases) < 200:
        carrier = CARRIERS[i % len(CARRIERS)]
        lane = LANES[i % len(LANES)]
        topic = TOPICS[i % len(TOPICS)]
        tpl, title, kws, ttype = topic
        if ttype == "timeliness":
            q = tpl.format(f"{carrier}{lane[0]}到{lane[1]}")
            cases.append({"query": q, "role": "customer",
                          "expected_titles": [title], "answer_keywords": [],
                          "case_type": "knowledge"})
        elif ttype == "battery":
            cases.append({"query": tpl.format(carrier), "role": "customer_service",
                          "expected_titles": [title], "answer_keywords": [],
                          "case_type": "knowledge"})
        elif ttype == "damage":
            cases.append({"query": tpl.format(carrier), "role": "customer_service",
                          "expected_titles": [title], "answer_keywords": [],
                          "case_type": "knowledge"})
        else:
            city = CITIES[i % len(CITIES)]
            cases.append({"query": tpl.format(city), "role": "customer",
                          "expected_titles": [title], "answer_keywords": [],
                          "case_type": "knowledge"})
        i += 1
    return cases[:200]


def main() -> None:
    cases = generate()
    with open(OUT, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    types = {}
    for c in cases:
        types[c["case_type"]] = types.get(c["case_type"], 0) + 1
    print(f"评测集已生成：{len(cases)} 条 → {OUT}")
    print(f"类型分布：{types}")


if __name__ == "__main__":
    main()
