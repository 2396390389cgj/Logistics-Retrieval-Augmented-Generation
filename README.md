# Logistics RAG —— 物流客服智能问答系统

对接 TMS/OMS 的物流领域 RAG 问答系统，实现「知识问答 + 实时业务查询」双通道融合。

## 架构

```
用户问题
   │
   ├─ 多轮查询改写（指代消解/承运商上下文继承/时效补全）
   │
   ├───────── 知识通道（RAG）──────────┐        ├──── 实时通道（Agent）────┐
   │ 多源解析 → 层级切片 → 元数据规范    │        │ 意图识别                │
   │ BM25 ─┐                          │        │ ├ TMS：运单状态/轨迹     │
   │        ├→ RRF 融合 → 权限过滤 →    │        │ ├ OMS：订单/揽收/费用    │
   │ 向量  ─┘   时间衰减 → bge-reranker │        │ └ 规则引擎：运费/时效    │
   │              精排 Top-K            │        └──────────┬─────────────┘
   └──────────────┬────────────────────┘                   │
                  └─────────── 融合生成（SSE 流式）←────────┘
                     证据不足 → 澄清/拒答
```

## 岗位职责覆盖

| 简历需求 | 实现 |
|---|---|
| 多源知识（政策/SOP/时效表/网点表）解析、层级切片与元数据规范 | `app/ingestion/` |
| BM25 与向量混合召回、RRF 融合与重排序、多轮查询改写 | `app/retrieval/` |
| RAG 与实时运单工具双通道融合生成、对接 TMS/OMS 接口 | `app/tools/` + `app/generation/` |
| 承运商规则版本化、时间衰减与按角色的检索权限过滤 | `app/retrieval/filters.py` |
| 构造问答与评测集，跟踪 Bad Case 迭代 | `app/eval/` |

## 技术栈

Python、FastAPI、LangChain、LlamaIndex、bge-large-zh、Elasticsearch、FAISS、
bge-reranker、OpenAI API、Redis、MySQL、Docker。

> **零依赖可运行**：所有外部依赖（OpenAI/ES/Redis/MySQL/本地大模型）均为可选，
> 缺失时自动降级（Hash 向量 + 内置 BM25 + SQLite + 内存缓存 + 抽取式兜底生成），
> 保证 `pip install -r requirements-core.txt` 后即可端到端跑通。

## 快速开始

```bash
cd "D:/Python_code/Logistics Retrieval-Augmented Generation"
pip install -r requirements-core.txt

# 1. 构建索引（解析 data/raw → 层级切片 → BM25 + FAISS）
python scripts/build_index.py

# 2. 启动服务
uvicorn app.main:app --reload --port 8000
```

- Swagger 文档：<http://127.0.0.1:8000/docs>
- 问答（SSE 流式）：`GET /api/chat/stream?q=顺丰到北京要几天&role=customer_service`
- 问答（JSON）：`POST /api/chat` `{"query": "...", "role": "customer_service", "history": [...]}`
- 运单查询：`GET /api/track/{waybill_no}`
- 评测：`python -m app.eval.gen_evalset && python -m app.eval.evaluate`

## 离线评测基线（200 条评测例，兜底生成器）

| 指标 | 结果 |
|---|---|
| Hit@5（Top-5 文档命中） | 97.5% |
| MRR@5 | 78.8% |
| 回答关键词准确率 | 100% |
| 澄清率（应澄清 4 条全部正确触发） | 1.5% |


## 示例角色

`customer`（客户）/ `customer_service`（客服）/ `operations`（运营）/ `admin`（管理员），
不同角色可见知识范围不同，防止越权查看内部规则。

## 目录结构

```
config/settings.py        全局配置
app/ingestion/            多源解析、层级切片、元数据、摄取管线
app/retrieval/            bm25 / vector / hybrid(RRF) / reranker / filters / rewrite
app/generation/           融合生成、Prompt、SSE 流式、澄清拒答
app/tools/                TMS/OMS 客户端与 Agent 双通道编排
app/storage/              MySQL / Redis / ES 封装（含降级）
app/eval/                 评测集与指标（Hit@5、MRR、回答准确率）
data/raw/                 示例知识库（承运商规则/SOP/理赔/时效表/网点表）
scripts/build_index.py    索引构建脚本
Dockerfile / docker-compose.yml
```
