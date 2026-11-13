"""层级切片（Hierarchical Chunking）：

- 按标题层级（#/##/### 或 [H1]/[H2] 标记）递归切分；
- 每个父级（h1/h2）生成一份"摘要切片"用于粗召回，叶子切片用于精引用；
- 切片自动携带面包屑（所属章节路径），保证检索命中叶子时上下文完整。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.metadata import ChunkMeta, stable_id

MAX_LEAF_CHARS = 500
OVERLAP_CHARS = 60


@dataclass
class Chunk:
    meta: ChunkMeta
    text: str
    breadcrumb: str = ""  # 如 "承运商规则 > 顺丰 > 禁运品清单"


_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$")


def _split_long(text: str, max_chars: int = MAX_LEAF_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """长段落滑窗切分，保留重叠避免语义截断。"""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    pieces, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # 优先在句号/换行处断开
        for sep in ("。", "；", "\n"):
            idx = text.rfind(sep, start + max_chars // 2, end)
            if idx != -1:
                end = idx + 1
                break
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return pieces


def hierarchical_chunk(meta_base: dict, text: str, doc_id: str) -> list[Chunk]:
    """输入整篇文档文本（Markdown 风格），输出层级切片列表。"""
    lines = text.splitlines()
    # 解析标题树
    sections: list[dict] = []  # {level, title, body_lines}
    current = {"level": 0, "title": meta_base.get("title", "文档"), "body": []}
    sections.append(current)
    for line in lines:
        m = _HEADING_RE.match(line.strip())
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            current = {"level": level, "title": title, "body": []}
            sections.append(current)
        else:
            current["body"].append(line)

    chunks: list[Chunk] = []
    breadcrumb_stack: dict[int, str] = {}
    order = 0

    for sec in sections:
        if sec["level"] == 0:
            continue
        # 维护面包屑
        breadcrumb_stack[sec["level"]] = sec["title"]
        higher = [breadcrumb_stack[k] for k in sorted(breadcrumb_stack) if k < sec["level"]]
        breadcrumb = " > ".join(higher + [sec["title"]])

        body = "\n".join(sec["body"]).strip()
        if not body:
            continue

        # 父级摘要切片（h1/h2）：短摘要，用于粗召回定位
        if sec["level"] <= 2:
            summary = body[:200]
            pm = dict(meta_base)
            pm["level"] = f"h{sec['level']}"
            pm["chunk_id"] = f"{doc_id}-h{sec['level']}-{stable_id(sec['title'])}"
            pm["order"] = order
            chunks.append(Chunk(meta=ChunkMeta(**pm), text=f"【{breadcrumb}】\n{summary}",
                                breadcrumb=breadcrumb))
            order += 1

        # 叶子切片
        for piece in _split_long(body):
            lm = dict(meta_base)
            lm["level"] = "leaf"
            lm["chunk_id"] = f"{doc_id}-leaf-{stable_id(breadcrumb + piece)}"
            lm["order"] = order
            chunks.append(Chunk(meta=ChunkMeta(**lm), text=f"【{breadcrumb}】\n{piece}",
                                breadcrumb=breadcrumb))
            order += 1

    # 无任何标题的纯文本文档
    if order == 0:
        whole = "\n".join(lines).strip()
        for piece in _split_long(whole):
            lm = dict(meta_base)
            lm["level"] = "leaf"
            lm["chunk_id"] = f"{doc_id}-leaf-{stable_id(piece)}"
            lm["order"] = order
            chunks.append(Chunk(meta=ChunkMeta(**lm), text=piece))
            order += 1
    return chunks
