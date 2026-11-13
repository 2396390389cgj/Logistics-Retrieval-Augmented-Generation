"""中文分词工具：优先 jieba，缺失时回退字符 bigram。"""
from __future__ import annotations

try:
    import jieba

    def tokenize(text: str) -> list[str]:
        return [t.strip() for t in jieba.lcut_for_search(text) if t.strip()]

except ImportError:  # pragma: no cover

    def tokenize(text: str) -> list[str]:
        text = "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
        tokens: list[str] = []
        buf = ""
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
                buf += ch
            else:
                if buf:
                    tokens.extend(buf[i:i + 2] for i in range(max(len(buf) - 1, 1)))
                    buf = ""
                if ch.isalnum():
                    tokens.append(ch.lower())
        if buf:
            tokens.extend(buf[i:i + 2] for i in range(max(len(buf) - 1, 1)))
        return tokens
