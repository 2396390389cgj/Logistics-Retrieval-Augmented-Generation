"""向量召回：bge-large-zh + FAISS；无本地模型时回退 Hash 词向量（离线可用）。"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import numpy as np

from app.retrieval.bm25 import Hit
from app.retrieval.tokenize import tokenize

logger = logging.getLogger(__name__)

_DIM = 768  # 与 bge-large-zh 对齐


class VectorRetriever:
    """FAISS 向量索引。embedding 源自动选择：
    1) sentence-transformers 加载 bge-large-zh（USE_LOCAL_EMBEDDING=true 时）
    2) 离线 Hash 词向量（确定性、无需网络，保证系统可运行）
    """

    def __init__(self, index_dir: Path, use_local: bool = False, model_name: str = "") -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._chunks: list[dict] = []
        self._faiss_index = None
        self._model = None
        if use_local:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
                logger.info("已加载本地 embedding 模型 %s", model_name)
            except Exception as e:  # noqa: BLE001
                logger.warning("本地 embedding 加载失败，回退 Hash 向量：%s", e)

    # ---------- embedding ----------
    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension() if self._model else _DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            vecs = self._model.encode(texts, normalize_embeddings=True,
                                      show_progress_bar=False)
            return np.asarray(vecs, dtype="float32")
        return np.vstack([_hash_embedding(t) for t in texts]).astype("float32")

    # ---------- 索引 ----------
    def build(self, chunks: list[dict]) -> None:
        self._chunks = chunks
        vecs = self.encode([c["text"] for c in chunks])
        try:
            import faiss
            index = faiss.IndexFlatIP(vecs.shape[1])
            index.add(vecs)
            self._faiss_index = index
            faiss.write_index(index, str(self.index_dir / "faiss.index"))
        except ImportError:
            self._faiss_index = None  # 回退为暴力余弦
            np.save(self.index_dir / "vectors.npy", vecs)
        with open(self.index_dir / "chunks.meta", "w", encoding="utf-8") as f:
            import json
            json.dump(chunks, f, ensure_ascii=False)

    def load(self) -> bool:
        import json
        meta_path = self.index_dir / "chunks.meta"
        if not meta_path.exists():
            return False
        with open(meta_path, encoding="utf-8") as f:
            self._chunks = json.load(f)
        try:
            import faiss
            self._faiss_index = faiss.read_index(str(self.index_dir / "faiss.index"))
        except Exception:  # noqa: BLE001
            self._faiss_index = None
        return True

    # ---------- 检索 ----------
    def search(self, query: str, topk: int = 50) -> list[Hit]:
        if not self._chunks:
            return []
        qv = self.encode([query])
        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(qv, min(topk, len(self._chunks)))
            pairs = list(zip(idxs[0].tolist(), scores[0].tolist()))
        else:
            import json
            vecs = np.load(self.index_dir / "vectors.npy")
            sims = (vecs @ qv[0]).tolist()
            order = np.argsort(sims)[::-1][:topk]
            pairs = [(int(i), sims[i]) for i in order]
        hits = []
        for idx, score in pairs:
            if idx < 0 or idx >= len(self._chunks):
                continue
            c = self._chunks[idx]
            hits.append(Hit(chunk_id=c["chunk_id"], score=float(score),
                            source="vector", text=c["text"], meta=c.get("meta", {})))
        return hits


def _hash_embedding(text: str) -> np.ndarray:
    """确定性 Hash 词向量：字符 n-gram + 词典词哈希到固定维度，L2 归一化。

    用于离线环境兜底——保证无 GPU/无网络时检索链路仍然可用且可复现。
    """
    vec = np.zeros(_DIM, dtype="float32")
    tokens = tokenize(text)
    # 词级 + 字符 bigram 双通道，增强中文召回
    grams = list(tokens)
    for t in tokens:
        if len(t) >= 2 and re.search(r"[\u4e00-\u9fff]", t):
            grams.extend(t[i:i + 2] for i in range(len(t) - 1))
    for g in grams:
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
        i = h % _DIM
        sign = 1.0 if (h >> 128) % 2 == 0 else -1.0
        vec[i] += sign * (1.0 if len(g) > 1 else 0.6)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec
