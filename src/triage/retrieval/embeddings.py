"""Embedding backends behind the LangChain `Embeddings` interface, with a content-addressed cache.

* sentence-transformers (default in .env.example): a small local model, no API cost.
* hashing: deterministic feature hashing of word uni/bi-grams. No download; used when the model cannot be fetched
  (EMBEDDING_ALLOW_FALLBACK=true) and in offline tests. The index records which backend built it, and a backend
  change triggers re-embedding rather than silently mixing vector spaces.

The cache key is (model id, sha256(text)); re-running ingestion only embeds new or changed chunks.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from collections import OrderedDict
from typing import Optional

import numpy as np
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)
_STOP = set("the a an of to and or in on for by is are be as at with from that this which shall any its it under "
            "not no if such than other all has have was were".split())


def _tokens(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]
    stem = [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words]
    return stem + [f"{a}_{b}" for a, b in zip(stem, stem[1:])]


class HashingEmbeddings(Embeddings):
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.model_id = f"hashing-{dim}"

    def _one(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.model_id = model_name
        self._lock = threading.Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings(backend: str, model: str, allow_fallback: bool) -> Embeddings:
    if backend == "hashing":
        return HashingEmbeddings()
    if backend == "sentence-transformers":
        if not model:
            raise ValueError("EMBEDDING_MODEL must be set for the sentence-transformers backend")
        try:
            return SentenceTransformerEmbeddings(model)
        except Exception as exc:
            if not allow_fallback:
                raise
            logger.error("embedding model unavailable (%s); falling back to hashing embeddings", exc)
            return HashingEmbeddings()
    raise ValueError(f"unknown EMBEDDING_BACKEND {backend!r}")


class DiskEmbeddingCache:
    """Content-addressed vector cache stored next to the index (one JSON file per model)."""

    def __init__(self, directory: str, model_id: str):
        os.makedirs(directory, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model_id)
        self.path = os.path.join(directory, f"emb_cache_{safe}.json")
        self.data: dict[str, list[float]] = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as fh:
                self.data = json.load(fh)

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        return self.data.get(self.key(text))

    def put(self, text: str, vec: list[float]) -> None:
        self.data[self.key(text)] = vec

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh)
        os.replace(tmp, self.path)


class QueryEmbeddingLRU:
    def __init__(self, emb: Embeddings, size: int = 2048):
        self.emb = emb
        self.size = size
        self._d: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def embed_query(self, text: str) -> tuple[list[float], bool]:
        with self._lock:
            if text in self._d:
                self._d.move_to_end(text)
                self.hits += 1
                return self._d[text], True
        vec = self.emb.embed_query(text)
        with self._lock:
            self.misses += 1
            self._d[text] = vec
            if len(self._d) > self.size:
                self._d.popitem(last=False)
        return vec, False
