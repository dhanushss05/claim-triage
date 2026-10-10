"""Served index: LangChain FAISS (dense) + BM25 (lexical), fused with reciprocal rank fusion, plus the clause
reference graph used by the `get_clause` and `find_related` tools.

Hybrid retrieval matters here: queries that name a paragraph number or an exact term ("alternate flight",
"block time") are matched lexically, while paraphrased passenger language is matched densely.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from .embeddings import QueryEmbeddingLRU, _tokens

logger = logging.getLogger(__name__)


def current_version_dir(index_dir: str) -> Optional[str]:
    ptr = os.path.join(index_dir, "CURRENT")
    if not os.path.exists(ptr):
        return None
    with open(ptr, encoding="utf-8") as fh:
        name = fh.read().strip()
    path = os.path.join(index_dir, name)
    return path if os.path.isdir(path) else None


class ClauseIndex:
    def __init__(self, version_dir: str, embeddings: Embeddings):
        self.version_dir = version_dir
        with open(os.path.join(version_dir, "state.json"), encoding="utf-8") as fh:
            self.state: dict[str, Any] = json.load(fh)
        with open(os.path.join(version_dir, "chunks.json"), encoding="utf-8") as fh:
            self.chunks: dict[str, dict[str, Any]] = json.load(fh)
        self.vs = FAISS.load_local(os.path.join(version_dir, "faiss"), embeddings,
                                   allow_dangerous_deserialization=True)  # our own artefact, written by pipeline.py
        self.qcache = QueryEmbeddingLRU(embeddings)
        self.ids = list(self.chunks)
        self.bm25 = BM25Okapi([_tokens(self._embed_text(self.chunks[i])) for i in self.ids])
        self.by_clause: dict[str, list[str]] = {}
        for cid, c in self.chunks.items():
            self.by_clause.setdefault(c["clause_id"], []).append(cid)
        self.cited_by: dict[str, set[str]] = {}
        for c in self.chunks.values():
            for ref in c.get("refs_out", []):
                for target in self._resolve(ref):
                    self.cited_by.setdefault(target, set()).add(c["clause_id"])
        self.version = self.state["index_version"]

    @staticmethod
    def _embed_text(c: dict[str, Any]) -> str:
        return f"{c['doc_title']} | {c['title_path']}\n{c['number']} {c['text']}"

    def _resolve(self, ref: str) -> list[str]:
        """A reference to a heading (e.g. para 3.4) resolves to its child provisions."""
        if ref in self.by_clause:
            return [ref]
        return sorted(cid for cid in self.by_clause if cid.startswith(ref + "."))

    def normalise_id(self, clause_id: str) -> str:
        cid = clause_id.strip()
        cid = re.sub(r"^(para(graph)?\s*)", "", cid, flags=re.I)
        if ":" not in cid:
            cid = f"CAR:{cid}"
        doc, num = cid.split(":", 1)
        return f"{doc.upper()}:{num}"

    def passage(self, clause_id: str, relation: str = "", score: float = 0.0) -> Optional[dict[str, Any]]:
        cid = self.normalise_id(clause_id)
        parts = self.by_clause.get(cid)
        if not parts:
            return None
        texts = [self.chunks[p]["text"] for p in parts]
        c = self.chunks[parts[0]]
        return {"clause_id": cid, "doc_id": c["doc_id"], "number": c["number"], "title_path": c["title_path"],
                "text": "\n".join(texts), "authority": c["authority"], "provenance": c["provenance"],
                "doc_version": c["doc_version"], "refs_out": c.get("refs_out", []),
                "cited_by": sorted(self.cited_by.get(cid, set())), "terms_used": c.get("terms_used", []),
                "relation": relation, "score": round(score, 4)}

    def search(self, query: str, k: int = 5, mode: str = "hybrid", doc_id: Optional[str] = None) -> tuple[list[dict], dict]:
        t0 = time.perf_counter()
        pool = max(k * 4, 20)
        dense_rank: dict[str, int] = {}
        bm25_rank: dict[str, int] = {}
        cache_hit = False
        if mode in ("dense", "hybrid"):
            vec, cache_hit = self.qcache.embed_query(query)
            docs = self.vs.similarity_search_with_score_by_vector(vec, k=pool)
            for r, (doc, _score) in enumerate(docs):
                dense_rank.setdefault(doc.metadata["chunk_id"], r)
        if mode in ("bm25", "hybrid"):
            scores = self.bm25.get_scores(_tokens(query))
            order = sorted(range(len(self.ids)), key=lambda i: -scores[i])[:pool]
            for r, i in enumerate(order):
                if scores[i] > 0:
                    bm25_rank[self.ids[i]] = r
        fused: dict[str, float] = {}
        for ranks in (dense_rank, bm25_rank):
            for cid, r in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + r + 1)
        best: dict[str, float] = {}
        for chunk_id, s in fused.items():
            c = self.chunks[chunk_id]
            if doc_id and c["doc_id"] != doc_id:
                continue
            best[c["clause_id"]] = max(best.get(c["clause_id"], 0.0), s)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])[:k]
        out = [self.passage(cid, relation="search", score=s) for cid, s in ranked]
        return [p for p in out if p], {"latency_ms": (time.perf_counter() - t0) * 1000, "embed_cache_hit": cache_hit}

    def related(self, clause_id: str, relations: Optional[set[str]] = None) -> list[dict[str, Any]]:
        cid = self.normalise_id(clause_id)
        base = self.passage(cid)
        if not base:
            return []
        out: dict[str, dict[str, Any]] = {}

        def add(target: str, relation: str) -> None:
            if relations is not None and relation not in relations:
                return
            if target != cid and target not in out:
                p = self.passage(target, relation=relation)
                if p:
                    out[target] = p

        for ref in base["refs_out"]:
            for t in self._resolve(ref):
                add(t, "referenced_by_this")
        for t in sorted(self.cited_by.get(cid, set())):
            add(t, "cites_this")  # exemptions and modifiers usually point back at the provision they modify
        parent = self.chunks[self.by_clause[cid][0]].get("parent")
        if parent:
            for sib in sorted(x for x in self.by_clause if x.startswith(parent + ".") and x.count(".") == cid.count(".")):
                add(sib, "sibling")
            for t in sorted(self.cited_by.get(parent, set())):
                add(t, "cites_parent")
        defs = {c["defines"]: c["clause_id"] for c in self.chunks.values() if c.get("defines")}
        for term in base["terms_used"]:
            if term in defs:
                add(defs[term], "definition")
        return list(out.values())


class IndexHolder:
    """Hot-swappable index: reloads when CURRENT points at a new version."""

    def __init__(self, index_dir: str, embeddings: Embeddings):
        self.index_dir = index_dir
        self.embeddings = embeddings
        self._lock = threading.Lock()
        self.index: Optional[ClauseIndex] = None
        self.loaded_dir: Optional[str] = None

    def maybe_reload(self) -> bool:
        d = current_version_dir(self.index_dir)
        if d and d != self.loaded_dir:
            idx = ClauseIndex(d, self.embeddings)
            with self._lock:
                self.index, self.loaded_dir = idx, d
            logger.info("index loaded", extra={"fields": {"version": idx.version, "chunks": len(idx.chunks)}})
            return True
        return False
