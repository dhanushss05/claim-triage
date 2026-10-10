"""Retriever service: serves the clause index to the agents.

Its own container because its resource profile differs (embedding model in memory, CPU-bound query embedding)
and the index can be rebuilt and hot-reloaded without restarting the agents.

Endpoints: POST /search, GET /clause/{id}, GET /related/{id}, GET /index/info, POST /admin/reload, /health, /metrics.
Retrieval cache: results are cached in Redis keyed by (index version, mode, k, doc filter, query); a new index version
naturally invalidates the cache.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Literal, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

from ..config import get_settings
from ..metrics import (EMBED_CACHE, EMBEDDING_INFO, INDEX_AGE, INDEX_BUILT, INDEX_CHUNKS, INDEX_DOCS, INGEST_DURATION,
                       INGEST_EMBEDDED, INGEST_THROUGHPUT, RETRIEVAL_CACHE, SEARCH_LATENCY)
from ..retrieval.embeddings import build_embeddings
from ..retrieval.index import IndexHolder
from ..retrieval.pipeline import ingest
from ..store import connect
from .http_common import make_app

logger = logging.getLogger("retriever")
app = make_app("retriever")
s = get_settings()
emb = build_embeddings(s.embedding_backend, s.embedding_model, s.embedding_allow_fallback)
holder = IndexHolder(s.index_dir, emb)
r = connect(s.redis_url)


class SearchReq(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    doc_id: Optional[str] = None


def _publish_freshness() -> None:
    idx = holder.index
    if not idx:
        return
    st = idx.state
    INDEX_CHUNKS.set(st["chunk_count"])
    INDEX_DOCS.set(len(st["documents"]))
    INDEX_BUILT.set(st["built_ts"])
    INDEX_AGE.set(time.time() - st["built_ts"])
    li = st.get("last_ingest", {})
    INGEST_DURATION.set(li.get("duration_s") or 0)
    INGEST_THROUGHPUT.set(li.get("throughput_chunks_per_s") or 0)
    INGEST_EMBEDDED.set(li.get("chunks_embedded") or 0)
    EMBEDDING_INFO.labels(s.embedding_backend, st["embedding_model"]).set(1)


async def _watch() -> None:
    while True:
        try:
            if holder.maybe_reload():
                _publish_freshness()
            elif holder.index:
                INDEX_AGE.set(time.time() - holder.index.state["built_ts"])
        except Exception:
            logger.exception("index reload failed; keeping current index")
        await asyncio.sleep(10)


@app.on_event("startup")
async def startup() -> None:
    if not holder.maybe_reload():
        if holder.index is None:
            logger.warning("no index found; running ingestion pipeline at start-up")
            await asyncio.to_thread(ingest, s.corpus_manifest, s.index_dir, emb)
            holder.maybe_reload()
    _publish_freshness()
    asyncio.create_task(_watch())


@app.get("/health")
async def health():
    if holder.index is None:
        raise HTTPException(503, "index not loaded")
    return {"status": "ok", "index_version": holder.index.version, "chunks": len(holder.index.chunks)}


@app.get("/index/info")
async def info():
    if holder.index is None:
        raise HTTPException(503, "index not loaded")
    return holder.index.state


@app.post("/admin/reload")
async def reload():
    changed = holder.maybe_reload()
    _publish_freshness()
    return {"reloaded": changed, "index_version": holder.index.version if holder.index else None}


@app.post("/search")
async def search(req: SearchReq):
    idx = holder.index
    if idx is None:
        raise HTTPException(503, "index not loaded")
    key = "retr:" + hashlib.sha256(json.dumps([idx.version, req.mode, req.k, req.doc_id, req.query]).encode()).hexdigest()[:32]
    if s.retrieval_cache_enabled:
        cached = await r.get(key)
        if cached:
            RETRIEVAL_CACHE.labels("hit").inc()
            return {"passages": json.loads(cached), "cached": True, "index_version": idx.version}
        RETRIEVAL_CACHE.labels("miss").inc()
    t0 = time.perf_counter()
    passages, meta = await asyncio.to_thread(idx.search, req.query, req.k, req.mode, req.doc_id)
    SEARCH_LATENCY.labels(req.mode).observe(time.perf_counter() - t0)
    if req.mode != "bm25":
        EMBED_CACHE.labels("hit" if meta["embed_cache_hit"] else "miss").inc()
    if s.retrieval_cache_enabled:
        await r.set(key, json.dumps(passages), ex=3600)
    return {"passages": passages, "cached": False, "index_version": idx.version, **meta}


@app.get("/clause/{clause_id}")
async def clause(clause_id: str):
    idx = holder.index
    if idx is None:
        raise HTTPException(503, "index not loaded")
    p = idx.passage(clause_id, relation="lookup")
    if not p:
        raise HTTPException(404, f"clause {clause_id} not found")
    return p


@app.get("/related/{clause_id}")
async def related(clause_id: str, relations: str = ""):
    idx = holder.index
    if idx is None:
        raise HTTPException(503, "index not loaded")
    if not idx.passage(clause_id):
        raise HTTPException(404, f"clause {clause_id} not found")
    rel = {x for x in relations.split(",") if x} or None
    return {"clause_id": idx.normalise_id(clause_id), "passages": idx.related(clause_id, rel)}
