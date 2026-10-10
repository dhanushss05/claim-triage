"""Idempotent corpus pipeline: read -> parse -> chunk -> embed -> index.

    python -m triage.retrieval.pipeline ingest [--manifest PATH] [--index-dir DIR] [--reembed] [--chunking clause|fixed]
    python -m triage.retrieval.pipeline info

* Idempotent: a document whose file hash and version are unchanged is skipped. Re-running with no changes writes
  nothing and reports "up to date".
* Incremental: within a changed document only new/changed chunks are embedded; removed chunks are deleted from
  the FAISS index by id. Vectors come from a content-addressed cache, so re-embedding (`--reembed`, or a change of
  embedding model) does not re-parse documents or rebuild the whole index from source.
* Versioned output: each run that changes something writes INDEX_DIR/v_<version>/ and then atomically repoints
  INDEX_DIR/CURRENT. The retriever hot-reloads. The last three versions are kept for rollback.
* Freshness: state.json records when the index was built, from which document versions, how many chunks, the
  embedding model, and the run's throughput. The retriever exposes these as Prometheus metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Any

import yaml
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from ..config import get_settings
from ..logging_setup import setup_logging
from .chunking import CHUNKER_VERSION, Chunk, chunk_document, chunk_fixed, read_document
from .embeddings import DiskEmbeddingCache, build_embeddings
from .index import current_version_dir

logger = logging.getLogger("pipeline")


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def ingest(manifest_path: str, index_dir: str, embeddings, reembed: bool = False, chunking: str = "clause",
           keep_versions: int = 3) -> dict[str, Any]:
    t0 = time.perf_counter()
    os.makedirs(index_dir, exist_ok=True)
    base = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    model_id = getattr(embeddings, "model_id", type(embeddings).__name__)

    prev_dir = current_version_dir(index_dir)
    prev_state: dict[str, Any] = {}
    prev_chunks: dict[str, dict[str, Any]] = {}
    vs = None
    if prev_dir:
        with open(os.path.join(prev_dir, "state.json"), encoding="utf-8") as fh:
            prev_state = json.load(fh)
        with open(os.path.join(prev_dir, "chunks.json"), encoding="utf-8") as fh:
            prev_chunks = json.load(fh)
    model_changed = bool(prev_state) and prev_state.get("embedding_model") != model_id
    chunking_changed = bool(prev_state) and prev_state.get("chunking", "clause") != chunking
    full_reembed = reembed or model_changed
    if prev_dir and not full_reembed and not chunking_changed:
        vs = FAISS.load_local(os.path.join(prev_dir, "faiss"), embeddings, allow_dangerous_deserialization=True)

    cache = DiskEmbeddingCache(os.path.join(index_dir, "cache"), model_id)
    new_chunks: dict[str, dict[str, Any]] = {} if (full_reembed or chunking_changed) else dict(prev_chunks)
    if chunking_changed:
        prev_chunks = {}
    docs_state: dict[str, Any] = {}
    to_add: list[dict[str, Any]] = []
    to_delete: list[str] = []
    skipped = 0
    processed = 0

    wanted = {d["doc_id"] for d in manifest["documents"]}
    for cid, c in list(new_chunks.items()):
        if c["doc_id"] not in wanted:
            to_delete.append(cid)
            new_chunks.pop(cid)

    for doc in manifest["documents"]:
        path = os.path.join(base, doc["path"])
        fhash = _file_hash(path)
        prev_doc = prev_state.get("documents", {}).get(doc["doc_id"])
        unchanged = (prev_doc and prev_doc["file_hash"] == fhash and prev_doc["version"] == doc["version"]
                     and prev_doc.get("chunker") == CHUNKER_VERSION and not chunking_changed)
        if unchanged and not full_reembed:
            docs_state[doc["doc_id"]] = prev_doc
            skipped += 1
            continue
        text = read_document(path)
        maker = chunk_document if chunking == "clause" else chunk_fixed
        chunks: list[Chunk] = maker(text, doc["doc_id"], doc["title"], doc["version"], doc.get("authority", ""),
                                    doc.get("provenance", ""))
        processed += len(chunks)
        old_ids = {cid for cid, c in prev_chunks.items() if c["doc_id"] == doc["doc_id"]}
        new_ids = {c.chunk_id for c in chunks}
        for cid in old_ids - new_ids:
            if not full_reembed:
                to_delete.append(cid)
            new_chunks.pop(cid, None)
        for c in chunks:
            d = c.to_dict()
            old = prev_chunks.get(c.chunk_id)
            if full_reembed or not old or old["content_hash"] != c.content_hash:
                if old and not full_reembed:
                    to_delete.append(c.chunk_id)
                to_add.append(d)
            new_chunks[c.chunk_id] = d
        docs_state[doc["doc_id"]] = {"version": doc["version"], "file_hash": fhash, "chunks": len(chunks),
                                     "path": doc["path"], "provenance": doc.get("provenance", ""), "chunker": CHUNKER_VERSION,
                                     "effective_date": str(doc.get("effective_date", ""))}

    if full_reembed:
        to_add = list(new_chunks.values()) if not to_add else to_add
        seen = {d["chunk_id"] for d in to_add}
        to_add += [c for cid, c in new_chunks.items() if cid not in seen]

    if not to_add and not to_delete and prev_dir and not full_reembed:
        return {"status": "up to date", "index_version": prev_state["index_version"], "docs_skipped": skipped,
                "chunks": len(new_chunks), "duration_s": round(time.perf_counter() - t0, 3)}

    embedded = 0
    texts = [f"{c['doc_title']} | {c['title_path']}\n{c['number']} {c['text']}" for c in to_add]
    vectors: list[list[float]] = []
    missing = [i for i, t in enumerate(texts) if cache.get(t) is None]
    if missing:
        fresh = embeddings.embed_documents([texts[i] for i in missing])
        for i, v in zip(missing, fresh):
            cache.put(texts[i], v)
        embedded = len(missing)
    vectors = [cache.get(t) for t in texts]
    cache.save()

    pairs = list(zip(texts, vectors))
    metas = [{"chunk_id": c["chunk_id"], "clause_id": c["clause_id"], "doc_id": c["doc_id"]} for c in to_add]
    ids = [c["chunk_id"] for c in to_add]
    if vs is not None:
        existing = set(vs.index_to_docstore_id.values())
        dels = [i for i in set(to_delete) if i in existing]
        if dels:
            vs.delete(dels)
        if pairs:
            vs.add_embeddings(pairs, metadatas=metas, ids=ids)
    else:
        vs = FAISS.from_embeddings(pairs, embeddings, metadatas=metas, ids=ids,
                                   distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)

    version = hashlib.sha256((model_id + "|" + "|".join(sorted(c["content_hash"] for c in new_chunks.values()))).encode()).hexdigest()[:12]
    out_dir = os.path.join(index_dir, f"v_{version}")
    staging = out_dir + ".staging"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    vs.save_local(os.path.join(staging, "faiss"))
    with open(os.path.join(staging, "chunks.json"), "w", encoding="utf-8") as fh:
        json.dump(new_chunks, fh)
    duration = time.perf_counter() - t0
    state = {
        "index_version": version,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "built_ts": time.time(),
        "embedding_model": model_id,
        "chunking": chunking,
        "documents": docs_state,
        "chunk_count": len(new_chunks),
        "faiss_vectors": vs.index.ntotal,
        "last_ingest": {"duration_s": round(duration, 3), "chunks_processed": processed, "chunks_embedded": embedded,
                        "chunks_added": len(to_add), "chunks_deleted": len(set(to_delete)), "docs_skipped": skipped,
                        "throughput_chunks_per_s": round((processed or len(to_add)) / duration, 2) if duration else None,
                        "full_reembed": full_reembed, "previous_version": prev_state.get("index_version")},
    }
    with open(os.path.join(staging, "state.json"), "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(staging, out_dir)
    tmp_ptr = os.path.join(index_dir, "CURRENT.tmp")
    with open(tmp_ptr, "w", encoding="utf-8") as fh:
        fh.write(os.path.basename(out_dir))
    os.replace(tmp_ptr, os.path.join(index_dir, "CURRENT"))
    versions = sorted((d for d in os.listdir(index_dir) if d.startswith("v_") and not d.endswith(".staging")),
                      key=lambda d: os.path.getmtime(os.path.join(index_dir, d)), reverse=True)
    for old in versions[keep_versions:]:
        shutil.rmtree(os.path.join(index_dir, old), ignore_errors=True)
    return {"status": "built", **state}


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    setup_logging("ingest", s.log_level, stream=sys.stderr)
    ap = argparse.ArgumentParser(prog="triage-ingest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("--manifest", default=s.corpus_manifest)
    p.add_argument("--index-dir", default=s.index_dir)
    p.add_argument("--reembed", action="store_true", help="re-embed every chunk from cache/model without re-parsing")
    p.add_argument("--chunking", choices=["clause", "fixed"], default=os.environ.get("CHUNKING", "clause"))
    i = sub.add_parser("info")
    i.add_argument("--index-dir", default=s.index_dir)
    args = ap.parse_args(argv)
    if args.cmd == "info":
        d = current_version_dir(args.index_dir)
        if not d:
            print("no index built")
            return 1
        with open(os.path.join(d, "state.json"), encoding="utf-8") as fh:
            print(fh.read())
        return 0
    emb = build_embeddings(s.embedding_backend, s.embedding_model, s.embedding_allow_fallback)
    result = ingest(args.manifest, args.index_dir, emb, reembed=args.reembed, chunking=args.chunking)
    logger.info("ingest finished", extra={"fields": {k: v for k, v in result.items() if k != "documents"}})
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
