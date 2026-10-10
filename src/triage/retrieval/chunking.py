"""Clause-aware chunking.

Why not fixed-size chunks: in the CAR the compensation amounts and the conditions that remove them sit in the same
numbered paragraph (3.3.2 lists the amounts *and* says no compensation is payable once the alternate flight is
accepted), and the terms they use are defined at the start of the document. A 500-character window splits the
amounts from their conditions, and a retriever that finds the amounts and stops will state an entitlement that may
not exist. So:

  * one chunk = one numbered provision (e.g. CAR:3.3.2), keeping its sub-paras (a)-(d) and trailing conditions
    together. Provisions longer than MAX_CHARS are split on sub-paragraph boundaries and share the clause id;
  * each chunk carries its heading path ("3 REQUIREMENTS > 3.3 Cancellation") and the document title, prepended to
    the embedded text so short provisions keep their context;
  * cross-references ("para 3.4.1", "paras 3.4.1 to 3.4.3") and defined terms used are extracted as metadata, giving
    the agent's `find_related` tool a reference graph in both directions (what a provision cites, and which
    provisions cite it - that is where exemptions live).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

MAX_CHARS = 2400
CHUNKER_VERSION = "clause-2"  # bump when chunking logic changes; forces re-processing
CLAUSE_RX = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){1,3}|\d{1,2}\.)\s+(\S.*)$")
REF_RX = re.compile(r"\bparas?(?:graphs?)?\s+(\d+(?:\.\d+)+)(?:\s*(to|and|or|,)\s*(\d+(?:\.\d+)+))?", re.I)
DEF_RX = re.compile(r'^"([^"]+)"\s+means', re.I)


@dataclass
class Chunk:
    chunk_id: str
    clause_id: str
    doc_id: str
    number: str
    title_path: str
    text: str
    doc_title: str
    doc_version: str
    authority: str
    provenance: str
    parent: Optional[str] = None
    refs_out: list[str] = field(default_factory=list)
    defines: Optional[str] = None
    terms_used: list[str] = field(default_factory=list)
    content_hash: str = ""

    def embed_text(self) -> str:
        return f"{self.doc_title} | {self.title_path}\n{self.number} {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def _expand_range(a: str, b: str) -> list[str]:
    pa, pb = a.split("."), b.split(".")
    if len(pa) == len(pb) and pa[:-1] == pb[:-1] and pa[-1].isdigit() and pb[-1].isdigit():
        lo, hi = int(pa[-1]), int(pb[-1])
        if 0 <= hi - lo <= 10:
            return [".".join(pa[:-1] + [str(i)]) for i in range(lo, hi + 1)]
    return [a, b]


def extract_refs(text: str) -> list[str]:
    refs: list[str] = []
    for m in REF_RX.finditer(text):
        a, joiner, b = m.group(1), m.group(2), m.group(3)
        if b and joiner and joiner.lower() == "to":
            refs.extend(_expand_range(a, b))
        else:
            refs.append(a)
            if b:
                refs.append(b)
    seen, out = set(), []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def strip_front_matter(text: str) -> tuple[dict, str]:
    meta: dict = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            import yaml

            meta = yaml.safe_load(text[3:end]) or {}
            text = text[end + 4:]
    return meta, text


def parse_clauses(text: str) -> list[tuple[str, str, list[str]]]:
    """Return (number, heading_or_first_line, body_lines) per numbered block. Text before the first number is dropped."""
    blocks: list[tuple[str, str, list[str]]] = []
    current: Optional[tuple[str, str, list[str]]] = None
    for line in text.splitlines():
        m = CLAUSE_RX.match(line)
        if m and not re.match(r"^\d{1,2}\.\d", m.group(2)):
            if current:
                blocks.append(current)
            current = (m.group(1).rstrip("."), m.group(2).strip(), [])
            continue
        if current is not None:
            current[2].append(line.rstrip())
    if current:
        blocks.append(current)
    return blocks


def _is_heading(first: str, body: list[str]) -> bool:
    body_text = " ".join(x for x in body if x.strip())
    return not body_text and len(first) < 80 and not first.endswith(".")


def chunk_document(text: str, doc_id: str, doc_title: str, doc_version: str, authority: str,
                   provenance: str) -> list[Chunk]:
    _, text = strip_front_matter(text)
    blocks = parse_clauses(text)
    headings: dict[str, str] = {}
    chunks: list[Chunk] = []
    for number, first, body in blocks:
        if _is_heading(first, body):
            headings[number] = first.title() if first.isupper() else first
            continue
        parts = number.split(".")
        path = [f"{'.'.join(parts[:i])} {headings['.'.join(parts[:i])]}" for i in range(1, len(parts))
                if ".".join(parts[:i]) in headings]
        body_text = "\n".join([first] + body).strip()
        body_text = re.sub(r"[ \t]+", " ", body_text)
        body_text = re.sub(r"\n{2,}", "\n", body_text)
        parent = ".".join(parts[:-1]) if len(parts) > 1 else None
        dm = DEF_RX.match(first)
        pieces = _split_long(body_text)
        for i, piece in enumerate(pieces):
            cid = f"{doc_id}:{number}"
            chunk = Chunk(chunk_id=cid if i == 0 else f"{cid}#p{i + 1}", clause_id=cid, doc_id=doc_id, number=number,
                          title_path=" > ".join(path), text=piece, doc_title=doc_title, doc_version=doc_version,
                          authority=authority, provenance=provenance,
                          parent=f"{doc_id}:{parent}" if parent else None,
                          refs_out=[f"{doc_id}:{r}" for r in extract_refs(piece)],
                          defines=dm.group(1).lower() if dm else None)
            chunk.content_hash = hashlib.sha256((chunk.embed_text()).encode()).hexdigest()
            chunks.append(chunk)
    defined = {c.defines: c.clause_id for c in chunks if c.defines}
    for c in chunks:
        low = c.text.lower()
        c.terms_used = sorted(t for t, cid in defined.items() if cid != c.clause_id and t in low)
    return chunks


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    parts = re.split(r"\n(?=\([a-z]\))", text)
    out, buf = [], ""
    for p in parts:
        if len(buf) + len(p) > MAX_CHARS and buf:
            out.append(buf.strip())
            buf = ""
        buf += ("\n" if buf else "") + p
    if buf:
        out.append(buf.strip())
    return out


def chunk_fixed(text: str, doc_id: str, doc_title: str, doc_version: str, authority: str, provenance: str,
                size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Naive baseline used only for the chunking comparison in the evaluation."""
    _, text = strip_front_matter(text)
    flat = " ".join(text.split())
    chunks = []
    i = 0
    n = 0
    while i < len(flat):
        piece = flat[i:i + size]
        m = re.search(r"\b(\d\.\d(?:\.\d)?)\b", piece)
        number = m.group(1) if m else f"w{n}"
        c = Chunk(chunk_id=f"{doc_id}:w{n}", clause_id=f"{doc_id}:{number}", doc_id=doc_id, number=number,
                  title_path="", text=piece, doc_title=doc_title, doc_version=doc_version, authority=authority,
                  provenance=provenance)
        c.content_hash = hashlib.sha256(c.embed_text().encode()).hexdigest()
        chunks.append(c)
        i += size - overlap
        n += 1
    return chunks


def read_document(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages = [(p.extract_text() or "") for p in reader.pages]
        text = "\n".join(pages)
        if len(text.strip()) < 200:
            raise ValueError(f"{path} has no usable text layer; run OCR (e.g. ocrmypdf) before ingesting")
        return text
    with open(path, encoding="utf-8") as fh:
        return fh.read()
