import asyncio
import os

from triage.retrieval.chunking import chunk_document, read_document
from triage.retrieval.embeddings import HashingEmbeddings
from triage.retrieval.pipeline import ingest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "corpus", "manifest.yaml")
CAR_PDF = os.path.join(ROOT, "corpus", "raw", "dgca_car_s3_m_p4_rev4.pdf")


def test_clause_chunking_keeps_conditions_with_amounts():
    chunks = chunk_document(read_document(CAR_PDF), "CAR", "CAR", "r4", "regulation", "x")
    c = next(c for c in chunks if c.clause_id == "CAR:3.3.2")
    assert "INR 10,000" in c.text and "either provide alternate flight as acceptable" in c.text
    assert "CAR:3.8.1" in c.refs_out


def test_ingest_is_idempotent_and_incremental(tmp_path):
    emb = HashingEmbeddings()
    first = ingest(MANIFEST, str(tmp_path), emb)
    assert first["status"] == "built" and first["chunk_count"] > 40
    second = ingest(MANIFEST, str(tmp_path), emb)
    assert second["status"] == "up to date"


def test_agent_handles_difficult_case(tmp_path):
    from triage.agents.entitlement import build_graph, initial_state
    from triage.agents.intake import run_intake
    from triage.agents.runtime import AgentDeps, LocalRetriever, Recorder
    from triage.config import get_settings
    from triage.retrieval.index import ClauseIndex, current_version_dir
    from triage.stub_responses import InProcessStubLLM

    emb = HashingEmbeddings()
    ingest(MANIFEST, str(tmp_path), emb)
    deps = AgentDeps(llm=InProcessStubLLM(), retriever=LocalRetriever(ClauseIndex(current_version_dir(str(tmp_path)), emb)),
                     prompts_dir=os.path.join(ROOT, "prompts"), prompt_versions={})
    claim = {"client_reference": "T", "passenger": {"full_name": "A B", "email": "a@b.co"},
             "flight": {"airline_code": "DA", "flight_number": "1", "origin": "BLR", "destination": "DEL",
                        "scheduled_departure": "2026-09-14T09:30:00+05:30", "carrier_type": "indian_scheduled"},
             "booking": {"pnr": "ABC123"}, "disruption": {},
             "narrative": "My flight was cancelled. I was already at the airport and was told 90 minutes before departure. "
                          "The airline put me on a flight four hours later, which I took. I want the compensation for cancellation."}

    async def go():
        h = await run_intake("t", claim, deps, Recorder("t", "intake"))
        g = build_graph(deps, get_settings())
        return await g.ainvoke(initial_state("t", h), config={"configurable": {"rec": Recorder("t", "entitlement")}, "recursion_limit": 100})

    final = asyncio.run(go())
    r = final["result"]
    assert r["route"] == "decline" and r["entitlement"]["compensation_inr"] == 0
    assert r["verification"]["passed"]
    assert any("meals" in f for f in r["entitlement"]["facilities"])
    # the stub's naive "pay INR 10,000" answer must have been caught by verification
    assert r["basis"].startswith("model answer failed verification")


def test_wrapped_lines_do_not_start_new_clauses():
    # In the official PDF, "Para\n3.8.1 (a)." and "Para\n1.5 which ..." wrap onto new lines and look like headings.
    chunks = chunk_document(read_document(CAR_PDF), "CAR", "CAR", "r4", "regulation", "x")
    ids = [c.clause_id for c in chunks]
    assert len(ids) == len(set(ids)), "a clause id appears twice"
    by = {c.clause_id: c.text for c in chunks}
    assert by["CAR:3.3.3"].rstrip().endswith("3.8.1 (a).")
    assert "reasonable measures had" in by["CAR:3.4.4"]
    assert "c) 4 hours or more" in by["CAR:3.4.1"]
    assert "CIVIL AVIATION REQUIREMENTS" not in by["CAR:3.5.1"]  # running page header removed


def test_rule_terms_match_the_corpus():
    import yaml
    from triage.agents.verifier import norm
    texts: dict[str, str] = {}
    for path, doc in ((CAR_PDF, "CAR"), (os.path.join(ROOT, "corpus/raw/airline_passenger_charter.md"), "CHARTER")):
        for c in chunk_document(read_document(path), doc, doc, "v", "a", "p"):
            texts[c.clause_id] = texts.get(c.clause_id, "") + " " + c.text
    rules = yaml.safe_load(open(os.path.join(ROOT, "src/triage/rules/car_rules.yaml")))
    missing = [(k, t) for k, v in rules["clauses"].items() for t in v["terms"]
               if norm(t) not in norm(texts.get(v["id"], ""))]
    assert not missing, missing


def test_chunker_version_change_forces_rechunk(tmp_path, monkeypatch):
    from triage.retrieval import pipeline
    emb = HashingEmbeddings()
    ingest(MANIFEST, str(tmp_path), emb)
    assert ingest(MANIFEST, str(tmp_path), emb)["docs_skipped"] == 2
    monkeypatch.setattr(pipeline, "CHUNKER_VERSION", "clause-test")
    # every document is re-chunked; identical chunks reuse their vectors, so nothing is re-embedded
    assert ingest(MANIFEST, str(tmp_path), emb)["docs_skipped"] == 0
