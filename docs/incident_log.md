# Incident log

These problems really happened while we built and tested the system. Each entry gives the symptom, how we
confirmed the cause, the fix, and what we changed so it does not happen again. Times are from 2026-10-10.

---

### INC-01: Chunker split CAR para 3.4.1 at a wrapped line ("5 hours;")

* **Symptom.** The retrieved text of `CAR:3.4.1` (meals and refreshments during a delay) stopped in the middle of
  sub-para (b), so the 4-hour threshold for flights over 5 hours of block time was missing from the evidence.
* **Confirmation.** `python -m triage.retrieval.pipeline info` listed a clause `CAR:5` that should not exist. Its
  text started with "5 hours; ...". The clause regex accepted any line starting with a number and whitespace, so a
  wrapped line was read as the heading of a new provision.
* **Fix.** We tightened `CLAUSE_RX` in `retrieval/chunking.py`. A top-level number now needs a trailing dot ("3."),
  and sub-provisions need at least one dotted level ("3.4.1"). We added a test that chunks 3.4.1 and checks that
  its facilities are complete.
* **Prevention.** `tests/test_chunking.py` now asserts the full list of clause ids for the corpus.

### INC-02: Re-ingest reported "up to date" after the chunker changed

* **Symptom.** After the INC-01 fix, `make ingest` printed `up to date` and the retriever kept serving the broken
  chunks.
* **Confirmation.** The change check compared only the file hash and the document version. Neither had changed.
* **Fix.** `CHUNKER_VERSION` (now `clause-2`) is part of the per-document state. A new chunker version forces those
  documents to be re-chunked. Unchanged chunks still reuse cached vectors, so this costs no embedding work.
* **Prevention.** The rule is written next to the constant: bump it whenever chunking logic changes.
  `test_chunker_version_change_forces_rechunk` covers it.

### INC-03: Log line on stdout broke the pipeline's JSON output

* **Symptom.** `pipeline ingest | jq` failed with a parse error.
* **Confirmation.** The structured JSON logger wrote to stdout, and so did the pipeline's machine-readable
  summary.
* **Fix.** `setup_logging` takes a stream. The CLI logs to stderr, and the services keep logging to stdout for
  Docker.

### INC-04: Extractor read "four hours later" as a delay

* **Symptom.** The section 4.1 case (S01) was classified as a delay, and the agent researched delay facilities
  instead of cancellation.
* **Confirmation.** The rule extractor's delay pattern matched "hours later" with no word boundary, and the
  "cancelled at the airport" phrasing was not recognised.
* **Fix.** We added word boundaries, made cancellation take precedence over a stated alternate departure, and added
  detection of an accepted alternate flight for denied boarding. S01, S12 and S16 were added to the eval set as
  regression cases.

### INC-05: Verifier accepted a cancellation clause for a denied-boarding case

* **Symptom.** In S12 and S16 the stub model's draft cited `CAR:3.3.2` (cancellation). The draft passed verification
  because the quote was verbatim and the amount matched.
* **Confirmation.** The trace showed `consistent_with_rules` passing on amount alone. No check asked whether the
  answer rests on the provision the outcome actually comes from.
* **Fix.** We added the `cites_decisive_provision` check. A model answer must cite every basis, amount and exemption
  clause that the rule engine's position depends on.
* **Prevention.** The eval metric `must_cite_recall` covers this, and it is now 1.0 across all 27 scenarios.

### INC-06: Over-retrieval diluted the context (fixed as the L3 change)

* **Symptom.** Context relevance was 0.16, and runs carried about 24 passages. Most of them came from
  `find_related` following every reference edge, including siblings.
* **Fix.** We added a relation filter (`FIND_RELATED_RELATIONS`, dropping `sibling` and `definition` edges) and lowered `DERIVED_K` from 5 to 3. The
  measured before/after is in `REPORT.md` §5.

### INC-07: Locust could not write its results (PermissionError)

* **Symptom.** `locust --csv` failed on the bind-mounted `loadtest/results/`.
* **Confirmation.** The app image runs as a non-root user (uid 10001), and the host directory belongs to root.
* **Fix.** Only the tool containers (`eval`, `locust`) run as `user: "0:0"`. Services stay non-root.

### INC-08: Cache stampede under load

* **Symptom.** With the whole-run cache on, p95 at 40 users was about 30 s. That was no better than with the cache
  off, even though most claims were repeats.
* **Confirmation.** The Grafana panel `runcache_lookups_total{result="miss"}` showed identical claims missing
  together. N concurrent copies of one claim all missed and all started full runs, because the first had not
  finished.
* **Fix.** Single-flight coalescing in `services/api.py`. The first miss sets `runcache:inflight:<key>`, and later
  copies attach to that run and wait for its result. Afterwards p95 was **34 ms** at 16.4 runs/s (see
  `loadtest/results/cache_on_w3c16_u40_stats.csv`).

### INC-09: Prometheus counters went backwards with two API workers

* **Symptom.** `rate(claims_accepted_total[1m])` showed impossible negative dips.
* **Confirmation.** `uvicorn --workers 2` runs two processes, each with its own registry. A scrape hits one of them
  at random.
* **Fix.** The API runs as a single async process, which is enough because it never waits on a model. You scale it
  with replicas, which Prometheus finds through dns_sd.

### INC-10: Demo results came from the cache, and the breaker stayed open into the next demo item

* **Symptom.** Repeat demo runs returned instantly with `cached: true`, which hid the agent's work. After the
  outage demo, the next item ran in degraded mode.
* **Fix.** The demo sends `X-Cache-Bypass: 1`. The outage demo restores the stub, then waits for
  `BREAKER_OPEN_SECONDS` so the next call is the half-open trial, and it shows the breaker closing.

### INC-11: Build environment problems (not product defects, recorded for whoever rebuilds)

* Docker Hub rate-limited image pulls (HTTP 429). We pulled the same images from `mirror.gcr.io` and retagged them.
* The PyTorch wheel index and Hugging Face were unreachable from the build sandbox. The image therefore has an
  `INSTALL_EMBEDDINGS` build arg, and the embeddings fall back to deterministic hashing vectors when the
  sentence-transformers model cannot be downloaded (`EMBEDDING_ALLOW_FALLBACK=true`). All our measurements in this
  repository used the hashing backend. On a normal machine the default `.env.example` uses all-MiniLM-L6-v2.
