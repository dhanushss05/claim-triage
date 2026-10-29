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

---

The entries below come from replacing the working-text corpus with the official CAR PDF (Rev. 4) and switching the
model from the stub to Gemini, on 2026-10-29.

### INC-12: Wrapped cross-references in the official PDF became duplicate clauses

* **Symptom.** After ingesting the official PDF, the index had two chunks with id `CAR:3.8.1` and two with
  `CAR:1.5`. `CAR:3.3.3` and `CAR:3.4.4` were cut off mid-sentence.
* **Confirmation.** The PDF wraps "in accordance with Para" / "3.8.1 (a)." and "as defined in Para 1.4 and Para" /
  "1.5 which could not ..." across lines, so the second line starts with a clause number. INC-01's stricter regex
  does not help here, because "3.8.1" is a valid clause number.
* **Fix.** A line is accepted as a new clause only if numbering moves forward, the number has not been seen before,
  and the previous line does not end in "Para", "and", "to", "of", "with" and similar words.
* **Prevention.** `test_wrapped_lines_do_not_start_new_clauses` runs against the official PDF and asserts that every
  clause id is unique and that 3.3.3 and 3.4.4 are complete.

### INC-13: Running page headers inside provisions

* **Symptom.** The text of `CAR:3.3.4` and `CAR:3.5.1` contained "CIVIL AVIATION REQUIREMENTS SERIES 'M' PART IV
  SECTION 3 6TH AUGUST, 2010 6 Rev. 4, dated 25th Jan, 2023" in the middle of a sentence. A verbatim quote spanning
  the page break would never match.
* **Fix.** `clean_pdf_text` drops lines that recur on at least half of the pages, and lines that hold only a page
  number. It also closes up "3.8 .1" to "3.8.1", so the cross-reference resolves.

### INC-14: The rule engine's clause map no longer matched the regulation

* **Symptom.** Checking the old `car_rules.yaml` against the official chunks showed 42 of its 65 required phrases
  missing, affecting 33 of 36 rule components. Every citation built on them would have failed `quotes_in_text`, so
  runs would have escalated.
* **Confirmation.** The official CAR is organised differently from the working text. Extraordinary circumstances
  are in 1.4 and 1.5, not 3.6. Facilities are in 3.8.1, and the delay paragraphs refer to it. Para 3.3.2 has no "no
  compensation once the alternate flight is accepted" sentence; instead the airline must provide *either* an
  acceptable alternate flight *or* compensation plus refund. Para 3.4.4 removes the delay facilities under
  extraordinary circumstances, and there is no baggage clause. This is the behaviour we designed for: the verifier
  refused to cite text that was not there.
* **Fix.**
  * `car_rules.yaml` and the engine were remapped to the official paragraphs and wording.
  * A decline may now rest on a provision that makes compensation conditional (`decline_markers`), as well as on
    one that removes it.
  * The eval set's expectations and supporting passages were re-quoted from the official text. S26 changed from
    "facilities regardless of cause" to an escalation, because 3.4.4 makes facilities depend on the cause.
* **Prevention.** `test_rule_terms_match_the_corpus` fails if any phrase in `car_rules.yaml` is missing from the
  clause it names.

### INC-15: The official PDF was unreadable inside the container

* **Symptom.** `ingest` exited with `PermissionError: /app/corpus/raw/dgca_car_s3_m_p4_rev4.pdf`.
* **Confirmation.** The uploaded file had mode 600, `COPY` keeps the mode, and the container runs as uid 10001.
* **Fix.** `chmod 644`. Git stores the file as 644, so a fresh clone is not affected.

### INC-16: Gemini free-tier quota opened the circuit breaker during evaluation

* **Symptom.** The first Gemini evaluation finished in about a minute with 27/27 correct, but its prompt-token count
  was 74 per run.
* **Confirmation.** The gateway logged HTTP 429: "Generate content requests per minute limit exceeded on Free Tier,
  limit: 15". A run makes about 13 calls in about 15 s, so five 429s opened the breaker and every run completed on
  the degraded path. The metrics showed it, but we had not checked `degraded` before reading the result.
* **Fix.**
  * The gateway now paces itself to `LLM_MAX_RPM`, using a slot reservation in Redis shared by all replicas. A call
    that would wait longer than `LLM_RATE_WAIT_MAX_S` gets 503 and the agent falls back, instead of the provider
    returning 429 and opening the breaker.
  * The gateway also logs the provider's response body, so a 429 or 401 is visible without guessing.
  * The model ids in `.env.example` were updated: `gemini-2.0-flash` now returns 404 ("no longer available").

### INC-17: A real model spent the research budget, so recovery never ran

* **Symptom.** With Gemini, the 4.1 case escalated. The model's draft (decline, INR 0, which is correct) cited the
  airline charter instead of CAR 3.3.2. Verification failed, and the run went straight to fallback. The fallback
  then failed `citations_resolve`, because the grievance clause 3.9.2 had never been retrieved.
* **Confirmation.** The Gemini planner used all 12 tool calls. `after_verify` only allowed recovery while budget
  remained.
* **Fix.** Recovery and fallback now have a separate hard allowance of 6 exact `get_clause` lookups
  (`RESERVED_LOOKUPS`). Recovery runs even when research spent its budget. Fallback fetches the provisions the
  engine's position rests on before verifying it. After the fix, the same claim gets a second model attempt and then
  the verified engine position: decline, INR 0, meals owed.

### INC-18: The daily free-tier quota ended the prompt comparison after one run

* **Symptom.** The second full Gemini evaluation (determination prompt v1) finished in under 5 minutes with 27/27
  correct, and 26 of its runs were marked `degraded`.
* **Confirmation.** The gateway logged HTTP 429 with quota id `GenerateRequestsPerDayPerProjectPerModel-FreeTier`,
  limit 500. One full evaluation makes about 390 model requests, and our probes earlier in the day used the rest.
  The breaker opened after five 429s, and the remaining runs finished on the engine before its 30 s cool-down ended.
* **Fix.** We discarded that run. The prompt comparison now runs both versions on the same 13-scenario subset, which
  fits in one day's quota. The eval summary lists `degraded_runs` and prints a warning, so a run that did not use the
  model cannot be mistaken for a model result again (this is the second time, after INC-16).

### INC-19: Gemini answers the verifier accepted but the eval marked wrong

* **Symptom.** With Gemini, 3 of 27 answers had the right route but were scored wrong. In S18 (5-hour delay) the
  model declined compensation, which is correct, but listed no facilities, although 3.4.1 requires meals. In S09
  and S23 the model escalated correctly, but its free-text reasons lost the escalation category.
* **Confirmation.** `consistent_with_rules` compared only route and amount with the engine. Facilities were never
  compared, so a model answer that dropped an owed facility passed. The stub never produced such an answer, so the
  gap only showed with a real model.
* **Fix.** `consistent_with_rules` now fails a model answer that omits a facility the engine finds owed (meals,
  hotel), so recovery runs and, if needed, the engine's position is returned. A model escalation now carries the
  engine's fixed-wording reasons first. Test: `test_verifier_rejects_answer_that_drops_owed_facilities`.

### INC-20: The web client showed an answer computed before the corpus changed

* **Symptom.** After re-ingesting the official CAR, the web client's 4.1 case returned in 0.0 s with "served from
  whole-run cache", and its 3.3.2 quote was not the one the current code selects.
* **Confirmation.** The run cache key covered the claim, prompt versions, agent modes and model id, but not the
  corpus or the rule map. A new CAR would have kept serving old answers for the 24-hour TTL. `architecture.md`
  already said the index version was in the key; it was not.
* **Fix.** The API adds the retriever's live index version (refreshed every 30 s) and a hash of `car_rules.yaml` to
  the key. If the retriever cannot report its version, the run is not cached.
