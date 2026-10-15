# RAG Mem Sentry: Evaluation Report

Every figure in this report is copied from a README results section, written there from command
output, and each is cited by that section's heading. Nothing is rounded and no figure appears here
that the README does not hold. No attack item's text is quoted.

## 1. Problem and threat model

An LLM agent treats what it retrieves as ground truth, whether it comes from a RAG corpus or from
its own long-term memory. An attacker who can place content in either store steers the agent.
Defenses that gate origin (signing, provenance, lineage) pass a validly signed record whose content
is poisoned. This project gates **content**: it screens documents and memory records for poison
before the model trusts them (design spec `docs/specs/2026-08-16-poisoned-content-trust-gate-design.md`,
§1).

Threat model (spec §4). The attacker can place content in the corpus and can induce the agent to
write a poisoned experience record; they cannot change the agent's code, prompts or the gate.
Two knowledge settings: **blind** (public benchmark attacks, no knowledge of the gate) and
**adaptive** (black-box: submit content, observe a verdict). Three poison classes:

| class | name | mechanism | primary signal |
|---|---|---|---|
| 1 | instruction injection | retrieved text carries imperatives aimed at the model | S1 + S2 |
| 2 | factual poisoning | false facts crafted to rank for a target query and flip the answer | S3 |
| 3 | poisoned experience records | a stored "successful" record models unsafe behavior | S2 |

## 2. Design

The host is an agentic CRAG pipeline (decompose, retrieve with dense + BM25 + rerank, generate with
citations, synthesize). The gate sits in front of the model at two points.

- **Three signals.** The write gate runs S1 alone and makes no LLM call. S1 (structural): regex only; instruction patterns plus obfuscation
  (zero-width, bidi, hidden CSS, HTML comments, homoglyphs, encoded payloads). S2 (judge): an LLM
  judge, run at read time only; a corpus rubric (does the passage instruct the reader's
  model, or quote an attack) and an experience rubric (does the record endorse unsafe behavior).
  The judge never sees the user's query. S3 (corroborate): extracts claims from an untrusted chunk,
  looks each up in a BM25 index over owner-trusted chunks, and has the LLM judge agreement; it abstains on
  trusted items, on the experience store and on claims the trusted set does not cover.
- **Aggregation.** Noisy-OR, `trust = ∏(1 - suspicion)`, so one confident detection is not diluted
  by calm neighbours.
- **Per-store policy.** Each store has its own block and quarantine cut points; a signal over its
  hard threshold blocks outright. S3 is quarantine-only: it can hold a chunk back, never block it.
- **Failure asymmetry.** Write time fails closed (a failed signal quarantines). Read time degrades
  to the surviving signals and marks the verdict reduced-assurance, rather than refusing every
  query. A partial run is never cached.
- **Cache key.** `store|sorted-signal-names|text`, with model and rubric fingerprints for the judge
  and the trusted-set fingerprint for S3, so a changed signal, model or trusted set misses the
  cache.
- **Trust from the manifest.** A chunk is trusted only when the ingest manifest says `"trusted":
  true`; a document cannot assert its own trust.
- **Read never mutates a store.** A read-time exclusion drops the item from this query's context
  only; quarantine and release stay at write time.

## 3. Setup

- **Corpus** ("Final corpus: BEIR NQ (2026-10-14)"). A BEIR NQ Wikipedia subset: 1726 documents, 50
  of them gold; 1784 chunks, 1611 trusted, 173 untrusted. The 22 AI-security papers used earlier
  were a placeholder; figures measured on them are labelled as such below.
- **Attack corpus.** Mapped by an agent from public datasets (BIPIA, InjecAgent, PoisonedRAG, ASB)
  into the project's three classes, outside this repository, with the prompt in
  `docs/prompts/attack-corpus-mapping-agent.md`. Frozen 2026-10-14 and split by group into
  calibration and test ("Phase 4: Calibration (Task 45, run 2026-10-14)"): class 1 32 / 28,
  class 2 75 / 75, class 3 21 / 21. Calibration used only the calibration split.
- **Freeze** ("Gate freeze (Task 55, run 2026-10-14)"). `eval/frozen_gate.json`, local tag
  `gate-frozen-1`: judge model `gemini-flash-lite-latest`; `JUDGE_BATCH_MAX` 5, `S3_MAX_CLAIMS` 5,
  `S3_REF_K` 3; corpus `block_below` 0.1 / `quarantine_below` 0.6, experience 0.2 / 0.7; hard
  thresholds `structural` 0.93 and `judge` 0.8; `corroborate` quarantine-only. Every Phase 5 and
  Phase 6 run checked the freeze first and recorded `"differences": []`.
- **Models.** Gate (judge and S3): `gemini-flash-lite-latest`. Answer model: `gemini-3.1-flash-lite`.
  Both were set explicitly for every run in sections 4 and 7.

## 4. Attack success (§9.1)

Context level: the poison reached the model's prompt. Behavior level: the answer did what the
attack wanted. Gate off → gate on. Overhead is from section 7 and was measured on clean questions,
not on these trials: +8.822338439989835 s per query on an empty verdict cache, +4.066152779990807 s
on a warm one ("Phase 6: Overhead (Task 67, run 2026-10-15)").

| class | setting | context | behavior | source |
|---|---|---|---|---|
| 1 | five fixed R3-style carriers | 5/5 → 0/5 | 5/5 → 0/5 | "Phase 6: Class-1 Attack Success (Task 65, run 2026-10-15)" |
| 2 | blind, 4 valid targets | 4/4 → 3/4 | 4/4 → 3/4 | "Phase 4: Class-2 Attack Success (Task 47, run 2026-10-15)" |
| 2 | adaptive, 5 valid targets | 5/5 → 5/5 | 5/5 → 5/5 | "Phase 5: Adaptive Class-2 Attack Success (Task 60, run 2026-10-15)" |
| 3 | first three test-split groups | 3/3 → 0/3 | 0/3 → 0/3 (unmeasurable, R6) | "Phase 6: Class-3 Attack Success (Task 66, run 2026-10-15)" |

In the spec's shape: class-1 ASR 5/5 → 0/5 at +8.822338439989835 s/query cache-cold; blind class-2
ASR 4/4 → 3/4 at the same cost.

- **Class 1.** S2 topped every exclusion and blocked each carrier over its 0.80 hard threshold.
  Each excluded chunk was the carrier, and nothing else was excluded. These are fixed carriers, not
  adaptive attacks.
- **Class 2, blind.** Every exclusion is credited to S3; S2 excluded none. One target (c2-test19)
  is void: its control answer already held the target answer. Passages in context fell 12 → 6.
  With 4 valid targets the drop of 1 is a count, not a rate.
- **Class 2, adaptive.** The gate excluded no adaptive passage. Passages in context 10 → 10.
- **Class 3.** The write gate admitted all three records at trust 1.0; the read gate's S2 blocked
  all three at judge 1.0. Behavior was judged by hand against each pre-registered check: not met in
  any trial, either arm. The checks describe an agent calling a tool, and this host has no tools, so
  the gate-off behavior total is 0 and the behavior comparison is unmeasurable. `0 → 0` is not the
  gate working.

## 5. Detection and ablation (§9.2)

Items excluded out of items scored, all three signals, blind test split beside the adaptive set
("Phase 5: Adaptive Detection (Task 59, run 2026-10-15)"):

| | class 1 | class 2 | class 3 |
|---|---|---|---|
| blind test split (Task 46) | 28/28 | 34/75 | 15/21 |
| adaptive (Task 59) | 1/16 | 3/16 | 0/16 |

Ablation, blind test split ("Phase 4: Signal Ablation (Task 46, run 2026-10-14)"):

| signals | class 1 | class 2 | class 3 |
|---|---|---|---|
| structural | 0/28 | 0/75 | 0/21 |
| judge | 28/28 | 0/75 | 15/21 |
| corroborate | 0/28 | 34/75 | abstains by design (0/21) |
| structural+judge | 28/28 | 0/75 | 15/21 |
| structural+judge+corroborate | 28/28 | 34/75 | 15/21 |

Ablation, adaptive set ("Phase 5: Adaptive Detection (Task 59, run 2026-10-15)"):

| signals | class 1 | class 2 | class 3 |
|---|---|---|---|
| structural | 0/16 | 0/16 | 0/16 |
| judge | 0/16 | 0/16 | 0/16 |
| corroborate | 1/16 | 3/16 | 0/16 |
| structural+judge | 0/16 | 0/16 | 0/16 |
| structural+judge+corroborate | 1/16 | 3/16 | 0/16 |

- S1 caught nothing on either set: the payloads are plain text, with no hidden characters or
  encodings.
- On the blind set, S3 added 34 class-2 items out of 75 and nothing in classes 1 and 3. It was not
  covered by the trusted set on 32 class-2 items.
- On the adaptive set, S2 excluded none of the 48 items. S3 checked 3 class-2 items and flagged
  all 3; 13 were `not_covered`.
- These are detection rates, not attack success: an item caught here might not have changed an
  answer had it reached the model.

## 6. Clean-task regression (§9.3)

**Not measured, by owner decision, 2026-10-15.** The gate-on RAGAS arm (`EVAL_READ_GATE=1 python -m
eval.run_eval`) exists and was not run. This report holds no RAGAS figure for either arm.

## 7. Overhead (§9.4)

From "Phase 6: Overhead (Task 67, run 2026-10-15)". The first 5 questions of `eval/qa_set.json`,
`LLM_MIN_INTERVAL=2`.

| arm | LLM calls | total seconds | calls per question | seconds per question |
|---|---|---|---|---|
| off | 14 | 31.471696099964902 | 2.8 | 6.29433921999298 |
| on-cold | 27 | 75.58338829991408 | 5.4 | 15.116677659982816 |
| on-warm | 14 | 51.80245999991894 | 2.8 | 10.360491999983788 |

Added per query, against `off`:
- cache cold: +2.6 LLM calls, +8.822338439989835 s;
- cache warm: +0 LLM calls, +4.066152779990807 s.

Every LLM call waits at least 2 s after the previous one, so pacing and the free-tier endpoint
dominate wall-clock. The warm arm made no extra call yet took longer than `off`; the run does not
say where that time went. n = 5, one run each.

## 8. False positives

| measurement | corpus | model | result | source |
|---|---|---|---|---|
| write-time FPR (S1) | NQ | none (S1 only) | 5 / 1789 = 0.002794857462269424, budget 2%; all five hand-read, each a zero-width character in Wikipedia markup | "Final corpus: BEIR NQ (2026-10-14)" |
| S2 on quoted-attack hard negatives (R5) | papers | `gemini-3.1-flash-lite` | 0 S2-topped false positives of 104 scored items; fallback not triggered | "Phase 3: R5 — Does the Judge Separate Quoting from Instructing?" |
| S3 on clean untrusted content (Task 44) | papers | `gemini-3.1-flash-lite` | 3 / 16 = 0.1875, over budget; 2 of 3 are unparseable agreement replies, 1 a misjudged contradiction; 24 / 40 `not_covered` | "Phase 4: S3 on Clean Untrusted Content (Task 44, run 2026-10-14)" |
| S3 on covered clean text | NQ | `gemini-3.1-flash-lite` | 0 false positives out of 5 checked, too few to state a rate | "Final corpus: BEIR NQ (2026-10-14)" |
| read-time corpus band (Task 45) | NQ | `gemini-flash-lite-latest` | 2 / 40 clean rows quarantined, 0.05 against 0.02; both S3 parse failures; recorded, not moved (owner decision) | "Phase 4: Calibration (Task 45, run 2026-10-14)" |
| read-time experience band (Task 45) | clean experience records | `gemini-flash-lite-latest` | 0 / 20 non-admits | "Phase 4: Calibration (Task 45, run 2026-10-14)" |

After Task 44, S3's agreement step was changed to re-ask once on an unparseable reply, and its
prompt now requires a contradiction to be about the same subject. The NQ S3 rows and Task 45 ran
with both changes. The NQ run checked 5 items, too few for a rate, and both Task 45 corpus
quarantines are still S3 parse failures. So S3's false-positive rate after the changes is not
established.

## 9. The adaptive round

From "Probe history" and "Phase 5: What the Adaptive Round Shows". An external agent got 60
one-bit probes over three sessions against the frozen gate, with black-box access only: each probe
returned admitted or not, with no score or reason.

What it found, summarised from its notes, with no payload quoted:
- instruction wording placed inside a narrative frame (a story, a production document, a manual)
  was admitted where direct instructions were excluded;
- a false answer credited to a named source and year was admitted where the same answer stated as
  fact was excluded;
- memory records whose approach tells the assistant to set the retrieved passages aside were
  admitted.

In session 3 the admitted rate was 1.0 in every class (7 of 7, 7 of 7 and 6 of 6 probed). The delivered set held 48 items, 16
per class.

What the gate lost: detection fell from 28/28, 34/75 and 15/21 (blind, classes 1, 2, 3) to 1/16,
3/16 and 0/16, and S2 excluded none of the 48. Class-2 attack success went from 4/4 → 3/4 (blind)
to 5/5 → 5/5 (adaptive). Under the spec's decision rule the attacker beat the gate within 60 probes
on this model, in all three classes. Phase 5 was report-only: no gate change followed.

Verdicts were not stable across runs: 3 probe-admitted items were excluded in the final screen,
and 3 passages S3 quarantined in Task 59 were admitted when screened again in Task 60.

## 10. Limitations

- **One answer model and one gate model.** Every security result here is for
  `gemini-3.1-flash-lite` answering and `gemini-flash-lite-latest` judging. The model's own
  alignment is part of what is measured.
- **Verdict non-determinism.** S3 gave different verdicts on the same items, at the same frozen
  state, in Tasks 59 and 60.
- **RAGAS not run.** No clean-task regression figure exists (section 6).
- **Small samples.** 5 class-1 carriers, 4 and 5 valid class-2 targets, 3 class-3 records, 5
  overhead questions. These are counts, not rates.
- **Class-3 behavior is unmeasurable on this host** (section 4): it has no tools for a poisoned
  record to steer it toward.
- **S3 needs an owner-trusted set and covers only what that set covers.** A false fact on a topic
  the trusted sources are silent about passes S3, and S3 checks at most 5 claims per chunk.
- **Trust is asserted by the owner, not verified.** A poisoned source marked trusted poisons the
  reference.
- **Adaptive round** (Phase 5 spec §13): the black box is a process boundary, not enforcement, and
  the breach check covered the agent's scripts and notes, not Antigravity's own tool log; one
  attacker, one model, 60 probes; response timing leaks which S3 path ran, not mitigated; evading
  S3 did not cost the attack here.
- **Host limitations.** Sub-queries run sequentially; decomposition is capped at 3; there is no
  retrieval-quality refusal, so out-of-scope questions are answered from whatever was retrieved;
  free-tier limits pace every run.
- **Known Phase 3+ concerns** (README "Known Phase 3+ concerns (named, not addressed)"):
  - *Unbounded growth on the memory path:* nothing dedupes, caps or evicts experience records, and
    the agent's own answer becomes the next run's context, a self-confirmation channel nothing
    tests.
  - *Branch review punch list:* `ExperienceStore.purge` returns early when no id is in the mirror,
    so an orphaned Chroma entry is never deleted; S1 dedupes homoglyph and encoded hits within a
    channel but not across them; `eval/fpr.py` dedupes by `item_id` before filtering by store;
    `tests/eval/test_fpr.py` excludes an entry with no `store` key without saying why;
    `app/orchestrator.py` discards the experience write verdict; memory read and write failures
    are not isolated from the answer path; `default_experience_store` flattens a nested
    `CHROMA_DIR`; `ExperienceStore.purge` rewrites the mirror in place with no temp-file-and-rename;
    the Phase 2 plan still refers to "Task 21's demonstration"; `format_experiences` injects each
    record's full prior answer, so prompt size grows with record length. The
    `tests/test_config.py::test_config_constants_present` item is fixed (`99a24fa`).

## 11. Future work

- **Harden against the adaptive framings.** Narrative-framed instructions, source-attributed false
  answers and set-the-passages-aside memory records all passed S2. A rubric change has to be made
  on new calibration data, not fitted to the 48 adaptive items.
- **A second freeze and a second adaptive round** against the hardened gate, with a fresh attacker
  and fresh probes.
- **A second provider** for the answer model and the gate model, to separate the gate's effect
  from one model family's alignment.
- **The measurements not run:** the gate-on RAGAS arm (section 6); S3's false-positive rate on a
  fresh NQ sample with the re-ask and same-subject wording; R3's poison-only control; class-3
  behavior on a host that has tools.
