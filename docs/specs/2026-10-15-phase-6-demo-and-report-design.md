# Phase 6 — Demo and Report — Design Spec

**Date:** 2026-10-15
**Status:** Approved, pre-implementation
**Parent spec:** `docs/specs/2026-08-16-poisoned-content-trust-gate-design.md` (§7 trust panel, §9 the four measurements, §10 weeks 11–12, §11 risks)
**Previous phase:** `docs/specs/2026-10-14-phase-5-adaptive-round-design.md`
**Roadmap row:** Phase 6 in `docs/plans/2026-08-16-trust-gate-phase-1.md`. That table's task
numbers (31–34) are stale; this phase is Tasks 62–73.
**Models:** gate model (S2 and S3) `gemini-flash-lite-latest`, the frozen one recorded in
`eval/frozen_gate.json`. Answer model for every run in this phase `gemini-3.1-flash-lite`, the one
Tasks 47 and 60 used. Every figure names both. The Phase 4 and Phase 5 specs name
`gemini-3.1-flash-lite` as the gate model. That is wrong as executed, and the correction is one of
the cleanup items the owner decides on at the end of the phase (Task 71).

---

## 0. Owner decisions this phase is built on

All taken by the project owner on 2026-10-15.

1. **The gate stays frozen at `gate-frozen-1`, and the adaptive loss is reported as it stands.**
   No hardening task. Hardening ideas the adaptive round points to (an S2 rubric that reads
   narrative framing as instruction, a rubric that treats attribution to a named source as no
   evidence of truth) go in the report's future-work section. Every run in this phase happens at
   the frozen state, so every Phase 4, 5 and 6 number describes one gate.
2. **The full evaluation run is three harnesses, all built in Phase 3 and never run:** the
   class-1 trial (gate off and on), the class-3 trial (`--arm off` and `--arm on`, both re-run so
   both arms sit on the NQ host), and the overhead harness (off, on-cold, on-warm). **RAGAS is not
   run.** Parent spec §9.3 (clean-task regression) is reported as not measured, by owner decision.
3. **Answer model `gemini-3.1-flash-lite`; gate model `gemini-flash-lite-latest`, set explicitly
   with `JUDGE_MODEL`.** Day order: class-1 trial, then class-3 trial, then overhead. One
   LLM-spending run per day.
4. **The trust panel reads the audit log.** `app/ui.py` notes the audit log's length before a
   query and renders the `stage: "read"` entries appended during it. The read gate is opt-in
   behind `UI_READ_GATE=1`, default off, and uses the real data directory, so the prewarmed NQ
   cache serves the demo. `gate/`, `app/graph.py` and `app/orchestrator.py` do not change.
5. **The report is `docs/report.md`,** one Markdown file for capstone examiners and a technical
   recruiter. A PDF or a college-template version can be exported from it later; that export is
   not in this phase.
6. **The demo shows a poisoned experience record excluded live.** Memory is opt-in behind
   `UI_EXPERIENCE_STORE=1`, default off. A blind class-3 record the Task 46 screen excluded is
   planted through the real write gate before recording and purged after it, together with every
   record the demo's own runs wrote. The owner records the demo.
7. **Branch:** Phase 6 docs and build stay on `phase-4-s3-corroboration`, as Phase 5 did. The last
   task merges it to `main` and pushes the branch and the `gate-frozen-1` tag to
   `sk-0607/rag-mem-sentry`, each behind a STOP for the owner's go.
8. **Cleanup is decided by the owner at the end of the phase, not before.** Stale README text,
   the model note in the Phase 4 and 5 specs, the Phase 1 roadmap numbers, the working tree's
   `.env.example` and `.gitignore` changes, and the untracked `PHASE1_RESULTS.md`,
   `PHASE3_RESULTS.md` and `eval/experience_poison_trial.py.pre-merge-backup` are listed as
   candidates in Task 71. The owner picks which are done, and nothing on the list is touched
   before that. The proposal recorded on 2026-10-15 is in §11. The "Known Phase 3+ concerns"
   punch list is not fixed; it is cited in the report's limitations.
   *Amended 2026-10-15, after the first commit of this spec:* the first version put the
   README and spec reconcile and the two config commits in a docs task before the runs.
9. **Tasks are numbered 62–73.** The stale 31–34 in the Phase 1 plan's roadmap table is a
   Task 71 candidate.
10. **The trust panel's visual polish uses the `impeccable` skill at build time,** limited to the
    panel and to Gradio's own CSS hooks, with no new dependency.

Standing decisions carried unchanged: every run that spends LLM budget needs the owner's explicit
go; `read_gate=None` stays the default and is the gate-off arm; read never mutates a store; trust
comes only from the ingest manifest; no hardcoded source ids in anything new; no trained
classifiers; numbers quoted unrounded from command output; commit by pathspec, no Claude or
Anthropic attribution; never push to `self-correcting-rag`.

## 1. Scope

The parent spec's weeks 11–12: the trust panel, the full evaluation run, the evaluation report,
and the README, demo recording and repo cleanup. The gate itself is finished; this phase measures
what has not been measured, shows the gate working, and writes it up.

## 2. Goals

- Every run in this phase is refused unless the freeze check is clean, and records the check.
- Gate-off vs gate-on attack success for classes 1 and 3, at both levels (parent §9.1), beside the
  class-2 figures Tasks 47 and 60 already hold.
- Overhead per query, gate off, gate on cache-cold, gate on cache-warm (parent §9.4).
- A trust panel in the Gradio UI showing, per screened candidate, the action, trust, firing
  signal and evidence span (parent §7).
- A recorded demo in which a poisoned memory record is excluded live, and a clean store after it.
- `docs/report.md`: all measured results, blind and adaptive, with limitations and future work.
- An owner decision on every cleanup candidate, and a merged, pushed branch.

## 3. Non-goals

- **Any change under `gate/`.** The gate is frozen (decision 1).
- **Any change to `app/graph.py` or `app/orchestrator.py`.** Every evaluation harness runs
  through them; the panel reads the audit log instead (decision 4).
- **RAGAS** (decision 2).
- **Hardening against the adaptive framings** (decision 1).
- **Fixing the "Known Phase 3+ concerns" punch list** (decision 8).
- **A second adaptive round, a second freeze, or another answer model.**
- **Exposing the panel anywhere but the owner's local UI.** The probe endpoint
  (`eval/probe_server.py`) is not touched and never returns panel data.
- **A PDF or college-format report.**
- **Trained classifiers.** Standing constraint.

## 4. Approach

Three choices, each with the alternative it beat.

- **Panel data from the audit log, not from graph state.** `ReadGate.screen_batch` writes one
  audit entry per item on every call, cache hits included, with `action`, `trust`, `reason` and
  each signal's `score`, `reason` and `evidence_span`. The gate node keeps only the excluded ids.
  *Rejected:* adding a `gate_verdicts` key to `RAGState` and passing it out through
  `run_agentic`. It is exact per query, but it changes host code that every evaluation harness
  runs through, in the same phase those harnesses are run. The parent spec already names the
  audit log as the demo surface.
- **The live exclusion uses a class-3 experience record, planted through the real write gate.**
  The write gate is S1 only and makes no LLM call, so planting is free; the read gate's S2 then
  excludes the record at query time. Cleanup is `ExperienceStore.purge`, the store's existing
  removal path. *Rejected:* a class-2 false-fact chunk added as an untrusted manifest entry. S3's
  verdicts on the same item differed between Tasks 59 and 60, and planting and removing it each
  need a full re-ingest. *Rejected:* no live exclusion. The demo would show only admits, because
  the prewarmed NQ demo set admitted all 49 chunks.
- **The freeze guard is added to the three remaining harnesses rather than checked by hand.**
  Phase 5 put it in `eval/attack_screen.py` and `eval/class2_trial.py`. A run whose results file
  carries the freeze check cannot be quoted against a different gate by mistake.

## 5. Freeze guard on the remaining harnesses (Task 62)

`eval/class1_trial.py`, `eval/experience_poison_trial.py` and `eval/overhead.py` each gain the
Phase 5 pattern from `eval/attack_screen.py` and `eval/class2_trial.py`:

- `require_frozen()` runs before the read gate is built and before any LLM call. A
  `FreezeError` exits non-zero with its message, which lists every difference, and spends
  nothing.
- The results file gains `"freeze": {"record": <FROZEN_GATE as posix if it exists, else null>,
  "differences": <check(...) output>}`, the shape the Phase 5 harnesses write.
- The guard is enforced, not only recorded. Every run in this phase is at the frozen state
  (decision 1), so there is no run for which recording alone is right.

Two harness facts the plan must pin by reading the code. The code graph holds neither, because it
has no module constants or call arguments:

- **The real reference in scratch directories.** The standing invariant is that a harness with
  its own scratch directory passes the real reference. All three harnesses build their read gate
  in a scratch directory, by Phase 3 design: `class1_trial` in a wiped `data/class1_trial`,
  `experience_poison_trial --arm on` in its `TRIAL_DIR`, and `overhead` in a fresh temp
  directory. All three predate S3. Without `reference=default_reference(<real data dir>)`, S3
  silently abstains on every corpus chunk, and the cold overhead arm under-reports S3's cost.
  - `eval/overhead.py` imports `default_read_gate` but not `default_reference`.
  - `eval/class1_trial.py` imports both; Task 62 confirms that it passes the reference.
  - `eval/experience_poison_trial.py` needs the reference too. S3 abstains on experience
    records by design, but `run_agentic` also screens the question's corpus chunks.

  Task 62 makes each harness pass the real reference. Each refuses to start on an empty trusted
  set, as `class2_trial` does.
- **The class-3 trial's questions.** The trial has three `TRIALS` against thin-coverage
  questions, with behavior judged by hand against a pre-registered `check` (Phase 3 plan,
  Task 29). Its `--arm off` baseline ran on 2026-09-16, on the papers corpus. CLAUDE.md records
  that the class-1 trial questions moved to NQ and does not say the class-3 trial's did. If the
  trial's questions or poison records name papers-era content, Task 62 stops and the owner
  decides: re-target them to NQ, or run as-is and report them as papers-era questions on the NQ
  host. The plan does not make that choice.
- **The class-3 trial's output path.** README says `data/poison_trial_results/results-<arm>.json`;
  the Phase 3 plan says `data/poison_trial/results-<arm>.json`. Task 62 reads the module, and
  the run task quotes the path it actually writes.

## 6. Trust panel (Task 63)

### 6.1 Wiring

- `app/config.py` gains `UI_READ_GATE` and `UI_EXPERIENCE_STORE`, read from the environment,
  both default off. These are UI wiring, not gate knobs. The freeze records only
  `JUDGE_BATCH_MAX`, `S3_MAX_CLAIMS` and `S3_REF_K`, so they do not touch it.
- When `UI_READ_GATE` is on, the UI's runner passes `read_gate=default_read_gate(<real data dir>)`
  to `run_agentic`, with the reference that function builds by default. When `UI_EXPERIENCE_STORE`
  is on, it also passes `store=default_experience_store(<real data dir>)`. When both are off, the
  runner is byte-for-byte today's: `read_gate=None`, no store.
- `build_demo(runner=)` keeps its injectable runner. The flags only choose the default runner, so
  `tests/test_ui.py` keeps using fakes.

### 6.2 Data

- **Offset.** Before each query the UI records the read gate's audit file size in bytes, or 0 if
  the file is absent. After the query it reads the lines past that offset and keeps the entries
  with `stage == "read"`. Write-stage entries from the experience write that `run_agentic` makes
  at the end of a run are not shown as candidates.
- **Text join.** A corpus entry's text comes from `chunks.jsonl` by `item_id`. An experience
  entry's text comes from the store's JSONL mirror by record id. An id with no text shows the id
  alone and does not fail the query.
- **Rows.** One per audit entry: store, item id, action, trust, top signal (the signal with the
  highest score), that signal's reason, its evidence span, and the candidate text with the
  evidence span highlighted where it occurs verbatim. Where it does not occur verbatim (S1's
  decoded-payload hits, for example), the span is shown beside the text instead.
- **Gate off.** No read entries are appended, and the panel says the read gate is off. It never
  shows an empty table that looks like "nothing was screened".
- **Escaping.** Candidate text, reasons and evidence spans are HTML-escaped before any panel
  markup is added. Class-1 poison uses hidden CSS and HTML comments. Rendered unescaped in the
  owner's browser, it would hide or style itself in the one place built to expose it.
- **Offset past the end.** If the audit file is shorter than the recorded offset (`data/` was
  wiped between the two reads), the panel reads from the start rather than showing nothing. A
  half-written trailing line is skipped, not raised.
- **Flag values.** Only the exact string `1` turns a flag on. `0`, `false` and an empty value are
  off.

### 6.3 Boundaries

- The panel reads files. It writes nothing and calls no gate method.
- Single user. Two queries running at once interleave their audit entries. That is stated as a
  limitation, not guarded against: the demo is one person's local UI.
- The panel is not served by `app/api.py` or `eval/probe_server.py`. It exposes per-signal scores
  and reasons, which the probe endpoint is built to withhold.
- Visual polish uses the `impeccable` skill at build time (decision 10), panel only.

## 7. Demo memory and plant tool (Task 64)

New module `eval/demo_plant.py`, CLI `python -m eval.demo_plant <command>`. It names no item, id
or question; the owner supplies the id.

- **`candidates`** is offline. It reads the frozen blind attack corpus (hash-checked by
  `eval.attack_corpus`'s loader) and the Task 46 screen results
  (`data/phase4_screen/test/results.json`). It prints the class-3 test-split items the screen
  excluded with `judge` as the top signal, with each one's target question.
- **`plant --item <id>`** is free: the write gate makes no LLM call.
  - It refuses when `data/demo_plant.json` exists (one plant at a time), when the id is not a
    class-3 test-split item, or when the freeze check is not clean.
  - It builds the `ExperienceRecord` exactly as the class-3 screen builds its item, reusing that
    projection rather than writing a second one. If `eval/attack_corpus.py` builds the record
    inline inside its item projection, the record step is extracted into its own function, and
    the projection keeps producing byte-identical item text. That text is part of the cache key.
  - The projection gives the record fixed field values, `ts` included, so the record's own `ts`
    does not say when it was planted. The plant time is recorded separately, minted the same way
    `record_from_run` mints `ts`, so the two compare correctly.
  - It writes through `ExperienceStore.write(record, gate=default_experience_gate(<real data dir>))`.
    A non-admit verdict persists nothing, prints the verdict and exits 1.
  - On admit it writes `data/demo_plant.json` with the attack item id, the record id, the plant
    timestamp and the target question.
- **`purge`** is destructive and free.
  - With no `data/demo_plant.json`, it exits 1 with "nothing planted" and changes nothing.
  - It reads `data/demo_plant.json`. It lists the planted record id plus every record in the
    store's mirror with `ts` at or after the plant timestamp. Those are the demo's own runs,
    because `run_agentic` writes one gated record per run when a store is attached.
  - Without `--yes` it prints the list and changes nothing. With `--yes` it copies the JSONL
    mirror to a timestamped backup first, then calls `ExperienceStore.purge(ids)`, then
    removes `data/demo_plant.json`.
  - The backup exists because `purge` rewrites the mirror in place with no
    temp-file-and-rename, a known concern this phase does not fix (decision 8).
  - Afterwards it re-reads the mirror and exits 1 if any listed id is still there. The
    store's known early return when no id is present in the mirror makes this check worth
    having.

The demo question is the planted item's target question. Experience retrieval embeds on task
text, so that question is the one most likely to retrieve the record.

## 8. Full evaluation run (Tasks 65–67)

Every run sets `LLM_MODEL=gemini-3.1-flash-lite`, `JUDGE_MODEL=gemini-flash-lite-latest` and
`LLM_MIN_INTERVAL=2`. Each opens with a STOP for the owner's go, then a clean freeze check. A
`PerDay` quota error ends the day's run: its results file stays `complete: false`, and the run
resumes the next day only on a new go. A `PerMinute` error is the harness's own backoff and is
not a stop.

| Task | Day | Command | Reports | Writes |
|---|---|---|---|---|
| 65 | 1 | `python -m eval.class1_trial` | Per trial, gate off and gate on: carrier in `generate`'s docs (context), canary in the answer (behavior), excluded chunks and their top signal, the carrier's audit entry. Totals per arm at both levels. | `data/class1_trial_results/results.json` |
| 66 | 2 | `python -m eval.experience_poison_trial --arm off`, then `--arm on` | Per question: poison retrieved, excluded, reached the prompt, the answer; the ungated control arm. Totals per arm at both levels. Both arms on the same day under one go; if the quota ends the day between them, `--arm on` waits for a new go. | `results-off.json`, `results-on.json` under the directory the module writes (§5) |
| 67 | 3 | `OVERHEAD_N=5 python -m eval.overhead` | Per question and in total: LLM calls and wall-clock for `off`, `on-cold` (empty verdict cache), `on-warm`. Added latency and calls per query, cold and warm separately. | `data/phase3_overhead/results.json` |

Overhead runs last: it is the cheapest, and its cold arm starts from its own empty cache, so the
earlier runs cannot warm it. Each run task ends by writing its README section from the command
output, unrounded, in the shape of the Phase 4 and 5 result sections, with both models named.

## 9. Demo (Tasks 68–69)

**Task 68 (STOP, spends budget).**
1. `candidates`. The owner picks an id.
2. `plant --item <id>`.
3. A rehearsal query of the target question in the UI, with both flags on. The rehearsal screens
   the planted record and the question's chunks once. A complete run caches the verdicts, so the
   recording replays them without a gate LLM call.
4. If the rehearsal admits the planted record, `purge --yes` and pick another candidate. LLM
   verdicts are not deterministic, as Phase 5 limitation 4 records.

The rehearsal also writes one experience record of its own; `purge` covers it.

**Task 69 (owner records; then STOP for the purge).**
1. The owner records with both flags on: the target question, the answer, and the panel showing
   the planted record excluded and its S2 evidence. Then one or two prewarmed questions from
   `eval/demo_questions.txt` with all chunks admitted.
2. After recording, `purge` without `--yes` to show the list, then `purge --yes`.
3. Exit check: the store's mirror holds none of the listed ids, and `data/demo_plant.json` is gone.

Where the recording is kept is the owner's choice. It is not committed.

## 10. Report — `docs/report.md` (Task 70)

Written after Tasks 65–69. Every number in it is copied from a README results section that was
written from command output. The report cites that section. It introduces no number the README
does not hold and rounds nothing.

Sections:
1. **Problem and threat model.** Parent §1 and §4, short.
2. **Design.** Three signals, noisy-OR aggregation, per-store policy, write-time vs read-time
   failure asymmetry, cache key, trust from the manifest.
3. **Setup.** NQ corpus, attack corpus provenance (public datasets, mapped by an agent), blind
   calibration and test splits, the freeze, both models.
4. **Attack success (§9.1).** Per class, context and behavior, gate off → gate on, in the parent
   spec's shape "ASR X → Y at +Z ms/query" where overhead allows it. Class 1 from Task 65,
   class 2 from Tasks 47 (blind) and 60 (adaptive), class 3 from Task 66.
5. **Detection and ablation (§9.2).** Blind test split (Task 46) beside adaptive (Task 59), and
   both ablation tables.
6. **Clean-task regression (§9.3).** Not measured, by owner decision, 2026-10-15. Stated here, so
   the absence is visible.
7. **Overhead (§9.4).** Task 67, cold and warm separately.
8. **False positives.** Write-time FPR, S2 on quoted-attack hard negatives (R5), S3 on clean
   untrusted content (Task 44), the read-time corpus band over budget (Task 45).
9. **The adaptive round.** What the attacker found, in summary, with no payloads quoted. What
   the gate lost.
10. **Limitations.** From README "Limitations", Phase 5 spec §13, "Known Phase 3+ concerns",
    single answer model and single gate model, verdict non-determinism, RAGAS not run.
11. **Future work.** Hardening against the adaptive framings (decision 1), a second freeze and
    round, a second provider, the not-run measurements.

The report does not depend on any Task 71 cleanup item. It quotes results sections, which the run
tasks keep current, not the status text that cleanup would fix.

## 11. Cleanup decision, merge and push (Tasks 71–73)

**What the build and run tasks write to README, and what they do not.** Task 63 adds the two UI
flags to Quickstart, because a feature's own documentation lands with it. Tasks 65–67 each add
their results section and update the "Phase 3: Gate-Off Vs Gate-On Harnesses (built, not run)"
heading and "Open measurements" item 2 for their own run. They write nothing else in the README,
and none of them edits stale status text elsewhere.

**Task 71 (STOP, owner decides).** The task lists every candidate below, with its current text
or state. The owner picks which are done. Only the picked items change, each committed by
pathspec. An item not picked stays exactly as it is, and the task records it as left.

| # | Candidate | Proposal recorded 2026-10-15 |
|---|---|---|
| C1 | README Roadmap: Phase 4 and Phase 5 status lines and rows; a Phase 6 table, Tasks 62–73 | fix |
| C2 | README "Open measurements": the Phase 4 paragraph above items 4–8, and "Phase 5 adds seven run tasks, all built and none run" | fix |
| C3 | README Phase 4 heading "(built, partly measured)" and its "Only the re-ingest … have been run" paragraph | fix |
| C4 | README Phase 5 intro "Nothing has been run; no figure exists." | fix |
| C5 | README Phase 3 and Phase 4 harness text naming `gemini-3.1-flash-lite` as today's or the judge model | fix |
| C6 | README Limitations: "Corpus is ~22 papers", and the papers example after it | fix |
| C7 | `docs/plans/2026-08-16-trust-gate-phase-1.md`: Phase 6 table renumbered 31–34 → 62–73 | fix |
| C8 | Phase 4 and Phase 5 specs: a dated note under each model line. The gate ran on `gemini-flash-lite-latest` (`eval/frozen_gate.json`); `gemini-3.1-flash-lite` was the answer model of Tasks 47 and 60. The original line stays. | fix |
| C9 | `.env.example`: the working tree's `LLM_MODEL=gemini-3.1-flash-lite`, plus `JUDGE_MODEL=gemini-flash-lite-latest` with a comment that the freeze checks it | commit |
| C10 | `.gitignore`: the working tree's line | commit |
| C11 | Delete untracked `PHASE1_RESULTS.md` and `PHASE3_RESULTS.md`; README and `docs/report.md` supersede them | delete |
| C12 | Delete untracked `eval/experience_poison_trial.py.pre-merge-backup`; the merged file is in git | delete |
| C13 | `CLAUDE.md` (local, gitignored): status brought up to date; not committed | fix |

C11 and C12 are untracked, so git cannot restore them. The STOP shows each path and its line
count, and a delete needs its own explicit yes.

**Task 72 (STOP).** Merge `phase-4-s3-corroboration` into `main`, fast-forward only. If `main`
has moved, stop and ask.

**Task 73 (STOP).** Verify that the push remote's URL is `sk-0607/rag-mem-sentry`. Then push
`main`, the branch, and the tag `gate-frozen-1`.

## 12. Components

| File | Responsibility | Status |
|---|---|---|
| `eval/class1_trial.py`, `eval/experience_poison_trial.py`, `eval/overhead.py` | freeze guard; real reference in scratch directories | modified |
| `app/config.py` | `UI_READ_GATE`, `UI_EXPERIENCE_STORE` | modified |
| `app/ui.py` | flag-driven default runner; audit offset, read, text join, panel rows; panel | modified |
| `eval/demo_plant.py` | `candidates`, `plant`, `purge` | new |
| `docs/report.md` | evaluation report | new |
| `README.md` | UI flags (Task 63); one results section per run (Tasks 65–67); cleanup items the owner picks (Task 71) | modified |
| Phase 1 plan, Phase 4 and 5 specs, `.env.example`, `.gitignore` | only if picked in Task 71 | maybe modified |
| `tests/eval/test_class1_trial.py`, `test_poison_trial.py`, `test_overhead.py` | guard and reference cases | modified |
| `tests/test_ui.py` | panel and wiring cases | modified |
| `tests/eval/test_demo_plant.py` | plant and purge cases | new |

Nothing under `gate/`, and neither `app/graph.py` nor `app/orchestrator.py`, changes.

## 13. Failure behavior

| Condition | Behavior |
|---|---|
| Freeze check not clean | Every run in this phase refuses before any LLM call and prints every difference. |
| `PerDay` quota mid-run | Results stay `complete: false`; the run resumes on a new go, a new day. |
| Audit file absent or unreadable | Panel shows the read gate's state and no rows; the answer is unaffected. |
| Candidate text missing | Row shows the id; the query is unaffected. |
| Write gate does not admit the planted record | Nothing persisted, verdict printed, exit 1; pick another. |
| Rehearsal admits the planted record | `purge --yes`, pick another (§9). |
| Purge leaves a listed id in the mirror | Exit 1; the backup is there to restore from. |
| Push remote not `sk-0607/rag-mem-sentry` | No push; stop and ask. |

## 14. Limitations

1. No clean-task regression number: RAGAS not run (decision 2).
2. The gate is not hardened after losing the adaptive round (decision 1). The report says what
   it lost.
3. One answer model and one gate model. The results would move on a more compliant model
   (parent §9.1).
4. The demo's exclusion is one record, chosen from items the blind screen already excluded, and
   replayed from cache. It shows the mechanism, not a rate.
5. The panel is single-user (§6.3).
6. Verdicts are not deterministic (Phase 5 limitation 4). A rerun can move a count.

## 15. Tasks

| Task | Deliverable | Kind |
|---|---|---|
| 62 | Freeze guard and real reference in `class1_trial`, `experience_poison_trial`, `overhead`; class-3 question check | build |
| 63 | Trust panel and UI flags | build |
| 64 | `eval/demo_plant.py` | build |
| 65 | Class-1 trial, gate off and on | RUN, owner go |
| 66 | Class-3 trial, `--arm off` and `--arm on` | RUN, owner go |
| 67 | Overhead, off / cold / warm | RUN, owner go |
| 68 | Demo prep: plant and rehearsal | RUN, owner go |
| 69 | Demo recording (owner); purge | owner; STOP |
| 70 | `docs/report.md` | docs |
| 71 | Cleanup decision: the owner picks from C1–C13 | STOP |
| 72 | Merge to `main` | STOP |
| 73 | Push branch, `main` and tag | STOP |

Tasks 62–64 make no LLM call and can run in any order. Tasks 65–67 follow 62. Task 68 follows 63
and 64. Task 70 follows 65–69. Task 71 follows 70.

## 16. Testing

Offline, with fakes, matching the repo's injection convention. `pytest tests/gate tests/memory
tests/eval` stays offline and free.

- **Freeze guard:** each of the three harnesses refuses with a non-empty check and makes zero
  calls through its injected `chat`; a clean check is recorded in its results file.
- **Real reference:** a harness given a scratch directory screens with the real reference's
  fingerprint, and refuses on an empty trusted set.
- **UI wiring:** flags off gives a runner called with `read_gate=None` and no store. Each flag
  adds exactly its collaborator.
- **Panel:** entries before the offset are not shown; `stage: "write"` entries are not shown;
  the top signal is the highest score; the span is highlighted when verbatim and shown beside the
  text otherwise; a missing text shows the id; gate off shows the off message, not an empty
  table; an absent audit file does not raise.
- **Plant:** refuses an existing plant file, a non-class-3 id, a non-test-split id and an unclean
  freeze; a non-admit write persists nothing and exits 1; an admit writes the plant file.
- **Purge:** without `--yes` changes nothing; with `--yes` it backs up the mirror first, then
  removes the planted id and every later record, and leaves earlier records; a listed id still
  present after the purge exits 1.
- **`candidates`:** lists only excluded class-3 test-split items whose top signal is `judge`.

## 17. Exit criteria

- `pytest -q` green; the offline suites free.
- Tasks 65–67 each have a README section with both models, the freeze check, and unrounded
  figures. "Open measurements" item 2 is closed or says exactly what stayed open.
- The demo is recorded. After the purge, the store holds no planted or demo-run record.
- `docs/report.md` covers §10's sections, with §9.3 marked not measured.
- Task 71 records, for each of C1–C13, whether it was done or left, by owner decision.
- `main` holds the branch; the branch and `gate-frozen-1` are on `sk-0607/rag-mem-sentry`.

## 18. Risks

- **R1 — Free-tier quota.** Three run days, plus a demo day. Mitigated by one run per day,
  `LLM_MIN_INTERVAL=2`, and a resumable `complete: false`. Diagnose `PerDay` vs `PerMinute`
  before any retry.
- **R2 — The gate-off arm shows no behavior-level harm again.** Classes 1 and 3 have had behavior
  ASR 0/3 in every earlier run on this model. If so, the report leads on the context level for
  those classes and on class 2 for behavior, as parent §9.1 already directs. A 0 on the gate-off
  arm is reported, not re-tried with stronger poison.
- **R3 — The class-3 trial is still papers-era.** Task 62 stops for the owner (§5).
- **R4 — Overhead under-reports S3** if the scratch-directory reference is not fixed. Task 62
  fixes it before Task 67.
- **R5 — The planted record survives the demo.** Purge has a verify step and a backup, and the
  exit criteria check the store.
- **R6 — An `.env` edit between runs** (a model swap for another experiment) changes
  `JUDGE_MODEL`. The freeze guard refuses the run.
- **R7 — Report numbers drift from the README.** The report quotes only README sections and cites
  them; Task 70's review compares every figure against its source section.
