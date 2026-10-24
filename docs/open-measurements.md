# Open measurements and known concerns

## Open measurements

Things this project has decided are worth measuring and has not measured. Listed here rather
than inside a finished plan document, because an open item parked in a completed plan is an item
that gets lost.

1. **R3's poison-only control (5 LLM calls, open since 2026-08-19).** The R3 trials against a
   poison-only chunk graded weak, read as "rejected for having no content to answer with, not
   for containing poison." That reading is almost certainly right, but the confirming run —
   grade an off-topic *clean* chunk against the same five questions and check it also grades
   weak — was asserted, not executed. R3's conclusion does not depend on it; it rests on the
   five controlled carrier trials. Recorded because an asserted control is a hypothesis, and
   this project has been bitten once already by treating one as an observation.
2. **Behavior-level ASR under a less instruction-override-shaped poisoning strategy.** Measured
   twice now — the "False-fact / unsafe-approach poison retrial (2026-08-24)" and "Gate-off
   baseline on the simplified host (2026-09-16)" subsections under Phase 2 Results in [results.md](results.md) — with
   false-fact / unsafe-approach poison (R3a's class 2/3) against thin-coverage questions and no
   co-retrieved clean sibling. Behavior ASR stayed 0/3 both times, so this has not moved off zero;
   still open is whether *some* poisoning strategy against this model and system prompt would move
   it. *Update, 2026-10-13:* Phase 3 was built after the owner's 2026-09-15 override of the stop
   decision, and the gate-off vs gate-on comparison is **still open**. The harness now exists
   (`python -m eval.class1_trial`; `python -m eval.experience_poison_trial --arm on`, against the
   `--arm off` baseline; `EVAL_READ_GATE=1 python -m eval.run_eval`; `python -m eval.overhead`),
   and the run has not been done, by owner decision on 2026-10-13. No gate-on attack-success,
   RAGAS or overhead figure exists. *Update, 2026-10-15:* the class-1 trial ran ("Phase 6: Class-1
   Attack Success" in [results.md](results.md)): 5/5 → 0/5 at both levels on `gemini-3.1-flash-lite`. The class-3 trial
   ran too ("Phase 6: Class-3 Attack Success"): context 3/3 → 0/3, behavior 0/3 → 0/3, unmeasurable
   at behavior level (R6). The overhead harness ran ("Phase 6: Overhead"): +2.6 calls and
   +8.822338439989835 s per query cache-cold, +0 calls and +4.066152779990807 s cache-warm. **Closed,
   2026-10-15**, with Tasks 65-67 recorded in the three Phase 6 sections in [results.md](results.md). The RAGAS gate-on arm
   is not run, by owner decision on 2026-10-15.
3. **The experience store's block band.** `block_below=0.20` is stricter than the corpus's
   `0.10` and nothing in the clean calibration set came near it. See the calibration section
   in [results.md](results.md).

Phase 4 added five run tasks (43–47). All five ran, on 2026-10-14 and 2026-10-15, and are
recorded in the Phase 4 sections in [results.md](results.md).

4. *Closed 2026-10-14.* Trust-flagged re-ingest (Task 43): see "Trust-flagged re-ingest" in [results.md](results.md).
5. *Closed 2026-10-14.* S3 false positives on clean untrusted content (Task 44): see
   "Phase 4: S3 on Clean Untrusted Content" in [results.md](results.md). Over budget, 3 / 16.
6. *Closed 2026-10-14.* Calibration (Task 45): see "Phase 4: Calibration" in [results.md](results.md). Was: **Corpus frozen, calibration split screened, thresholds proposed (Task 45,
   `python -m eval.attack_corpus validate` and `freeze`, then
   `LLM_MIN_INTERVAL=2 python -m eval.attack_screen --split calibration` and `--clean-experience`,
   then `python -m eval.calibrate`).** A hard threshold for `judge` chosen on the
   calibration split only.
7. *Closed 2026-10-14.* Ablation (Task 46): see "Phase 4: Signal Ablation" in [results.md](results.md). Was: **Test split screened and ablation table (Task 46,
   `LLM_MIN_INTERVAL=2 python -m eval.attack_screen --split test`, then
   `python -m eval.ablation data/phase4_screen/test/results.json`).** The five-row ablation,
   screened once, after the thresholds are committed.
8. *Closed 2026-10-15.* Class-2 trial: see "Phase 4: Class-2 Attack Success" in [results.md](results.md). Was: **Class-2 trial, control / gate off / gate on (Task 47,
   `LLM_MIN_INTERVAL=2 python -m eval.class2_trial`).** Run last, on a smaller answer model the
   owner names, with `JUDGE_MODEL` kept on `gemini-3.1-flash-lite`. It is also the first time
   class 2 is attacked in fact on the ungated host.

Phase 5 added seven run tasks (55–61). All seven ran, on 2026-10-14 and 2026-10-15, and are
recorded in the Phase 5 sections in [results.md](results.md).

9. *Closed 2026-10-14.* Gate freeze (Task 55): see "Gate freeze" in [results.md](results.md).
10. *Closed 2026-10-15.* Probe session 1: see "Probe history" in [results.md](results.md). Was: **Probe session 1 (Task 56, `PROBE_SESSION=1 LLM_MIN_INTERVAL=2 python -m eval.probe_server`,
    then `python -m eval.probe_summary`).**
11. *Closed 2026-10-15.* Probe session 2: see "Probe history" in [results.md](results.md). Was: **Probe session 2 (Task 57, `PROBE_SESSION=2 ...`, same commands).**
12. *Closed 2026-10-15.* Probe session 3: see "Probe history" in [results.md](results.md). Was: **Probe session 3 (Task 58, `PROBE_SESSION=3 ...`, same commands).**
13. *Closed 2026-10-15.* Adaptive detection: see "Phase 5: Adaptive Detection" in [results.md](results.md). Was: **Adaptive set frozen and screened (Task 59,
    `ATTACK_CORPUS_DIR=data/phase5_adaptive_corpus python -m eval.attack_corpus freeze --adaptive`,
    then `ATTACK_CORPUS_DIR=data/phase5_adaptive_corpus LLM_MIN_INTERVAL=2 python -m eval.attack_screen --split adaptive`).**
    Adaptive detection per class, beside the blind test split.
14. *Closed 2026-10-15.* Adaptive class-2 trial: see "Phase 5: Adaptive Class-2 Attack Success" in [results.md](results.md). Was: **Adaptive class-2 trial (Task 60,
    `ATTACK_CORPUS_DIR=data/phase5_adaptive_corpus python -m eval.class2_trial --split adaptive`).**
15. *Closed 2026-10-15.* Round results: see "Phase 5: What the Adaptive Round Shows" in [results.md](results.md). Was: **Round results recorded (Task 61, `python -m eval.probe_summary --final data/phase5_screen/results.json`).**

## Known Phase 3+ concerns (named, not addressed)

- **Unbounded growth on the memory path.** Nothing dedupes, caps, or evicts experience records:
  every run appends one, `ExperienceStore.retrieve()` re-parses the entire JSONL mirror on every
  call, and the agent's own answer becomes the next run's injected context — a self-confirmation
  channel that nothing currently tests. All of this is fine at capstone scale (tens of records)
  and none of it is a Phase 2 problem, but each becomes one at any real volume.
  `ExperienceStore.purge(ids)` is the only removal path today, and it is manual.

- **Branch review punch list.** The whole-branch review of Phase 2 raised these and triaged them
  as non-blocking. Recorded here rather than fixed, so none of them is lost:
  - `ExperienceStore.purge` returns early when no id is present in the JSONL mirror, so a Chroma
    entry orphaned from the mirror is never deleted. Only reachable if the two ever diverge.
  - S1 dedupes homoglyph hits and encoded hits by label within each channel but not across them,
    so one payload reachable through both would count twice. No current input reaches both.
  - `eval/fpr.py` dedupes by `item_id` before filtering by store, so an id colliding across
    stores would drop a row. Unreachable with today's id schemes.
  - `tests/eval/test_fpr.py` excludes an audit entry with no `store` key without saying why.
  - `app/orchestrator.py` discards the experience write verdict — a quarantined write is
    indistinguishable from an admitted one to any caller.
  - Memory read and write failures are not isolated from the answer path: a Chroma error in
    either takes down a query that would otherwise have answered fine.
  - `default_experience_store` uses only the basename of `config.CHROMA_DIR`, so a configured
    absolute or nested Chroma path is silently flattened under `data_dir`.
  - `ExperienceStore.purge` rewrites the JSONL mirror in place with no temp-file-and-rename, so
    an interrupted purge can truncate it.
  - `docs/plans/2026-08-19-phase-2-experience-memory.md:551` still refers to "Task 21's
    demonstration"; the demonstration shipped as Task 20.
  - `tests/test_config.py::test_config_constants_present` asserts a value that `.env` can
    override, which is why it fails on this machine. Pre-existing and unrelated to Phase 2, but
    it is a real test bug, not just a local environment quirk — and the Phase 2 plan's
    "`pytest -q` green" exit checkbox is ticked next to it. *Fixed in Phase 3 (`99a24fa`): the
    test now asserts the defaults without the machine's `.env` answering for them.*
  - `format_experiences` injects each retrieved record's full prior answer into the prompt, so
    prompt size grows with record length as well as record count.
