# Roadmap

## Roadmap

Phase 3 (the S2 policy judge and the read-time gate) is **built; gate-off vs gate-on
measurements run 2026-10-15 in Phase 6, except RAGAS** (owner decision, 2026-10-15).

| Task | Deliverable | Status |
|---|---|---|
| 22 | Poison trial committed; gate-off baseline on the simplified host | measured |
| 23 | S2 policy judge, corpus rubric | built, offline tests |
| 24 | S2 experience rubric | built, offline tests |
| 25 | Batch signal runner, audit `stage`, placeholder `judge` hard threshold | built, offline tests |
| 26 | `gate/read.py`: cache-first `ReadGate.screen_batch`, S1 degradation | built, offline tests |
| 27 | Gate node in `app/graph.py`, experience filtering in `app/orchestrator.py`, behind `read_gate=None` | built, offline tests |
| 28 | R5: S2 false-positive rate on quoted-attack hard negatives | measured |
| 29 | Class-1 trial and `--arm off` / `--arm on` on the class-3 trial | run 2026-10-15 (Tasks 65, 66) |
| 30 | `EVAL_READ_GATE=1` RAGAS arm and the cold/warm overhead harness | overhead run 2026-10-15 (Task 67); RAGAS not run |
| 31 | Docs reconciled with what was built | docs |

Phase 4 (S3 corroboration) is **built and measured**. Plan:
`docs/plans/2026-10-13-phase-4-s3-corroboration.md`.

| Task | Deliverable | Status |
|---|---|---|
| 32 | Trust marking at ingest | built, offline tests |
| 33 | `gate/reference.py`: `ReferenceIndex` | built, offline tests |
| 34 | S3 claim extraction | built, offline tests |
| 35 | S3 applicability, reference lookup, agreement, score, fingerprint | built, offline tests |
| 36 | S3 wired into `default_read_gate`; placeholder `corroborate` threshold | built, offline tests |
| 37 | `eval/attack_corpus.py`: validate, freeze, split, load | built, offline tests |
| 38 | `eval/attack_screen.py` and `eval/ablation.py` | built, offline tests |
| 39 | `eval/corroborate_fpr.py` | built, offline tests |
| 40 | `eval/calibrate.py` | built, offline tests |
| 41 | `eval/class2_trial.py` | built, offline tests |
| 42 | Docs reconciled with what was built | docs |
| 43 | Re-ingest with trust flags; write-time false-positive rate re-checked | run 2026-10-14, unchanged |
| 44 | S3 false positives on clean untrusted content | run 2026-10-14, 3 / 16 = 0.1875, over the 2% budget |
| 45 | Corpus frozen; calibration split screened; thresholds proposed and committed | run 2026-10-14, judge 0.80; corpus band over budget, 2 / 40 |
| 46 | Test split screened; ablation table | run 2026-10-14 |
| 47 | Class-2 trial, control / gate off / gate on | run 2026-10-15 |

Phase 5 (adaptive round) is **built and run**. Plan:
`docs/plans/2026-10-14-phase-5-adaptive-round.md`.

| Task | Deliverable | Status |
|---|---|---|
| 48 | Freeze record and check | built, offline tests |
| 49 | Adaptive attack-corpus kind | built, offline tests |
| 50 | One-bit probe endpoint | built, offline tests |
| 51 | Adaptive split in the screen and the class-2 trial | built, offline tests |
| 52 | Probe summary | built, offline tests |
| 53 | Attacker brief | docs |
| 54 | Docs reconciled with what was built | docs |
| 55 | Freeze the gate | run 2026-10-14, tag `gate-frozen-1` |
| 56–58 | Probe sessions 1–3 | run 2026-10-15, 60 probes, none voided |
| 59 | Adaptive set frozen and screened | run 2026-10-15, detection below blind in every class |
| 60 | Adaptive class-2 trial | run 2026-10-15, gate on 5/5 at both levels |
| 61 | Round results recorded | done 2026-10-15 |

Phase 6 (demo and report). Plan: `docs/plans/2026-10-15-phase-6-demo-and-report.md`.

| Task | Deliverable | Status |
|---|---|---|
| 62 | Freeze guard and real reference in the three Phase 3 harnesses | built, offline tests |
| 63 | Trust panel and UI flags | built, offline tests |
| 64 | Demo plant tool | built, offline tests |
| 65 | Class-1 trial, gate off and on | run 2026-10-15, 5/5 → 0/5 at both levels |
| 66 | Class-3 trial, `--arm off` and `--arm on` | run 2026-10-15, context 3/3 → 0/3; behavior unmeasurable (R6) |
| 67 | Overhead, off / cold / warm | run 2026-10-15 |
| 68 | Demo prep: plant and rehearsal | done 2026-10-15 |
| 69 | Demo recording; purge | done 2026-10-15 |
| 70 | Evaluation report, `docs/report.md` | done 2026-10-15 |
| 71 | Cleanup decision | done 2026-10-24, C1-C13 all done |
| 72 | Merge to `main` | done 2026-10-24 |
| 73 | Push | done 2026-10-24 |

Later: multi-turn memory, tracing panel, larger corpus + eval set.
