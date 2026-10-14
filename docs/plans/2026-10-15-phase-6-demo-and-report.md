# Phase 6 — Demo and Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure what Phase 3 built and never ran (class-1 and class-3 attack success, gate off vs on, and overhead) at the frozen gate. Show the gate excluding a poisoned memory record live, in a Gradio trust panel. Write the evaluation report. Close with an owner decision on cleanup, then merge and push.

**Architecture:**
- Three Phase 3 harnesses gain the Phase 5 freeze guard and the real trusted reference.
- `app/ui.py` gains a trust panel that reads the read gate's audit log. Both the read gate and experience memory are opt-in through environment flags, default off.
- A small `eval/demo_plant.py` plants one blind class-3 record through the real write gate and purges it, plus the demo's own records, afterwards.
- Nothing under `gate/` changes, and neither does `app/graph.py` or `app/orchestrator.py`.

**Tech Stack:** Python 3.11+, pytest, Gradio (already the UI), the shipped `eval/` package and `eval/freeze_gate.py`. No new dependency.

**Spec:** `docs/specs/2026-10-15-phase-6-demo-and-report-design.md`. Read it before Task 62. Parent: `docs/specs/2026-08-16-poisoned-content-trust-gate-design.md`. Previous phase: `docs/specs/2026-10-14-phase-5-adaptive-round-design.md` and its plan `docs/plans/2026-10-14-phase-5-adaptive-round.md`.

## Global Constraints

- **Every run that spends LLM budget needs the owner's explicit go.** Tasks 62–64 and 70 make no LLM call. Tasks 65–68 each open with a STOP step. Do not pass a STOP on your own judgement, an earlier go, or a go given for a different task.
- **At most one LLM-spending run per day.** Diagnose `PerDay` vs `PerMinute` before any retry. A `PerDay` error ends the day; the results file stays `complete: false`.
- **The gate is frozen at `gate-frozen-1`.** Nothing under `gate/` changes. `app/graph.py` and `app/orchestrator.py` do not change either.
- **Models:** gate model `JUDGE_MODEL=gemini-flash-lite-latest`, set explicitly. Answer model `LLM_MODEL=gemini-3.1-flash-lite`. Every figure names both. Quote models from `eval/frozen_gate.json` and README results sections, never from the Phase 4 or 5 specs.
- **Every Phase 6 run refuses unless the freeze check is clean** (`require_frozen()`), and records the check in its results file.
- **A harness with its own scratch directory passes the real reference** (`default_reference(<real data dir>)`), and refuses to start on an empty trusted set.
- **`read_gate=None` stays the default.** The UI's read gate and experience store are on only when `UI_READ_GATE=1` / `UI_EXPERIENCE_STORE=1`. Any other value is off.
- **Read never mutates a store.** The panel reads files only. It writes nothing and calls no gate method.
- **The panel is never served by `app/api.py` or `eval/probe_server.py`.** Do not touch either file.
- **`pytest tests/gate tests/memory tests/eval` stays offline and free.** Fakes for `chat`; no SDK mocks, no network, no real LLM.
- **Nothing this phase adds names a source id of the current corpus:** no NQ document id, no attack item id, no question text, in code or tests. Fixtures use invented values. The planted item's id comes from the owner on the command line.
- **No trained classifiers.** Rules + LLM judge only.
- **Numbers in README and the report match command output exactly, unrounded.**
- **Cleanup waits for Task 71.** Do not edit stale README text, the Phase 1 plan's roadmap, the Phase 4 or 5 specs, `.env.example` or `.gitignore`, and do not delete `PHASE1_RESULTS.md`, `PHASE3_RESULTS.md` or `eval/experience_poison_trial.py.pre-merge-backup`, before Task 71's STOP. README edits before Task 71 are only the ones a task names: the UI flags (Task 63) and each run's own results section (Tasks 65–67).
- **TDD, one commit per task**, RED verified before GREEN. Commit style `feat(eval):` / `feat(ui):` / `fix(eval):` / `docs:`.
- **Commit by pathspec** (`git add <paths> && git commit -m "..." -- <paths>`), never a bare commit or `git add -A`. No `Co-Authored-By` or any Claude/Anthropic attribution.
- **Branch:** `phase-4-s3-corroboration`. **Never push to the `self-correcting-rag` portfolio repo.** Own remote: `sk-0607/rag-mem-sentry`. Merge and push are Tasks 72 and 73, each behind a STOP.
- **Comments explain why** (usually a rejected alternative), matching the repo's existing comments.

## Format note — why this plan carries almost no code

This plan has the same form as the Phase 3, 4 and 5 plans, at the owner's request. **Tests are specified by name and asserted behavior; implementation by signature and invariants.** The repo supplies every idiom this phase needs:

| Idiom | Where |
|---|---|
| Freeze guard and `freeze` field | `eval/attack_screen.py`, `eval/class2_trial.py`, and their tests |
| Passing the real reference from a scratch directory | `eval/class2_trial.py`, `eval/corroborate_fpr.py` |
| Gradio wiring with an injected runner | `app/ui.py`, `tests/test_ui.py` |
| Store fakes | `tests/memory/` |

Each task names the file to read before writing its sibling.

What stays exact: signatures, JSON contracts, file paths, commands, expected outcomes, and every number a later task depends on.

The code graph (`graphify`) holds no signatures, class fields or module constants. Where a task says "read X first", read the source file.

## Review Focus

Failure modes the spec implies, most likely to bite first. Each has a test in the task that owns the code.

1. **Poisoned content rendered as HTML in the panel.** Class-1 poison uses hidden CSS and HTML comments. If the panel injects candidate text into `gr.HTML` unescaped, the poison renders in the owner's browser, hiding or styling itself, exactly where the demo claims to expose it. Expected: every candidate text, reason and evidence span is HTML-escaped before rendering. The highlight markup is added around escaped text only. Test: Task 63 `test_render_escapes_candidate_html`.
2. **A harness builds its gate in a scratch directory and S3 silently abstains.** The numbers look normal and S3 never ran. Expected: each of the three harnesses screens with the real reference's fingerprint, and refuses on an empty trusted set. Tests: Task 62 `test_<harness>_passes_real_reference`, `test_<harness>_refuses_empty_reference`.
3. **The audit file is shorter than the recorded offset,** because someone wiped or rotated `data/` between the offset read and the query. Seeking past the end silently returns nothing, and the panel claims nothing was screened. Expected: an offset past the end reads from the start of the file. Test: Task 63 `test_offset_past_end_reads_from_start`.
4. **`purge` run twice, or with no plant file.** Expected: exit 1 with "nothing planted", no backup written, nothing removed. Test: Task 64 `test_purge_without_plant_file_exits_1_and_changes_nothing`.
5. **A flag set to `0`, `false` or empty in `.env`.** A truthiness check would turn the gate on. Expected: only the exact string `1` enables a flag. Test: Task 63 `test_flag_values_other_than_1_are_off`.

## Task numbering against the spec

The spec's §15 lists Tasks 62–73, and this plan keeps them one to one:

| Tasks | Kind |
|---|---|
| 62–64 | build |
| 65–68 | gated runs |
| 69 | owner recording, then purge |
| 70 | report |
| 71 | cleanup decision |
| 72–73 | merge and push |

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `eval/class1_trial.py`, `eval/experience_poison_trial.py`, `eval/overhead.py` | freeze guard; `freeze` field; real reference; empty-reference refusal | modified |
| `eval/attack_corpus.py` | `to_experience_record(entry)` extracted from the class-3 projection, only if it is built inline | maybe modified |
| `app/config.py` | `UI_READ_GATE`, `UI_EXPERIENCE_STORE` | modified |
| `app/ui.py` | flag-driven default runner; audit offset and read; text join; panel rows; HTML render | modified |
| `eval/demo_plant.py` | `candidates`, `plant`, `purge`; CLI | new |
| `docs/report.md` | evaluation report | new |
| `README.md` | UI flags (63); run sections (65–67); owner-picked cleanup (71) | modified |
| `tests/eval/test_class1_trial.py`, `test_poison_trial.py`, `test_overhead.py` | guard and reference cases | modified |
| `tests/eval/test_attack_corpus.py` | record extraction keeps item text identical | maybe modified |
| `tests/test_ui.py` | flags, panel, render | modified |
| `tests/eval/test_demo_plant.py` | candidates, plant, purge | new |

---

### Task 62: Freeze guard and real reference in the three Phase 3 harnesses

**Files:**
- Modify: `eval/class1_trial.py`, `eval/experience_poison_trial.py`, `eval/overhead.py`
- Test: `tests/eval/test_class1_trial.py`, `tests/eval/test_poison_trial.py`, `tests/eval/test_overhead.py`

**Interfaces:**
- Consumes: `eval.freeze_gate`: `require_frozen`, `check`, `FreezeError`, `FROZEN_GATE`. Also `gate.reference.default_reference(data_dir)`, `ReferenceIndex.__len__`, `ReferenceIndex.fingerprint()`, and `gate.read.default_read_gate(data_dir, chat=None, reference=None)`.
- Produces: each harness's results file gains `"freeze": {"record": <FROZEN_GATE as posix if it exists, else null>, "differences": <check(...) output>}`, the shape `eval/attack_screen.py` writes. No new public function is required. If a harness builds its gate in more than one place, factor one local builder that takes the scratch directory and the reference.

- [ ] **Step 0: Read, then decide whether to stop**

Read `eval/attack_screen.py` and `eval/class2_trial.py` for the guard, the `freeze` field and the real-reference wiring. Read the three harnesses in full.

Answer three questions in the task report:
1. Does each harness build its read gate in a scratch directory? Does it pass `reference=`?
2. What directory does `experience_poison_trial` write its `results-<arm>.json` to?
3. Do `experience_poison_trial`'s `TRIALS` questions, poison records or pre-registered `check`s name papers-era content (AI-security papers, arXiv topics) rather than NQ?

**If question 3's answer is yes: STOP.** Report the trial's questions and records to the owner and wait for their choice. The choices are: re-target to NQ (a separate task the owner defines), or run as-is and report them as papers-era questions on the NQ host. Do not change the questions on your own judgement. Steps 1–5 below do not depend on that choice and may proceed.

**Guard invariants.**
- `require_frozen()` runs at the start of each harness's `main`, before any read gate, retriever or LLM client is built. This includes the gate-off arms: `--arm off`, and `class1_trial`'s and `overhead`'s off passes. One invocation is one run at one state.
- A `FreezeError` exits non-zero with its message and writes no results file.
- The `freeze` field is written into every results file the harness writes, including partial ones (`complete: false`).

**Reference invariants.**
- Every read gate a harness builds in a directory other than the real data directory gets `reference=default_reference(<real data dir>)`. The real data directory is the one `eval/attack_screen.py::main` passes to `default_reference`; read it there and do not introduce a second constant.
- Before any LLM call, `len(reference) == 0` exits non-zero with a message naming the empty trusted set.
- `overhead`'s `on-cold` arm still starts from an empty verdict cache in a fresh temp directory. Only the reference comes from the real data directory, not the cache.

- [ ] **Step 1: Write the failing tests**

In each of the three test files, with `<harness>` being `class1_trial`, `poison_trial` or `overhead`:
- `test_<harness>_refuses_unclean_freeze_before_building_gate`. Monkeypatched `require_frozen` raises `FreezeError`. Expect: exit non-zero, `default_read_gate` never called, zero calls through the injected or fake `chat`, no results file.
- `test_<harness>_records_freeze_field`. Monkeypatched `check` returns `[]` and `FROZEN_GATE` exists. Expect: the results JSON has `freeze == {"record": <posix path>, "differences": []}`.
- `test_<harness>_passes_real_reference`. Monkeypatch `default_reference` to return a fake with a fixed `fingerprint()` and `len` 3, and record the `data_dir` it was called with. Expect: every `default_read_gate` call received that fake as `reference`, and `default_reference` was called with the real data directory, not the scratch one.
- `test_<harness>_refuses_empty_reference`. The fake has `len` 0. Expect: exit non-zero, zero `chat` calls.

`tests/eval/test_overhead.py` also gets:
- `test_cold_arm_cache_is_empty_with_real_reference`. The cold arm's temp directory holds no cache file before its first question, and its gate's reference is the real one.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/eval/test_class1_trial.py tests/eval/test_poison_trial.py tests/eval/test_overhead.py -v`
Expected: FAIL. No guard is called, no `freeze` key exists, and `default_reference` is never called by `overhead`.

- [ ] **Step 3: Implement** to the invariants. Existing tests that build these harnesses with fakes may need `require_frozen` and `default_reference` monkeypatched. Change them only that way, and say so in the commit body.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/eval/test_class1_trial.py tests/eval/test_poison_trial.py tests/eval/test_overhead.py -v` then `pytest tests/gate tests/memory tests/eval -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add eval/class1_trial.py eval/experience_poison_trial.py eval/overhead.py tests/eval/test_class1_trial.py tests/eval/test_poison_trial.py tests/eval/test_overhead.py
git commit -m "feat(eval): freeze guard and real trusted reference in the Phase 3 harnesses" -- eval/class1_trial.py eval/experience_poison_trial.py eval/overhead.py tests/eval/test_class1_trial.py tests/eval/test_poison_trial.py tests/eval/test_overhead.py
```

---

### Task 63: Trust panel and UI flags

**Files:**
- Modify: `app/config.py`, `app/ui.py`, `README.md` (Quickstart: the two flags only)
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `app.orchestrator.run_agentic` (read its signature; it takes `read_gate=` and `store=`). Also `gate.read.default_read_gate(data_dir)` and its `ReadGate.audit.path`; `memory.store.default_experience_store(data_dir)`; `memory.types.ExperienceRecord` and `screened_text()`; `config.EXPERIENCES_PATH`; the chunks path the retriever reads (read `app/config.py`).
- Produces:
  - `app/config.py`: `UI_READ_GATE: bool = os.getenv("UI_READ_GATE") == "1"`, `UI_EXPERIENCE_STORE: bool = os.getenv("UI_EXPERIENCE_STORE") == "1"`
  - `app/ui.py`:
    - `audit_offset(path: Path) -> int`. The file's size in bytes; 0 when absent.
    - `read_entries_since(path: Path, offset: int) -> list[dict]`. The parsed lines after `offset` with `stage == "read"`.
    - `load_texts(chunks_path: Path, experiences_path: Path) -> dict[str, str]`. Chunk id maps to chunk `text`. Experience record id maps to `ExperienceRecord(**row).screened_text()`, the text the gate saw.
    - `panel_rows(entries: list[dict], texts: dict[str, str]) -> list[dict]`. Keys: `store`, `item_id`, `action`, `trust`, `top_signal`, `top_reason`, `evidence_span`, `text` (`None` when unknown), `span_verbatim: bool`.
    - `render_panel(rows: list[dict], gate_on: bool) -> str`. Escaped HTML.
    - `make_runner(read_gate=None, store=None)`. A callable with the signature `build_demo`'s runner already has (read `app/ui.py`). It calls `run_agentic` with exactly these collaborators.
    - `build_demo(runner=None, audit_path: Path | None = None, texts_loader=load_texts)`. With `runner=None` it builds `make_runner` from the flags, and `audit_path` from the built read gate's `audit.path`. Every existing injectable argument keeps working.

**Wiring invariants.**
- Flags off: the default runner calls `run_agentic` with `read_gate=None` and no store, byte-for-byte today's call.
- `UI_READ_GATE` on: `read_gate=default_read_gate(<real data dir>)`, with its default reference.
- `UI_EXPERIENCE_STORE` on: `store=default_experience_store(<real data dir>)`.
- The real data directory is the one `eval/prewarm.py` uses (`GATE_DIR`, default `data`), so the prewarmed cache serves the UI. Read it there.
- `respond` returns the existing answer output plus the panel HTML. On each query, take `audit_offset` before calling the runner, then `read_entries_since` after.

**Panel invariants.**
- `read_entries_since`:
  - An offset past the end reads from 0.
  - A line that is not valid JSON is skipped. A half-written trailing line from a concurrent writer must not raise.
  - An absent file returns `[]`.
- `panel_rows`:
  - `top_signal` is the entry's highest-scoring signal; its `reason` and `evidence_span` come from that signal.
  - No signals gives `top_signal` `None`.
  - `span_verbatim` is true when the span is a substring of the text.
- `render_panel`:
  - **`html.escape` on every text, reason and span before any markup is added.**
  - A verbatim span is wrapped in `<mark>` inside the escaped text: escape the three parts, then join.
  - A non-verbatim span is shown beside the text.
  - A row with no text shows the id.
  - `gate_on=False` returns a line saying the read gate is off, never an empty table.
  - `gate_on=True` with no rows says nothing was screened for this query.
- **Polish:** use the `impeccable` skill for the panel's look. Use Gradio's own `css=` / `elem_classes` only, no new dependency, panel only. Polish never removes escaping.
- **README Quickstart:** one short paragraph naming both flags, their default (off), that the read gate uses `data/` and its prewarmed cache, and that the panel is single-user.

- [ ] **Step 1: Write the failing tests** (`tests/test_ui.py`, fakes only, `tmp_path` files)

- `test_flags_off_runner_passes_no_gate_and_no_store`
- `test_read_gate_flag_adds_only_the_gate`
- `test_store_flag_adds_only_the_store`
- `test_flag_values_other_than_1_are_off`. Monkeypatch env to `0`, `false`, `""`, `true`, reload `app.config`; each is off.
- `test_audit_offset_of_absent_file_is_zero`
- `test_entries_before_offset_are_not_returned`
- `test_write_stage_entries_are_not_returned`
- `test_offset_past_end_reads_from_start`
- `test_partial_trailing_line_is_skipped`
- `test_top_signal_is_highest_score`, plus `None` with no signals
- `test_verbatim_span_is_marked`
- `test_non_verbatim_span_is_shown_beside_text`
- `test_missing_text_shows_id`
- `test_render_escapes_candidate_html`. Text `<div style="display:none">x</div>`, reason `<b>r</b>` and span `<script>` appear escaped; the only tags in the output are the panel's own.
- `test_gate_off_renders_off_message_not_empty_table`
- `test_respond_shows_only_this_querys_entries`. A fake runner appends two read entries to a pre-filled audit file; the panel holds those two only.
- `test_build_demo_returns_blocks` (existing) still passes.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ui.py -v`
Expected: FAIL with `ImportError` on `audit_offset` (and the new names).

- [ ] **Step 3: Implement** to the interfaces and invariants. Then run the `impeccable` pass on the panel only.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ui.py -v` then `pytest -q`
Expected: PASS. `tests/test_retrieval.py` may skip without a built corpus.

- [ ] **Step 5: Manual check, no LLM**

Run `python app.py` with both flags unset and confirm the panel shows the off message. Do not submit a query with either flag on; that spends budget, and Task 68 does it under a go.

- [ ] **Step 6: Commit**

```bash
git add app/config.py app/ui.py tests/test_ui.py README.md
git commit -m "feat(ui): trust panel from the read gate's audit log, behind opt-in flags" -- app/config.py app/ui.py tests/test_ui.py README.md
```

---

### Task 64: Demo plant tool

**Files:**
- Create: `eval/demo_plant.py`
- Modify (only if needed, see Step 0): `eval/attack_corpus.py`
- Test: `tests/eval/test_demo_plant.py`; `tests/eval/test_attack_corpus.py` if `attack_corpus.py` changes

**Interfaces:**
- Consumes:
  - From `eval.attack_corpus`: `load(directory, split)`, `corpus_dir()`, `to_gate_item(entry)`.
  - From `eval.freeze_gate`: `require_frozen`, `FreezeError`.
  - From `memory`: `ExperienceStore.write(record, gate=) -> Verdict`, `all_records()`, `purge(ids)`, `default_experience_store(data_dir)`, `default_experience_gate(data_dir)`.
  - From `memory.types`: `ExperienceRecord`, and `record_from_run` for how `ts` is minted.
  - Also `config.EXPERIENCES_PATH`, `gate.types.Action`, and the Task 46 screen output `data/phase4_screen/test/results.json` (rows `{id, class, group, store, action, trust, top_signal, signals, reduced_assurance}`).
- Produces:
  - `eval/attack_corpus.py` (only if Step 0 finds the class-3 record built inline): `to_experience_record(entry: dict) -> ExperienceRecord`. `to_gate_item` calls it for class 3.
  - `eval/demo_plant.py`:
    - `PLANT_FILE = Path("data/demo_plant.json")`
    - `class PlantError(RuntimeError)`
    - `candidates(entries: list[dict], screen_rows: list[dict]) -> list[dict]`. Class-3 test-split entries whose row `action != "admit"` and `top_signal == "judge"`. Each carries `id` and the entry's target question.
    - `plant(item_id: str, entries: list[dict], store, gate, plant_file: Path = PLANT_FILE, now=<ts minter>) -> dict`
    - `purge_ids(plant: dict, records: list[ExperienceRecord]) -> list[str]`
    - `purge(store, mirror_path: Path, plant_file: Path = PLANT_FILE, yes: bool = False) -> list[str]`
    - CLI: `python -m eval.demo_plant candidates | plant --item <id> | purge [--yes]`

- [ ] **Step 0: Read** `eval/attack_corpus.py::to_gate_item`, `memory/types.py` and `memory/store.py`. If the class-3 `ExperienceRecord` is built inline, extract `to_experience_record` and keep `to_gate_item`'s output byte-identical. The item text is in the verdict cache key.

**Candidates invariants.**
- Entries come from `load(corpus_dir(), "test")`, which is hash-checked.
- Rows come from the fixed Task 46 path. A missing file raises `PlantError` naming the path.
- Rows with `reduced_assurance` true are excluded.
- The target question is the entry's own field. Read the corpus contract for its name; do not invent one.

**Plant invariants.**
- `require_frozen()` runs first.
- Refusals raise `PlantError` and write nothing: a `plant_file` that exists; an id not in the test split; an id that is not class 3.
- The record is `to_experience_record(entry)`, or the existing class-3 projection.
- `verdict = store.write(record, gate=gate)`. On a non-`ADMIT` verdict, raise `PlantError` with the verdict's action and reason. The store persisted nothing; it writes only on admit.
- On admit, write `plant_file` as JSON: `{"item_id", "record_id", "planted_ts", "target_question"}`. `planted_ts` is minted the way `record_from_run` mints `ts` (same format), so the string comparison in `purge_ids` is valid.
- The write gate makes no LLM call. The plant is free; the test asserts zero `chat` calls with a fake.

**Purge invariants.**
- With no `plant_file`, raise `PlantError("nothing planted")`. No backup, nothing removed.
- `purge_ids` returns the planted `record_id` plus every record whose `ts >= planted_ts`, in mirror order, deduplicated.
- Without `yes`: return the list and change nothing.
- With `yes`:
  1. Copy `mirror_path` to `<mirror_path>.<planted_ts with ':' replaced>.bak`. The copy happens first, before anything is removed.
  2. `store.purge(ids)`.
  3. Re-read `all_records()`. Any listed id still present raises `PlantError` naming it, and `plant_file` is kept.
  4. Otherwise delete `plant_file`.
- CLI exit codes: 0 on success, 1 on `PlantError` or `FreezeError`, with the message.

- [ ] **Step 1: Write the failing tests** (`tests/eval/test_demo_plant.py`)

Fixtures:
- A fake store with `write`, `all_records`, `purge` and a JSONL mirror in `tmp_path`, matching `tests/memory/` fakes.
- A fake write gate returning a chosen `Action`.
- A synthetic frozen corpus from `tests/eval/corpus_fixture.py`.
- Monkeypatched `require_frozen`.

Tests:
- `test_candidates_keeps_only_excluded_class3_judge_topped_test_rows`
- `test_candidates_skips_reduced_assurance_rows`
- `test_candidates_missing_screen_file_raises_naming_path`
- `test_plant_refuses_existing_plant_file_and_writes_nothing`
- `test_plant_refuses_non_class3_id`
- `test_plant_refuses_id_outside_test_split`
- `test_plant_refuses_unclean_freeze`
- `test_plant_non_admit_persists_nothing_and_raises_with_verdict`
- `test_plant_admit_writes_plant_file_with_four_keys`
- `test_plant_makes_no_chat_call`
- `test_purge_ids_include_planted_and_later_records_only`. An earlier record is kept.
- `test_purge_without_yes_changes_nothing`
- `test_purge_with_yes_backs_up_before_removing`. Assert the backup's bytes equal the mirror's bytes before the purge.
- `test_purge_reports_survivor_and_keeps_plant_file`. The fake `purge` leaves one id.
- `test_purge_without_plant_file_exits_1_and_changes_nothing`
- `test_cli_exit_codes`

If Step 0 extracted `to_experience_record`, add to `tests/eval/test_attack_corpus.py`:
- `test_class3_gate_item_text_unchanged_by_record_extraction`. Pin the item text for a fixture entry: the same value before and after.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/eval/test_demo_plant.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.demo_plant'`

- [ ] **Step 3: Implement** to the interfaces and invariants.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/eval/test_demo_plant.py tests/eval/test_attack_corpus.py -v` then `pytest tests/gate tests/memory tests/eval -q`
Expected: PASS

- [ ] **Step 5: Commit** (add `eval/attack_corpus.py tests/eval/test_attack_corpus.py` to both lists only if changed)

```bash
git add eval/demo_plant.py tests/eval/test_demo_plant.py
git commit -m "feat(eval): plant and purge one demo experience record through the real write gate" -- eval/demo_plant.py tests/eval/test_demo_plant.py
```

Do not run `plant` or `purge` against the real data directory. That is Tasks 68 and 69.

---

### Task 65: Class-1 trial, gate off and on — RUN, owner go required

**Files:**
- Modify: `README.md`. Add a new section "Phase 6: Class-1 Attack Success (Task 65, run <date>)" after "Phase 5: What the Adaptive Round Shows". Update the "Phase 3: Gate-Off Vs Gate-On Harnesses (built, not run)" table row for this harness and "Open measurements" item 2, for this run only.

- [ ] **Step 1: STOP.** Ask the owner for a go for this run, today. Name the command, both models, and that it spends budget. Do not continue without an explicit yes for Task 65.

- [ ] **Step 2: Freeze check**

Run: `JUDGE_MODEL=gemini-flash-lite-latest python -m eval.freeze_gate check`
Expected: `freeze check: clean`. If it is not clean, stop and report every line.

- [ ] **Step 3: Run**

```bash
LLM_MODEL=gemini-3.1-flash-lite JUDGE_MODEL=gemini-flash-lite-latest LLM_MIN_INTERVAL=2 python -m eval.class1_trial
```

On a `PerDay` error, stop: the file stays `complete: false`, and a rerun needs a new go on a new day.

- [ ] **Step 4: Check the output** (`data/class1_trial_results/results.json`)

`complete` is true, and `freeze.differences` is `[]`. Count rows whose gate-on verdicts ran with reduced assurance. If any did, report them to the owner before writing; do not rerun without a go.

- [ ] **Step 5: Write the README section** from the file, unrounded:
- Per trial: carrier in context (off/on), canary in answer (off/on), excluded chunks and their top signal.
- Totals per arm at both levels.
- Both models, the freeze check, the date.

A gate-off behavior total of 0 is reported as it is (spec R2).

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: record class-1 attack success, gate off vs on" -- README.md
```

---

### Task 66: Class-3 trial, `--arm off` and `--arm on` — RUN, owner go required

**Files:**
- Modify: `README.md`. Add a new section "Phase 6: Class-3 Attack Success (Task 66, run <date>)" after Task 65's. Update the Phase 3 harness table row for this harness, and "Open measurements" item 2, for this run.

- [ ] **Step 1: STOP.** Ask the owner for a go covering both arms, back to back, today. If Task 62 Step 0 found papers-era questions, restate the owner's choice; if they have not chosen, do not proceed.

- [ ] **Step 2: Freeze check** as in Task 65 Step 2.

- [ ] **Step 3: Run both arms**

```bash
LLM_MODEL=gemini-3.1-flash-lite JUDGE_MODEL=gemini-flash-lite-latest LLM_MIN_INTERVAL=2 python -m eval.experience_poison_trial --arm off
LLM_MODEL=gemini-3.1-flash-lite JUDGE_MODEL=gemini-flash-lite-latest LLM_MIN_INTERVAL=2 python -m eval.experience_poison_trial --arm on
```

If the quota ends the day between the arms, `--arm on` waits for a new go on a new day.

- [ ] **Step 4: Check the output** in the directory Task 62 Step 0 recorded. Both files are complete, and both have `freeze.differences == []`. Behavior is judged by hand against the pre-registered `check`, as in Phase 3 Task 29. Record each judgement with the answer text it rests on.

- [ ] **Step 5: Write the README section**, unrounded:
- Per question: poison retrieved, excluded (with top signal), reached the prompt, behavior judgement.
- The ungated control arm.
- Totals per arm at both levels.
- Both models, the freeze check, the output path.
- If the questions are papers-era by the owner's choice, say so in the section's first sentence.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: record class-3 attack success, gate off vs on" -- README.md
```

---

### Task 67: Overhead, off / cold / warm — RUN, owner go required

**Files:**
- Modify: `README.md`. Add a new section "Phase 6: Overhead (Task 67, run <date>)" after Task 66's. Update the Phase 3 harness table row and "Open measurements" item 2 for this run. If Tasks 65–67 are now all recorded, item 2 is closed with a pointer to the three sections.

- [ ] **Step 1: STOP.** Ask the owner for a go for this run, today.

- [ ] **Step 2: Freeze check** as in Task 65 Step 2.

- [ ] **Step 3: Run**

```bash
OVERHEAD_N=5 LLM_MODEL=gemini-3.1-flash-lite JUDGE_MODEL=gemini-flash-lite-latest LLM_MIN_INTERVAL=2 python -m eval.overhead
```

- [ ] **Step 4: Check the output** (`data/phase3_overhead/results.json`). It is complete, and `freeze.differences == []`. In `on-cold`, judge calls are above 0. If they read 0, the call-counting invariant broke (Phase 3 plan, Task 30): stop and report, do not record.

- [ ] **Step 5: Write the README section**, unrounded:
- Per arm: LLM calls and seconds, per question and in total.
- Added calls and added seconds per query, cold and warm separately, computed against `off`.
- `LLM_MIN_INTERVAL=2` named, because pacing inflates wall-clock.
- Both models, the freeze check.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: record read-gate overhead, cache-cold and cache-warm" -- README.md
```

---

### Task 68: Demo prep — plant and rehearsal — RUN, owner go required

No file changes.

- [ ] **Step 1: List candidates (free)**

Run: `python -m eval.demo_plant candidates`
Show the owner the list. The owner picks one id.

- [ ] **Step 2: STOP.** Ask the owner for a go for the rehearsal, today. Name the chosen id and its target question. The plant itself is free; the rehearsal spends budget.

- [ ] **Step 3: Plant (free)**

Run: `JUDGE_MODEL=gemini-flash-lite-latest python -m eval.demo_plant plant --item <owner's id>`
Expected: exit 0 and `data/demo_plant.json` written. On a non-admit exit, report the verdict and return to Step 1 with another id.

- [ ] **Step 4: Rehearsal**

The owner runs:

```bash
UI_READ_GATE=1 UI_EXPERIENCE_STORE=1 LLM_MODEL=gemini-3.1-flash-lite JUDGE_MODEL=gemini-flash-lite-latest LLM_MIN_INTERVAL=2 python app.py
```

They ask the target question once. Check the panel:
- The planted record appears with store `experience` and an action other than `admit`.
- Note its top signal and evidence span.
- Note whether any row ran with reduced assurance. A reduced-assurance verdict is not cached, so the recording would re-screen it; rerun the rehearsal under the same go.

- [ ] **Step 5: If the planted record was admitted:** run `python -m eval.demo_plant purge`, show the list, then `purge --yes` with the owner's yes. Return to Step 1 with another id; it needs a new go for its rehearsal.

- [ ] **Step 6: Report** the planted id, the panel's verdict for it, and that the cache now holds the question's verdicts.

---

### Task 69: Demo recording; purge

No file changes.

- [ ] **Step 1: Owner records.** Start with both flags on and the same environment as the rehearsal. The owner records:
  - the target question, its answer, and the panel showing the planted record excluded with its S2 evidence;
  - one or two questions from `eval/demo_questions.txt`, with their chunks admitted.

  The recording's location is the owner's choice; it is not committed.

- [ ] **Step 2: Show the purge list (free)**

Run: `python -m eval.demo_plant purge`
Show the owner the ids: the planted record plus every record the rehearsal and the recording wrote.

- [ ] **Step 3: STOP.** The purge is destructive. Ask the owner for an explicit yes.

- [ ] **Step 4: Purge**

Run: `python -m eval.demo_plant purge --yes`
Expected: exit 0, a `.bak` beside the mirror, `data/demo_plant.json` gone. On exit 1, report the surviving id and the backup path, and stop.

- [ ] **Step 5: Verify** the mirror holds none of Step 2's ids. Use `python -c` over `config.EXPERIENCES_PATH`, comparing against the printed list. Report the result.

---

### Task 70: Evaluation report

**Files:**
- Create: `docs/report.md`

No code, no LLM call. Follows Tasks 65–69.

**Sourcing invariants.**
- Every number is copied from a README results section, which was written from command output. Each figure cites that section by heading.
- No number the README does not hold. Nothing rounded. No payload text from any attack item.
- Models from README sections and `eval/frozen_gate.json` only.

**Sections.** Exactly the spec's §10 list, 1–11, in order:
1. Problem and threat model
2. Design
3. Setup
4. Attack success (§9.1): class 1 from Task 65, class 2 from Tasks 47 and 60, class 3 from Task 66. Use the "ASR X → Y at +Z ms/query" shape where Task 67's overhead allows it.
5. Detection and ablation (§9.2): Task 46 beside Task 59.
6. Clean-task regression (§9.3), stated as not measured by owner decision, 2026-10-15.
7. Overhead (§9.4)
8. False positives
9. The adaptive round
10. Limitations, including every "Known Phase 3+ concerns" item by name
11. Future work

- [ ] **Step 1: Write `docs/report.md`.**

- [ ] **Step 2: Figure check.** For every number in the report, find it in the cited README section and confirm the characters match. List any mismatch and fix the report, never the README.

- [ ] **Step 3: Commit**

```bash
git add docs/report.md
git commit -m "docs: Phase 6 evaluation report" -- docs/report.md
```

---

### Task 71: Cleanup decision — STOP, owner decides

**Files:** only those the owner picks, from the spec's §11 candidates C1–C13.

- [ ] **Step 1: Present the candidates.** For each of C1–C13, show the current text or state:
  - C1–C6: the README lines as they read now, with line numbers.
  - C7: the Phase 1 plan's Phase 6 table.
  - C8: the model lines in the Phase 4 and Phase 5 specs.
  - C9 and C10: `git diff .env.example .gitignore`.
  - C11 and C12: each path and its line count.
  - C13: the `CLAUDE.md` status paragraphs.

  Also show the proposal recorded in the spec.

- [ ] **Step 2: STOP.** The owner picks which candidates are done. C11 and C12 each need their own explicit yes, because the files are untracked and git cannot restore them.

- [ ] **Step 3: Do the picked items only.** One commit per file touched, by pathspec, `docs:` or `chore:`:
  - C8 adds a dated note under each model line and leaves the original line in place.
  - C9 adds `JUDGE_MODEL=gemini-flash-lite-latest` with a one-line comment that the freeze checks it.
  - C11 and C12 are deleted with `rm`, not `git rm`: they are untracked, so there is nothing to commit.
  - C13 edits the local, gitignored `CLAUDE.md`: no commit.

- [ ] **Step 4: Record the decision.** Report to the owner one line per candidate: done, or left. If any README item was done, put the same list in that commit's body.

---

### Task 72: Merge to `main` — STOP

- [ ] **Step 1: Check fast-forward**

Run: `git merge-base --is-ancestor main phase-4-s3-corroboration && echo ff-ok`
Expected: `ff-ok`. Otherwise `main` has moved: stop and ask the owner.

- [ ] **Step 2: STOP.** Ask the owner for a go to move `main` to the branch head. Name both commits.

- [ ] **Step 3: Fast-forward without checking out**

Run: `git fetch . phase-4-s3-corroboration:main`
This refuses anything that is not a fast-forward. The working tree, and any loose files the owner chose to keep, are untouched.

- [ ] **Step 4: Verify**

Run: `git rev-parse main phase-4-s3-corroboration`
Expected: two identical hashes.

---

### Task 73: Push — STOP

- [ ] **Step 1: Verify the remote**

Run: `git remote -v`
Find the remote whose URL is `sk-0607/rag-mem-sentry`. If none matches, or the only candidate points at `self-correcting-rag`, stop and ask.

- [ ] **Step 2: STOP.** Ask the owner for a go to push `main`, `phase-4-s3-corroboration` and the tag `gate-frozen-1` to that remote. Name the remote and the URL.

- [ ] **Step 3: Push**

```bash
git push <remote> main phase-4-s3-corroboration
git push <remote> gate-frozen-1
```

- [ ] **Step 4: Verify**

Run: `git ls-remote <remote> main phase-4-s3-corroboration gate-frozen-1`
Expected: three refs, with hashes equal to the local ones.

---

## Phase 6 exit criteria

- [ ] `pytest -q` green; `pytest tests/gate tests/memory tests/eval` offline and free.
- [ ] Tasks 65–67 each have a README section with both models, the freeze check, and unrounded figures. "Open measurements" item 2 is closed or says exactly what stayed open.
- [ ] The demo is recorded. The experience store's mirror holds no planted or demo-run record, and `data/demo_plant.json` is gone.
- [ ] `docs/report.md` covers the spec's §10 sections, with §9.3 marked not measured, and passes Task 70's figure check.
- [ ] Task 71 recorded, for each of C1–C13, whether it was done or left.
- [ ] `main` equals the branch head. `main`, the branch and `gate-frozen-1` are on `sk-0607/rag-mem-sentry`, if the owner gave the go.
