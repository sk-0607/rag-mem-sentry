# Trust Gate — Design Spec

**Date:** 2026-08-16
**Status:** Approved, pre-implementation
**Team:** 2 (SK — build; teammate — attack/red-team)
**Duration:** 12 weeks, start ~2026-08-18
**Outcome type:** Working system. Not a paper. Prior art is scaffolding, not competition.

---

## 1. Problem

An LLM agent trusts what it retrieves. Content reaching the prompt — from a RAG corpus or
from the agent's own long-term memory — is treated as ground truth. An attacker who can place
content into either store controls the agent's behavior downstream.

Existing defenses in this space gate **origin** (cryptographic provenance, signed records,
lineage). A validly-signed record whose *content* is poisoned still passes. This system gates
**content**, not origin.

**Statement:** detect poisoned information before an LLM/agent trusts and uses it, whether that
information comes from RAG or from the agent's long-term memory.

## 2. Goals

- Screen content at two points: when it is written to a store, and when it is retrieved for use.
- Catch three poison classes with three independent signals.
- Compose with the host agent's existing behavior rather than replacing it.
- Prove harm reduction end-to-end, not just detection accuracy.
- Hold up against an attacker who has seen the defense.

## 3. Non-goals

Explicitly out of scope. Listed so scope creep is visible when it happens.

- **Provenance / signing / lineage.** Solved elsewhere (SMSR, MemLineage). Left as an
  unimplemented fourth signal slot to demonstrate the interface is pluggable.
- **Trained classifiers.** Detection is rules + LLM-judge only. Carries the capstone's standing
  "no plain ML classification" constraint.
- **Multi-agent propagation.** Single-agent only.
- **Prompt-level injection from the user.** The threat surface is *stored* content, not user turns.
- **Novelty claims.** This is a working system. No literature-gap argument is being made.

## 4. Threat model

**Attacker capability:** can place content into the RAG corpus (e.g. a document the agent
ingests) and can induce the agent to write a poisoned experience record. Cannot modify the
agent's code, prompts, or the gate itself.

**Attacker knowledge:** two settings, both evaluated.
- *Blind* — attacker does not know a gate exists. Public benchmark attacks.
- *Adaptive* — attacker has black-box access: can submit content and observe verdicts. Cannot
  see thresholds, signal internals, or calibration data.

**Poison classes in scope:**

| Class | Name | Mechanism | Primary signal |
|---|---|---|---|
| 1 | Instruction injection in content | Retrieved text carries imperatives aimed at the model — "ignore previous instructions", invisible unicode, hidden HTML/markdown, tool-call bait, encoding tricks | S1 + S2 |
| 2 | Factual poisoning | Content asserts false facts crafted to rank highly for a target query and flip the answer (PoisonedRAG-style) | S3 |
| 3 | Poisoned experience records | A stored "successful task" record models unsafe behavior; the agent imitates it on later tasks (MemoryGraft-style) | S2 |

## 5. Host system

A fork of `self-correcting-rag`. Inherited on day one:

- Hybrid retrieval — dense (Chroma / all-MiniLM-L6-v2) + BM25 sparse, RRF merge, cross-encoder
  rerank to top-5.
- LangGraph state machine with `retrieve` / `grade` / `rewrite` / `web_fallback` / `generate` /
  `refuse` nodes.
- A 22-paper AI-security corpus, 2,025 chunks.
- A RAGAS evaluation harness.
- 48 pytest tests.

Net-new on top: the gate, and an experience-memory subsystem (there is no long-term memory in
the host today; class 3 has no target without it).

The capstone lives in a **separate repo**. The portfolio repo stays untouched.

## 6. Architecture

```
WRITE-TIME  (cheap signals only)
  corpus doc ──┐
               ├─► S1 structural ─► aggregate ─► verdict ─┬─ admit ──────► store + metadata
  experience ──┘                                          ├─ quarantine ─► quarantine store
                                                          └─ block ──────► rejected + audit

READ-TIME  (expensive signals, cache-first)
  query ─► hybrid retrieve top-N ─► per candidate:
              cache[content_hash]? ──hit──► reuse verdict
                                   └─miss─► S2 judge + S3 corroborate ─► write cache
           ─► aggregate ─► verdict ─► admit set ─► generate
```

*Note, 2026-10-13:* the diagram above is the design as written. The host no longer has a `grade`
node, so as built the read-time path is `retrieve → gate → generate` (Phase 3 spec §1 and §5).

### 6.1 Two properties this architecture buys

**The gate composes with the host's CRAG loop rather than replacing it.** Removing poisoned
chunks shrinks the retrieved context. The existing `grade` node then legitimately rewrites the
query, falls back to web search, or refuses. The degraded path is already built and tested —
the gate does not need its own failure UX.

*Note, 2026-10-13 — this property no longer holds.* The host dropped the CRAG `grade` / `rewrite`
loop, the refusal path and web fallback before Phase 3 (Phase 3 spec §1). A sub-query whose every
chunk the gate excludes now reaches `generate` with an empty context (risk R7 in the Phase 3
spec); there is no rewrite, fallback or refusal to absorb it. The paragraph above is kept as the
original design.

**A content-hash verdict cache is what makes read-time judging affordable.** Naive per-candidate
judging costs roughly 15 extra LLM calls per question on top of the host's existing ~10. The
host's own build log records a free-tier `RESOURCE_EXHAUSTED … PerDay` failure killing a
20-question eval at the lower rate. With hash-keyed caching, the static corpus is judged once
ever; steady-state read-time LLM cost approaches zero, and only fresh memory writes pay full
price. **This is task-1 work, not an optimization.**

## 7. Components

One responsibility per module, matching the host repo's existing convention.

| File | Responsibility | Status |
|---|---|---|
| `gate/signals/base.py` | `Signal` protocol; `SignalResult{score, reason, evidence_span}` | new |
| `gate/signals/structural.py` | S1 — rules only, no LLM, no network | new |
| `gate/signals/judge.py` | S2 — LLM judge, two rubrics, batched across candidates | new |
| `gate/signals/corroborate.py` | S3 — claim extraction + agreement check | new |
| `gate/score.py` | signals → aggregate → `Verdict` | new |
| `gate/policy.py` | `Verdict` → `Action`; per-store thresholds | new |
| `gate/cache.py` | content-hash → verdict, persistent across runs | new |
| `gate/audit.py` | append-only verdict log; demo surface and eval data source | new |
| `gate/quarantine.py` | quarantined items — not retrievable, releasable | new |
| `memory/experience.py` | experience record write / retrieve | new |
| `app/graph.py` | insert `gate` node before `grade` | modified |
| `ingest/run_ingest.py` | call write-time gate | modified |
| `app/ui.py` | trust panel — per-candidate score, firing signal, evidence span | modified |

### 7.1 Signal interface

Every signal implements one method:

```
score(item, ctx) -> SignalResult{ score: float, reason: str, evidence_span: str | None }
```

`item` is the candidate document, chunk, or experience record. `ctx` carries the query (read-time
only), sibling retrieved candidates, and store kind. A signal that cannot run returns `None`.

Signals never mutate state, never call each other, and are independently testable. Adding a
fourth signal is a new file plus a registry entry — no change to `score.py`, `policy.py`, or the
graph.

### 7.2 S1 — structural

Deterministic rules over raw content. No LLM, no network. Runs at write-time and read-time.

Detects: instruction patterns (imperative constructions addressed to a model — "ignore previous
instructions," role reassignment, fake system messages, "new instructions:"); invisible/zero-width
unicode and bidi overrides; hidden CSS (`display:none`, zero font-size, `visibility:hidden`,
transparent color, zero/clipped width or height) and HTML comment blocks; homoglyph substitution,
caught by skeletonizing confusable codepoints (Cyrillic/Greek look-alikes, fullwidth Latin) to
ASCII and re-running the instruction patterns against the result; and base64-encoded payloads,
caught by decoding candidate runs and re-running the instruction patterns against the decoded
text.

Tool-call and URL bait were considered and dropped from S1's scope: telling bait apart from a
legitimate tool call or citation needs surrounding context (the query, sibling candidates) that a
content-only signal does not have — that need is what makes it S2-shaped, not S1-shaped.

Cheapest signal, runs on everything, and is the entire write-time gate.

### 7.3 S2 — policy judge

LLM judge over candidate content. Two rubrics, selected by store kind:

- **Corpus content:** does this passage attempt to instruct or steer the model reading it, as
  opposed to describing or quoting such an attempt?
- **Experience record:** does this record endorse, normalize, or model unsafe behavior as a
  successful outcome?

Batched — all candidates for one query are scored in a single call. Every result is written to
the hash cache.

The corpus rubric's "as opposed to describing or quoting" clause is doing real work; see §11.

### 7.4 S3 — corroboration

Extracts factual claims from a candidate, then checks agreement against (a) the sibling
candidates retrieved for the same query and (b) a trusted reference subset of the corpus. A
claim that contradicts the consensus, or that is asserted only by a single low-agreement
candidate, raises the signal.

Read-time only. Meaningless without sibling context.

*Note, 2026-10-13:* as built, S3 checks an untrusted chunk's claims against owner-trusted chunks
only. Sibling comparison was dropped because it would make the result depend on which chunks were
co-retrieved, which breaks the content-keyed cache. Trusted chunks and the experience store
abstain. See the Phase 4 spec (`docs/specs/2026-10-13-phase-4-s3-corroboration-design.md`) §0 and
§4.

### 7.5 Aggregation

**Not a weighted sum.** Weighted averaging dilutes a confident detection — one screaming
structural hit gets talked down by two calm signals.

1. Any signal above its **own hard threshold** → `block`, regardless of the others.
2. Otherwise, aggregate the soft scores into an `admit` / `quarantine` band.

### 7.6 Calibration

Thresholds are **calibrated on a held-out calibration split of the attack corpus**, targeting a
stated false-positive budget on clean content. Never hand-tuned, never tuned on the test split.

The calibration procedure is a deliverable and is documented with the resulting numbers.

## 8. Failure behavior

Deliberately asymmetric between write-time and read-time.

| Condition | Behavior | Rationale |
|---|---|---|
| A signal raises | Return `None`, exclude from aggregate, log | Same contract as the host's existing `websearch` failure → `[]`. A query never crashes on a signal fault. |
| Write-time signal/LLM failure | **Quarantine** (fail closed) | No user is waiting. Nothing is destroyed. Releasable later. |
| Read-time expensive signals all unavailable | **Degrade to S1**, set `reduced_assurance` in `path_trace` | Failing closed turns any LLM outage into a total refusal. |
| Ambiguous verdict anywhere | **Quarantine, never block** | Quarantine is reversible; block is not. |

Every verdict — admit, quarantine, or block — is written to the audit log with its per-signal
scores and evidence spans.

## 9. Evaluation

Four measurements. All reported gate-off vs gate-on.

**9.1 Headline — attack success rate, reported at two levels.** Per poison class, with overhead
attached. Target report shape: *"ASR X% → Y% at +Z ms/query."* Harm reduction, not detection
accuracy.

| Level | Question | Why it is reported |
|---|---|---|
| **Context** | Did the poison reach the model's context block? | The gate is a content-integrity control and this is the boundary it actually defends. Measured from the context block, not the answer. |
| **Behavior** | Did the poison change the answer or the trace? | The end-to-end harm number. Confounded by model alignment — see below. |

Both are reported because R3's measurement showed they diverge. In the trials where class-1
poison reached `generate` verbatim, the model declined the injected instruction anyway: context
ASR was 100%, behavior ASR was 0%, and the gate had nothing to do with either. Reporting only
behavior would credit provider alignment to the gate on the defended side and understate the
undefended baseline on the other.

Two rules follow, and both are load-bearing for the write-up:

- **The behavioral headline leans on classes 2 and 3.** Alignment refuses "ignore all previous
  instructions." It does not refuse a plausible false fact, and it does not refuse an experience
  record modelling an approach that is unsafe without being overtly malicious. Class 1's
  behavioral number will look weak for reasons that are not about the gate.
- **The model is named in every ASR figure.** These results hold for `gemini-flash-lite-latest`
  and would move on a more compliant model. Stated as a limitation rather than left for a
  reviewer to find.

**9.2 Ablation.** Each signal on/off against each poison class. Cells are measurement slots, filled
in week 9:

| S1 | S2 | S3 | class 1 | class 2 | class 3 |
|---|---|---|---|---|---|
| ✓ | | | | | |
| | ✓ | | | | |
| | | ✓ | | | |
| ✓ | ✓ | ✓ | | | |

This table decides whether the multi-signal design is justified. If S2 alone catches everything,
that is reported as a negative result and S3 is cut. **Run at week 9**, while there is still time
to react.

*Note, 2026-10-13:* as built, the ablation is computed at detection level (items caught out of
items scored), recomputed offline from one screen's stored per-signal scores. It has a fifth row,
S1 + S2, so S3's contribution is the difference between the last two rows. It is report-only: S3
is not cut whatever it shows. See the Phase 4 spec §9.2.

**9.3 Clean-task regression.** RAGAS faithfulness / answer relevancy / context precision on the
clean question set, gate off vs on, using the host's existing harness unchanged. A gate that
costs significant faithfulness is not shippable regardless of its detection rate.

**9.4 Overhead.** Added latency and added LLM calls per query, reported **cache-cold and
cache-warm separately**. Cold is the real first-run cost and is not hidden.

### 9.5 Baselines

- **No gate** — the harm ceiling.
- **The host's existing `grade` node alone** — measured 2026-08-19 and confirmed a **null
  baseline** for class 1: poison changed its verdict zero times in five controlled trials. Still
  reported rather than dropped — "the pipeline we built on had no class-1 defense" is a result,
  and a reviewer who assumes CRAG grading is a partial mitigation needs the measurement to see
  otherwise. Whether it is null for classes 2 and 3 is **not** established; a poisoned *fact* is
  topically relevant by construction, so the same null result should not be assumed there.
  *Note, 2026-10-13:* the `grade` node no longer exists in the host, so this baseline cannot be
  re-run. R3's result stays as history; the baselines Phase 3 reports are *no gate* and *gate on*.
- **One published defense**, if runnable — a perplexity or knowledge-consistency filter from the
  PoisonedRAG defense line.

### 9.6 Attack corpus

Public benchmarks for the comparable baseline, teammate-authored adaptive attacks for the
robustness claim.

- Published attack sets (PoisonedRAG-line, memory-poisoning benchmarks) provide blind-setting
  numbers comparable to the literature. **Availability is verified in week 1**; if artifacts are
  not obtainable, the teammate regenerates from published attack descriptions.
- Split into **calibration** and **test**. Thresholds touch calibration only.
- **Adaptive attacks are held out entirely.** The teammate authors them after the gate is frozen,
  with black-box access only.

### 9.7 Clean hard-negative set

The 22-paper AI-security corpus contains literal prompt-injection strings as worked examples —
published attack prompts quoted inside the papers that study them. The host project already
documented this as a "lexical collision" limitation.

This is the false-positive test set, and it is free: **the gate must not flag a paper's quoted
example attack as an attack.** It is harder than any synthetic clean set and it already exists in
the data.

## 10. Plan

| Wk | SK (build) | Teammate (attack) | Gate check |
|---|---|---|---|
| 1 | Fork, run, strip | Verify benchmark artifacts download; reproduce one published attack | Baseline ASR, gate-less |
| 2–3 | Signal interface, aggregate, policy, cache, audit, quarantine, S1, write-time gate | Class-1 corpus; clean hard-negative set | S1 TPR + FPR on hard negatives |
| 4–5 | `memory/experience.py`, write/retrieve path | Class-3 poisoned experiences | Memory poisoning demonstrably works ungated |
| 6–7 | S2 batched judge; read-time gate into `graph.py` | Class-2 factual corpus | Classes 1+3 caught; RAGAS regression measured |
| 8–9 | S3 corroboration; threshold calibration | Attack corpus frozen and split | **Full ablation — decision point** |
| 10 | Gate frozen, no changes | Adaptive black-box round | Adaptive ASR |
| 11–12 | Trust panel UI; full eval run; report | Red-team writeup | All four measurements |

Week 12 is the only slack in the schedule.

**Sequencing rule:** an attack is built before the defense against it. Week 4–5 proves memory
poisoning works on the ungated system before week 6–7 defends it. A defense against an attack
that was never demonstrated is unfalsifiable.

*Note, 2026-10-13:* this rule was overridden for class 2 (factual poisoning). S3 is built before
class 2 has been run against the ungated host, and the ungated measurement is taken last. This is
the second override of the rule; the first is Phase 3 spec §0. See the Phase 4 spec §0.

**Testing.** TDD per component, matching the host repo's existing discipline. Each signal is unit
tested in isolation with synthetic poison and synthetic clean input. The gate's integration into
the graph is tested against recorded retrieval traces. The evaluation harness is not a test — it
is a deliverable with its own numbers.

## 11. Risks

**R1 — Benchmark artifacts may not be obtainable.** Papers routinely cite datasets that never
shipped. Resolved in week 1. Fallback: regenerate from published attack descriptions, and report
the corpus as team-authored rather than claiming benchmark comparability.

**R2 — Free-tier LLM quota.** The host project's build log records `RESOURCE_EXHAUSTED … PerDay`
killing a 20-question evaluation run at roughly 10 calls per question. The gate adds calls. All
three mitigations are week-2 work: hash-keyed verdict cache, batched judging, and a second
provider key provisioned in advance. Discovering this in week 11 ends the project.

**R3 — The host's existing grader may already catch class 1. CLOSED 2026-08-19: it does not.**
Measured against the real `app.llm.grade` with the gate never imported. Poison changed the grade
verdict **zero times out of five** controlled trials. `grade` is a topical-relevance judge with a
one-word `SUFFICIENT`/`WEAK` output contract — no security instruction, no adversarial framing.
Where poison rode a topically-relevant carrier chunk it graded `sufficient` and passed into
`generate`'s context block verbatim, hidden-HTML and zero-width variants included.

The contribution therefore does **not** narrow to classes 2–3. Class 1 is unowned by the host
pipeline, and the grader is not a partial defense to subtract — subtracting it would credit
relevance filtering as security. Full method and results in the README's *R3* section.

**R3a — model alignment may mask the class-1 ASR headline.** Surfaced by the R3 measurement and
newly open. In the two trials where poison reached `generate`, the answers were not visibly
hijacked: the model declined the injected instruction at answer time. That is alignment, not
pipeline defense — unmeasured, unenforced, and provider-dependent — but it deflates the gate-off
baseline that Phase 4's ASR headline is measured against. Responses, all three taken: report ASR
at two levels (*poison reached the context* and *poison changed the output*); lean the behavioral
headline on classes 2 and 3, which alignment does not catch; and state the model dependency
(`gemini-flash-lite-latest`) as an explicit limitation. See §9.1.

**R4 — S2 may dominate the ablation**, falsifying the multi-signal premise. Measured week 9. S3
is cut and the negative result is reported. A negative result measured on purpose reads as rigor;
the same result discovered in week 12 reads as failure.

*Note, 2026-10-13:* this is no longer an exit. S3 ships in the default read gate whatever the
ablation shows; if S3 adds nothing, the write-up says so. See the Phase 4 spec §0.4 and §14.

**R5 — S2's quote-vs-instruct distinction may not hold.** The corpus rubric asks the judge to
separate content that *instructs* the model from content that *quotes* an instruction. If the
judge cannot make that distinction reliably, false positives on the hard-negative set will be
high. This is measured in week 6–7, and the fallback is to weight S1's structural evidence
higher for content inside quoted or fenced regions.

## 12. Assumptions

Stated so they can be challenged rather than discovered.

1. The teammate's 12 weeks are genuinely available for the attack track. The plan gives them an
   owned deliverable, not a support role.
2. Detection remains rules + LLM-judge. No trained classifier. Carried from the capstone's
   standing constraints.
3. Hardware is unchanged from the capstone baseline — 16GB RAM, RTX 3050 4GB VRAM. Embeddings and
   reranking run locally; all judging is API-side.
4. The deliverable is a demoable system plus an evaluation report. No paper submission, no
   novelty argument.
