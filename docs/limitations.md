# Limitations

## Limitations

- Sub-queries run sequentially (parallel fan-out is future work).
- Decomposition capped at 3 sub-questions.
- No retrieval-quality check or refusal: out-of-scope questions are answered from
  whatever the top-5 chunks are, so answers can be ungrounded.
- Corpus is a small BEIR NQ subset: 50 questions, 1726 documents, 1784 chunks. Questions
  outside those Wikipedia articles are answered from whatever is retrieved.
- Free-tier request limits pace the eval run; it is serialized to stay within
  budget.
- Security results are **provider-dependent**, and two different Gemini models
  produced them: R3 (2026-08-19) ran against `gemini-flash-lite-latest`, the
  repository default in `app/config.py`; the Phase 2 calibration and poisoning
  demonstration ran against `gemini-3.1-flash-lite`, set via `.env`'s
  `LLM_MODEL`. From Phase 4 on, the gate ran on `gemini-flash-lite-latest` (frozen in
  `eval/frozen_gate.json`) and the trials answered with `gemini-3.1-flash-lite`. Each section
  names the model that produced its numbers. The
  finding that the model itself declines injected instructions at answer time
  is that provider's alignment, not a property of this pipeline, and a more
  compliant model would move the numbers.
- **S3 needs an owner-trusted set.** With no source marked trusted, S3 abstains on everything
  and only S1 and S2 apply. A deployment made entirely of user uploads has no trusted set.
- **S3 does not check a claim the trusted set does not cover.** It reports `not_covered` rather
  than guessing, so a false fact on a topic the trusted sources are silent about passes S3. How
  often that happens is what the not-checked counts are for. On clean untrusted content (Task 44,
  papers) 24 of 40 chunks came back `not_covered`; on the blind test split (Task 46) 21 class-1
  and 32 class-2 items did.
- **S3 checks at most `S3_MAX_CLAIMS` (5) claims per chunk.** A false claim outside the
  extracted claims is not checked, and the chunk can still read as supported. No rate has been
  measured.
- **Trust is asserted by the owner, not verified.** S3 never checks trusted content, so a
  poisoned source the owner marks trusted poisons the reference. S1 still screens it at write
  time and S2 at read time. The flag is a manifest entry, not signing or lineage.
- **The adaptive round's black box is a process boundary, not enforcement.** The attacker agent
  runs on the owner's machine and could in principle read this repository. Its brief forbids it
  and the owner checks its tool log after each session; a breach voids the round.
- **Probe timing leaks.** An uncached probe takes longer when S3 reaches its agreement step than
  when it stops early, and a cached probe is fast. An attacker that times requests learns which
  path ran. Not mitigated.
