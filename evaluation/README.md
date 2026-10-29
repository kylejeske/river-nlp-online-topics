# Contextual theme and salience evaluation

This is a labeled diagnostic benchmark, not a new topic model. It runs the
unchanged `find_themes.py` as the vocabulary baseline. No model API, network
access, pretrained weights, or additional runtime package is needed after
installation. The runner and metrics use Python's standard library.

## Reproduce

From the repository root, with CPython 3.12 (reference run: 3.12.14):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eval.lock
python -m unittest evaluation.test_evaluation -v
python -m evaluation.run --out evaluation/runs/baseline
```

`evaluation/baseline/` holds the recorded default run. Each run writes:

- `predictions.json`: all assignments, unsupported outputs, and rankings.
- `metrics.json`: counts, per-scenario scores, checkpoint rankings, failed probes.
- `manifest.json`: suite/evaluator hashes, baseline hashes, versions and settings.
- `report.md`: a readable report; no aggregate pass/fail accuracy claim.
- `raw/`: actual CLI inputs, CSV/JSON outputs and stdout for every replay.

The small recorded outputs are committed. Reproducible raw outputs stay under
the ignored `evaluation/runs/` directory. The baseline fingerprint in
`baseline-lock.json` pins source, original samples, outputs, and verification
to commit `7f8691d846cbd102e5a5ad5d279a96fcbca6e1c6`. A changed baseline fails
fast; implement improvements as candidate adapters. Root README edits are
allowed. Cluster IDs are arbitrary and never matched to gold label strings.

Check threshold sensitivity without changing the baseline:

```bash
python -m evaluation.run --radius 0.45 --out evaluation/runs/r045
python -m evaluation.run --radius 0.70 --out evaluation/runs/r070
```

`sensitivity.json` records those control runs, with hashes linking the exact
suite, evaluator and predictions. These are diagnostic settings, not tuned
held-out winners.

## Ground truth and annotation policy

`fixtures/contextual-v1.json` has 8 ordered scenarios, 64 messages, explicit
pair/transition probes, and 18 checkpoints. All messages are synthetic; the
12 original sample messages are included verbatim. Labels were specified
before inspecting baseline predictions. They are one author's expectations,
not an independently adjudicated corpus. The entire suite is visible and is
a development set: do not claim generalization from improvements on it.

| Scenario | What a correct system must do |
| --- | --- |
| Original sample | Keep failure and resolution in one subject, while updating invoice/build state |
| Semantic paraphrases | Link differently worded descriptions; separate retail checkout from book lending |
| Conversational references | Use reply/thread context across interleaved turns, including stopword-only replies |
| Recurring issue | Preserve a subject, close episode 1, open episode 2 after recovery |
| Singleton critical | Rank a valid, publicly leaked production signing key above frequent routine chatter |
| State and negation | Separate hypothetical/healthy statements from incidents; do not accept unverified recovery |
| Similar vocabulary, distinct incidents | Separate staging and production; staging success cannot resolve production |
| Ambiguous reference | Retain explicit uncertainty when “that's fixed” has no identifiable referent |

Each message has an ID, timestamp, thread, optional reply target, text, and:

- `theme`: the concrete problem/subject identity at this suite's chosen
  granularity. A theme survives recovery and recurrence. Independent staging
  and production incidents are distinct subjects, even if a broader “builds”
  category could contain both. Neutral conversations are also subjects.
- `episode`: an incident occurrence. Repeated reports and the eventual
  resolution belong to one episode; a verified recovery followed by a new
  failure starts another. `null` means no incident episode is asserted.
- `state`: the subject's state supported by this observation: `active`,
  `resolved`, `none` (routine, healthy, or hypothetical), or `unknown` (an
  explicit ambiguity). Missing candidate support is `null`, not `unknown`.

State labels describe what the stream supports, not objectively verified
system health. “Might be fixed, recovery not verified” remains active under
this policy; a proposed retry is not recovery. No reply context is guessed
when two plausible referents remain. Thread and reply metadata are valid
evidence when supplied. The legacy text CLI ignores that metadata; its loss
is intentionally visible, not patched with gold labels.

Checkpoint `relevance` covers exactly the subjects seen so far:

| Grade | Operational interpretation |
| ---: | --- |
| 0 | Routine/healthy/hypothetical, resolved, or ungrounded ambiguous content |
| 1 | Active follow-up with limited stated impact, e.g. printer or invoice |
| 2 | Confirmed release/workflow blocker needing action |
| 3 | Stated customer-wide impact or active production credential exposure |

Frequency alone never raises a grade. A resolution reduces current action
priority to 0 without deleting history; recurrence can raise it again. The
signing key is one unique critical event evaluated both immediately and
after a routine follow-up, not two independent events. This rubric measures
operational salience, not personal relevance, positive milestones, or every
possible meaning of “important.”

## What is scored

Keep the measures separate: they answer different questions and cannot be
collapsed into a trustworthy single accuracy number.

1. **Theme grouping:** label-invariant pairwise precision, recall and F1
   over every message pair. Overmerging raises false positives; fragmentation
   raises false negatives. Final-snapshot F1 is macro-averaged by scenario.
   A separate theme coverage denominator includes token-filtered messages.
   No-positive-pair cases have undefined F1 (`null`), excluded from macro F1.
2. **Targeted contextual probes:** same/different theme, same/different
   episode, and state transitions. Missing IDs never pass a probe, including
   “different” probes. A transition requires both correct event states and
   theme continuity. Counts and failing message IDs accompany every score.
3. **Episode grouping:** pair F1 among observations with gold incident
   episodes, plus per-scenario episode-output coverage. This distinguishes
   recurring failures from one long uninterrupted issue. The baseline has
   no episode output and receives zero coverage.
4. **State:** per-message coverage and exact accuracy, plus per-class counts.
   Unsupported states fail end-to-end accuracy; supported-only accuracy is
   `null` if nothing is supported. At each checkpoint, current issue-state
   accuracy checks the candidate state of the predicted cluster containing
   each subject's latest evidence. The evaluator identifies that evidence
   using gold; the adapter receives no gold identity or state.
5. **Salience ranking:** purity-discounted, duplicate-suppressed nDCG@3 at
   checkpoints. Gold gain is `2^grade - 1`. Each predicted cluster maps to its
   dominant gold subject by member count, with ties resolved by earliest
   member; multiply its gain by cluster purity. Only the first appearance
   of a subject earns gain, so split clusters cannot earn repeated credit.
   The ideal ranking uses the observed gold subjects. This is an explicitly
   adapted cluster-ranking metric, not unmodified textbook nDCG. Empty
   rankings score 0 where there is positive relevance. All-zero checkpoints
   are reported as N/A; ranking alone does not measure alert suppression.
6. **Critical singleton strict top-1:** the critical singleton must be a pure
   one-message cluster and strictly outrank every competitor. A tied top
   score does not count as reliable prioritization. Counts are over eligible
   checkpoints; there are two in this suite.

Ranking ties always use first assigned message order, never gold severity.
Tie flags are shown because favorable arrival order can produce high nDCG
by accident. nDCG must be read alongside grouping, coverage, and singleton
scores. A model can rank the first paraphrase highly while failing to connect
any of its later references. The baseline ranking is explicitly a **frequency
proxy**, not a salience capability that River claims to implement. State and
episode zeros likewise indicate missing capabilities, not errors in a
classifier that was never present.

## Final snapshots versus available-prefix evaluation

For compatibility, final grouping learns the whole scenario once and assigns
all messages against the final model, exactly as the original CLI does.
For each checkpoint the runner launches a fresh process over only that
prefix: no future messages, gold labels, future corpus length, or relevance
grades reach the adapter. The original corpus-scaled fading and cleanup
settings are recomputed for that prefix, with the original default radius
0.60, window 1, line mode and preprocessing. `--min-size 1` exposes all
clusters; it changes only terminal display, not clustering.

This is reproducible **prefix replay**, not a claim to evaluate stable online
IDs, bounded-memory behavior, latency, or a persistent production service.
IDs need only be internally consistent within a snapshot. Timestamps establish
order; the baseline CLI uses message order and does not parse wall-clock time.

## Evaluate an incremental candidate

```bash
python -m evaluation.run \
  --adapter-command '.venv/bin/python /absolute/path/to/candidate.py' \
  --out evaluation/runs/candidate
```

For each final/prefix replay, the executable gets one JSON object on stdin:

```json
{"schema_version":1,"scenario_id":"example","messages":[
  {"id":"m01","timestamp":"2026-10-01T09:01:00Z","thread":"release",
   "reply_to":null,"text":"The build pipeline failed."}
]}
```

It writes one snapshot on stdout (diagnostics go to stderr):

```json
{"messages":[{"id":"m01","theme_id":"local-7","episode_id":"episode-1","state":"active"}],
 "current_states":{"local-7":"active"},
 "ranking":[{"theme_id":"local-7","score":0.9}]}
```

Every input ID must appear once in input order; use `theme_id:null` for
unassigned text. IDs must be strings or null. `episode_id` and `state` must
be null when unsupported. `current_states` must cover exactly the non-null
theme IDs; values can be null. Ranking contains unique observed theme IDs
and finite numeric scores (larger means more salient); it may be empty.
The adapter is executed without a shell, with a 120-second per-prefix timeout.
It must not persist or retrieve information from future replays. Fresh
processes enforce the available input boundary, not a security sandbox.

Saved outputs can also be scored:

```bash
python -m evaluation.run --predictions evaluation/runs/candidate/predictions.json \
  --out evaluation/runs/rescored
```

That path validates completeness and scoring, but cannot prove the supplier
used only available context. Its manifest marks causality `supplier_asserted`.
Pin candidate weights, prompts, decoding settings and dependencies in the
candidate's own manifest; repeat stochastic candidates and report dispersion.

The oracle in `test_evaluation.py` validates metric ceilings only. It is never
run as an architecture candidate or presented as a measured improvement.
Before tuning a real model, add independently labeled, held-out real streams
and new paraphrases/incident types. Review disagreements with a second
annotator, keep development and holdout versions separate, and compare the
same baseline/candidate on the same frozen suite. See `ARCHITECTURE.md` for
the measured failure analysis and the next experiment.
