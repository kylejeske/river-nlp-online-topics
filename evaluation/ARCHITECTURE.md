# Next experiment: a contextual issue-event sidecar

Decision: retain River's lexical clusters and add a separate, context-aware
issue ledger through the candidate adapter interface. This is a proposed
experiment; no improvement from this design has been measured yet.

## Evidence from this suite

At radius 0.60, the unchanged baseline scores 1.000 theme-pair F1 on the
original sample and on the recurring-issue scenario. It nevertheless has
zero state/episode output coverage. The original invoice and build
resolutions are correctly grouped with their earlier failures; their shared
theme is not the error. The missing representation is what happened to the
issue, and whether action is still needed.

Other measured failures:

- 0/4 semantic-paraphrase links pass; the paraphrase scenario has F1 0.000.
- 3/7 reference-link probes pass; two reference messages are completely
  removed by the baseline stopword filter.
- 1/3 vocabulary-collision probes pass. The six staging/production messages
  merge into one cluster even though the environments have independent state.
- The recurring issue retains its subject, but episode identity and the
  resolved-to-active transition are unsupported.
- The valid leaked signing key ranks third behind lunch and documentation,
  both immediately and after the next routine message: strict top-1 0/2.
- The overall adapted nDCG@3 is 0.694 across 13 positive-relevance
  checkpoints. Favorable arrival-order ties produce some perfect rankings
  despite failures to understand paraphrases; this number alone would hide
  the important defects.

Radius controls give the following results on this same development suite:

| Radius | Theme-pair F1, macro | Adapted nDCG@3 | Paraphrase probes | Critical singleton top-1 | State coverage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.45 | 0.592 | 0.694 | 0/4 | 0/2 | 0% |
| 0.60 | 0.634 | 0.694 | 0/4 | 0/2 | 0% |
| 0.70 | 0.665 | 0.666 | 0/4 | 0/2 | 0% |

These are observed diagnostics, not statistically independent trials or an
exhaustive threshold search. Radius 0.70 improves grouping slightly while
making ranking worse here. A distance threshold cannot supply a missing
issue-state or importance representation.

## Smallest useful architectural change

Add one adapter that emits **evidence-backed issue events**, and maintains a
small ledger independently of River's mutable cluster IDs. Leave River as
a candidate-subject generator and a comparison baseline.

An extracted event should contain:

```json
{
  "message_id": "m12",
  "referenced_message_ids": ["m03"],
  "subject": {"service": "build pipeline", "environment": "production"},
  "assertion": "recovered",
  "impact": "release_blocked",
  "evidence": "the build pipeline passed",
  "confidence": 0.95
}
```

This is a proposed schema, not a prediction produced by the current code.
Confidence above is illustrative, not a calibrated probability.

The adapter should do three closely connected operations:

1. Resolve the subject using explicit reply targets first, then bounded
   thread history and semantic matching. Preserve distinct entities and
   environments. For an ambiguous “that's fixed,” emit uncertainty and
   leave existing issues unchanged. A mention's evidence IDs must be actual
   available messages; do not silently drop content because it lacks tokens.
2. Apply typed events to a ledger with stable issue ID, episode ID, current
   state, last evidence and impact. `failure → active`; verified `recovered
   → resolved`; a new failure after recovery opens another episode.
   Hypothetical failures and negated/unverified recovery must not cause
   those transitions. A repeated active report updates evidence rather than
   incrementing the episode count.
3. Rank issues using asserted impact and current actionability, with age and
   recurrence as secondary features. A single active production exposure
   can outrank many routine mentions. Keep historical resolutions available
   without continuing to rank them as active blockers. Do not hardcode the
   fixture nouns, gold grades, or expected number of themes.

The ledger itself can be deterministic. Its hard part is extracting and
linking events from unrestricted language; the ledger does not solve that
automatically. Use the adapter boundary to compare a lightweight NLP
extractor and a constrained language-model extractor, each with pinned
model/prompt/configuration and measured cost. Explicit reply metadata is a
cheap first control; it will not solve unreferenced semantic paraphrases.
Do not pick a model or promise a gain before running it.

## Why this before other changes

| Option | What it may improve | What it leaves unresolved |
| --- | --- | --- |
| Tune radius | Lexical fragmentation/merging | Subject references, incident state and salience |
| Concatenate nearby messages | Some short replies | Interleaved threads, distant references, state and priority |
| Replace BoW with semantic vectors | Paraphrases and candidate retrieval | Recovery versus failure, episode boundaries and one-off importance |
| Add contextual event ledger | Explicit identity, state, episodes and priority; supports semantic retrieval later | Still depends on accurate extraction and uncertainty handling |

Embeddings are a useful candidate-retrieval improvement, but they are not a
complete answer to the failures measured here. Similarity should often keep
“failed” and “recovered” close because they concern one issue; state must be
represented alongside that similarity. Do not feed dense signed embeddings
into TextClust's term-count interface as though they were BagOfWords.

## How to test the recommendation

Keep the suite and baseline fixed. Compare these ablations through the same
gold-blind prefix adapter interface:

1. River baseline (already recorded).
2. River plus explicit reply linkage and a ledger, with an unchanged lexical
   subject matcher: isolates whether state representation helps before
   adding semantic retrieval.
3. The same sidecar plus a pinned semantic subject/event extractor: tests
   the paraphrase/reference gains and the new risk of false subject merges.

Before claiming success, require improved paraphrase/reference probe counts,
nonzero episode/state coverage with accurate transitions, production/staging
separation, and critical-singleton top-1 at both checkpoints. Preserve the
original sample's theme grouping. Inspect all new false merges and false
resolutions. Report individual scenario changes and variability, not just
one improved macro score. The present recommendation is an engineering
inference from failure structure; the ablations are the evidence still
needed to establish an improvement.

Next, evaluate new subjects and independently labeled real streams that the
extractor has not been tuned against. Add event latency, false-alert rate,
positive milestones, calibrated abstention, time decay and persistent online
IDs before treating this as a production stream-monitoring system.
