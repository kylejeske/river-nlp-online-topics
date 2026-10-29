# Evaluation: river_frequency_r0.6

Synthetic diagnostic suite; these figures are not real-world accuracy estimates.

| Measure | Result |
| --- | ---: |
| Final theme pair F1 (scenario macro mean) | 0.634 |
| Message-state coverage / accuracy | 0.0% / 0.0% |
| Current issue-state coverage / accuracy at checkpoints | 0.0% / 0.0% |
| Purity-discounted, duplicate-suppressed nDCG@3 | 0.694 |
| Critical singleton strict top-1 | 0/2 |

Null states and episodes mean unsupported, not inferred errors by a state model.

| Scenario | Messages | Theme pair F1 | Theme coverage | State accuracy |
| --- | ---: | ---: | ---: | ---: |
| original_sample | 12 | 1.000 | 100.0% | 0.0% |
| semantic_paraphrases | 8 | 0.000 | 100.0% | 0.0% |
| conversational_references | 8 | 0.000 | 75.0% | 0.0% |
| recurring_issue | 8 | 1.000 | 100.0% | 0.0% |
| singleton_critical | 9 | 1.000 | 100.0% | 0.0% |
| state_and_negation | 8 | 1.000 | 100.0% | 0.0% |
| similar_vocabulary_distinct_incidents | 6 | 0.571 | 100.0% | 0.0% |
| ambiguous_reference | 5 | 0.500 | 100.0% | 0.0% |

| Diagnostic category | Passed |
| --- | ---: |
| ambiguous_reference | 2/2 |
| emergence | 0/1 |
| lexical_collision | 1/3 |
| negation | 1/1 |
| paraphrase | 0/4 |
| recurrence | 1/5 |
| reference | 3/7 |
| reopening | 0/1 |
| singleton | 1/1 |
| state_change | 0/7 |
| state_continuity | 2/2 |

## Checkpoint rankings

Scores are adapter outputs; River's scores are counts, not salience.
Ties use the first member's input order, never a gold label. nDCG excludes all-zero relevance checkpoints.

- `original_sample/m09`: 0→access (3, purity 1.00), 1→invoice (3, purity 1.00), 2→build (3, purity 1.00); nDCG@3=0.951; tied scores=True.
- `original_sample/m12`: 0→access (4, purity 1.00), 1→invoice (4, purity 1.00), 2→build (4, purity 1.00); nDCG@3=1.000; tied scores=True.
- `semantic_paraphrases/m08`: 0→checkout (1, purity 1.00), 1→printer (1, purity 1.00), 2→checkout (1, purity 1.00); nDCG@3=1.000; tied scores=True.
- `conversational_references/m04`: 0→access (1, purity 1.00), 1→invoice (1, purity 1.00); nDCG@3=1.000; tied scores=True.
- `conversational_references/m07`: 0→access (1, purity 1.00), 1→invoice (1, purity 1.00), 2→lunch (1, purity 1.00); nDCG@3=0.631; tied scores=True.
- `conversational_references/m08`: 0→access (1, purity 1.00), 1→invoice (1, purity 1.00), 2→lunch (1, purity 1.00); nDCG@3=N/A; tied scores=True.
- `recurring_issue/m03`: 0→exports (3, purity 1.00); nDCG@3=N/A; tied scores=False.
- `recurring_issue/m06`: 0→exports (5, purity 1.00), 1→lunch (1, purity 1.00); nDCG@3=1.000; tied scores=False.
- `recurring_issue/m08`: 0→exports (6, purity 1.00), 1→lunch (2, purity 1.00); nDCG@3=N/A; tied scores=False.
- `singleton_critical/m08`: 0→lunch (5, purity 1.00), 1→docs (2, purity 1.00), 2→key_leak (1, purity 1.00); nDCG@3=0.500; tied scores=False.
- `singleton_critical/m09`: 0→lunch (6, purity 1.00), 1→docs (2, purity 1.00), 2→key_leak (1, purity 1.00); nDCG@3=0.500; tied scores=False.
- `state_and_negation/m02`: 0→checkout (2, purity 1.00); nDCG@3=N/A; tied scores=False.
- `state_and_negation/m05`: 1→build (3, purity 1.00), 0→checkout (2, purity 1.00); nDCG@3=1.000; tied scores=False.
- `state_and_negation/m07`: 1→build (4, purity 1.00), 0→checkout (3, purity 1.00); nDCG@3=0.631; tied scores=False.
- `state_and_negation/m08`: 0→checkout (4, purity 1.00), 1→build (4, purity 1.00); nDCG@3=N/A; tied scores=True.
- `similar_vocabulary_distinct_incidents/m03`: 0→staging (3, purity 0.67); nDCG@3=0.000; tied scores=False.
- `similar_vocabulary_distinct_incidents/m06`: 0→production (6, purity 0.50); nDCG@3=0.500; tied scores=False.
- `ambiguous_reference/m03`: 0→search (2, purity 0.50), 1→ambiguous (1, purity 1.00); nDCG@3=0.307; tied scores=False.

## Failed probes

- `original_sample/m02→m11` (state_transition): The payment is explicitly resolved.
- `original_sample/m03→m12` (state_transition): The pipeline is explicitly resolved.
- `semantic_paraphrases/m01→m03` (same_theme): Same customer-facing purchase failure.
- `semantic_paraphrases/m01→m05` (same_theme): Transaction processor and payment gateway refer to the same outage.
- `semantic_paraphrases/m02→m04` (same_theme): Printer consumables expressed indirectly.
- `semantic_paraphrases/m02→m06` (same_theme): Hard copies and printer describe the same device.
- `conversational_references/m01→m03` (same_theme): Explicit reply target in another thread.
- `conversational_references/m02→m04` (same_theme): Stopword-only reply still has meaning.
- `conversational_references/m01→m06` (same_theme): Resolution follows a distracting turn.
- `conversational_references/m02→m08` (same_theme): Transitive reply chain links payment resolution.
- `conversational_references/m01→m06` (state_transition): Reference resolves the access blocker.
- `conversational_references/m02→m08` (state_transition): Reference resolves the payment failure.
- `recurring_issue/m01→m02` (same_episode): Repeated observations do not imply a new incident.
- `recurring_issue/m01→m03` (same_episode): Resolution belongs to the initial incident.
- `recurring_issue/m03→m05` (different_episode): A new failure after recovery begins a second incident.
- `recurring_issue/m05→m08` (same_episode): Second incident and its recovery share an episode.
- `recurring_issue/m03→m05` (state_transition): Resolved becomes active on recurrence.
- `state_and_negation/m03→m06` (state_transition): Only verified recovery resolves the build.
- `state_and_negation/m02→m07` (state_transition): A healthy system later becomes an actual incident.
- `state_and_negation/m07→m08` (state_transition): Checkout has explicit recovery.
- `similar_vocabulary_distinct_incidents/m01→m02` (different_theme): Production and staging have independent incident identities.
- `similar_vocabulary_distinct_incidents/m01→m03` (different_theme): A staging recovery is not production recovery.
- `similar_vocabulary_distinct_incidents/m02→m03` (state_transition): Only staging is resolved.
