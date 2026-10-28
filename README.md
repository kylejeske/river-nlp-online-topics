# River text theme lab

Feed UTF-8 text files into River TextClust and inspect groups of messages with similar vocabulary. No predefined topic names, API key, GPU, or pretrained model are needed. After installing Python dependencies, this script processes text entirely on your machine.

River is an open-source Python library for online machine learning: its models update one observation at a time. TextClust is its text-stream clustering algorithm. It uses term counts, TF-IDF cosine distance, small evolving clusters, and optional forgetting. This lab replays saved files to test the clustering before wiring up a live stream.

## Install and run

Use Python 3.11 or newer; this starter was tested with Python 3.12 and River 0.26.1. These commands are for macOS or Linux, from the extracted `river-theme-lab` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python find_themes.py sample-data --out results
```

On Windows PowerShell, create the environment with `py -3 -m venv .venv`, then use `.\.venv\Scripts\python.exe` in place of `python` for the install and run commands. Activation is optional.

The included example has 12 messages about access approvals, invoice payments, and build pipelines. With the default radius of 0.60, the verified result is three clusters, each containing four messages. Those theme descriptions are human descriptions; the program displays descriptive keywords rather than writing a theme name.

## Use your own files

Create an `input` directory and add `.txt` files, or pass an existing file or directory directly:

```bash
python find_themes.py input --out results
python find_themes.py /path/to/conversation.txt --out results-one-file
```

The default input unit is one nonblank line per message. Use plain text such as:

```text
Production access approval is blocking the deployment.
The build pipeline failed during the integration tests.
We need production access approval before deployment can start.
```

For paragraphs separated by blank lines:

```bash
python find_themes.py input --unit paragraph --out results-paragraphs
```

For short documents where each file should be one observation:

```bash
python find_themes.py input --unit file --out results-documents
```

For conversational replies that need nearby context, test three adjacent messages per window:

```bash
python find_themes.py input --window 3 --out results-windows
```

Windows are non-overlapping and never cross file boundaries. Keep each file to a coherent thread or conversation if you use this option. Larger windows can mix unrelated topics; inspect the examples to see whether they help. The last window may contain fewer messages.

Files are read recursively in sorted path order, with their messages in file order. Replay order can affect the clusters. Use filenames such as `001-chat.txt` and `002-chat.txt` to control replay order. The script does not parse or sort timestamps. For interleaved conversations, first prepare a combined chronological file if that is the stream order you want to simulate.

Remove repeated export headers, speaker labels, timestamps, and other boilerplate when they overwhelm the actual content. The built-in stop words are English. Use `--stopwords extra-stopwords.txt` to add one lowercase word per line, or edit `STOP_WORDS` for a different language.

## Inspect the output

The terminal lists clusters with at least two windows, their keywords, and example messages. It also writes:

| File | Contents |
| --- | --- |
| `themes.json` | All final clusters, keywords, window counts, distinct-file counts, examples, settings, and skipped input windows |
| `assignments.csv` | Every processed window with its theme ID, original text, filename, and location |

`--min-size 1` also prints singleton clusters. The threshold only filters the terminal display; all clusters remain in the output files. Cluster IDs are arbitrary and may change between runs with different input or settings. Window counts measure how often content occurs; they do not measure importance.

Examples are selected for overlap with a cluster's displayed keywords. They help inspection and are not generated summaries. Keyword scores use TF-IDF calculated across the final output groups.

## Tune the grouping

Start with the same input and compare these two runs:

```bash
python find_themes.py input --radius 0.45 --out results-narrow
python find_themes.py input --radius 0.70 --out results-broad
```

Lower radius values make merging stricter, usually producing more, narrower clusters. Higher values allow less similar messages to merge, usually producing fewer, broader clusters. The setting is a cosine-distance threshold, not a probability. If one cluster contains unrelated subjects, lower it. If near-duplicates fragment, raise it. Try one change at a time.

For an initial quality check, read 20–50 messages yourself, record the themes you expect, and compare the assignments and examples. Include related messages with different wording, unrelated messages sharing a generic word, and replies such as "same here". The sample data is a smoke test, not evidence of accuracy on real conversations.

## What this experiment does and does not establish

TextClust groups vocabulary overlap; it does not fully infer conversational meaning. "Waiting on permissions" and "production access approval is delayed" may split despite referring to one problem. It does not resolve "that issue" across distant turns or distinguish a resolved problem from an active blocker without another layer. It also does not score salience, urgency, or business impact.

This harness uses the adaptive small clusters directly; it does not request a fixed number of broad topics. It learns each window once and then assigns all windows against the final model snapshot. Historical online labels would be a different output.

For a finite replay, cleanup is scheduled beyond the corpus length, term fading is disabled, and a small positive corpus-scaled fading factor keeps old windows available. Live use would require deliberate time units and forgetting settings, ongoing evaluation, and a strategy for changing cluster IDs.

This is an exploratory file-based harness: it retains input windows and feature dictionaries to write the final report, and starts a fresh model each run. It is not a bounded-memory live-stream service. TextClust's search cost also increases with the number of small clusters, so begin with a modest dataset.

## Sources

- River project and installation: https://pypi.org/project/river/
- TextClust algorithm and API: https://riverml.xyz/latest/api/cluster/TextClust/
- BagOfWords API: https://riverml.xyz/latest/api/feature-extraction/BagOfWords/

This starter pins `river==0.26.1` for reproducible API behavior. Dependency versions are recorded in `VERIFICATION.txt`; exact transitive versions are not pinned for cross-platform installation.
