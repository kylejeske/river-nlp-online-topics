#!/usr/bin/env python3
"""Replay UTF-8 text files through River TextClust and inspect final clusters."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re

import river
from river import cluster, feature_extraction


# A deliberately small, editable English list. Keep domain-specific vocabulary.
STOP_WORDS = set("""
a an and are as at be been being but by can could did do does doing for from
had has have having he her hers him his how i if in into is it its just me more
most my no not of on or our ours out she should so some than that the their
theirs them then there these they this those through to too up us very was we
were what when where which who why will with would you your yours
also am any here im ive its lets now please really same still thanks thats
theres well were youre
""".split())


def records_from_files(source, unit, window):
    paths = [source] if source.is_file() else sorted(source.rglob("*.txt"))
    if not paths:
        raise ValueError(f"No .txt files found in {source}")
    records = []
    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        if unit == "line":
            parts = [(f"line {i}", t.strip()) for i, t in enumerate(text.splitlines(), 1) if t.strip()]
        elif unit == "paragraph":
            parts = [(f"paragraph {i}", t.strip()) for i, t in enumerate(re.split(r"\n\s*\n", text), 1) if t.strip()]
        else:
            parts = [("file", text.strip())] if text.strip() else []
        # Non-overlapping windows never cross a file boundary.
        for start in range(0, len(parts), window):
            group = parts[start:start + window]
            records.append({
                "source": path.name if source.is_file() else path.relative_to(source).as_posix(),
                "location": group[0][0] if len(group) == 1 else f"{group[0][0]} to {group[-1][0]}",
                "text": "\n".join(t for _, t in group),
            })
    return paths, records


def term_text(term):
    return " ".join(term) if isinstance(term, tuple) else str(term)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A UTF-8 .txt file or directory (recursive)")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--unit", choices=["line", "paragraph", "file"], default="line")
    parser.add_argument("--window", type=int, default=1, help="Adjacent units per window within each file")
    parser.add_argument("--radius", type=float, default=0.60, help="Higher merges more readily; range (0, 1]")
    parser.add_argument("--min-size", type=int, default=2, help="Minimum final windows to list as a theme")
    parser.add_argument("--top-terms", type=int, default=8)
    parser.add_argument("--stopwords", type=Path, help="Optional extra stop words, one per line")
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    if not 0 < args.radius <= 1:
        parser.error("--radius must be in (0, 1]")
    if min(args.window, args.min_size, args.top_terms) < 1:
        parser.error("--window, --min-size and --top-terms must be positive")
    if args.unit == "file" and args.window != 1:
        parser.error("--window must be 1 when --unit=file")

    stop_words = STOP_WORDS.copy()
    try:
        if args.stopwords:
            stop_words.update(w.strip().lower() for w in args.stopwords.read_text(encoding="utf-8-sig").splitlines() if w.strip())
        files, records = records_from_files(args.input, args.unit, args.window)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))

    bow = feature_extraction.BagOfWords(stop_words=stop_words, ngram_range=(1, 2))
    usable, features, skipped = [], [], []
    for record in records:
        counts = bow.transform_one(record["text"])
        if counts:
            usable.append(record)
            features.append(counts)
        else:
            skipped.append({**record, "reason": "No tokens after preprocessing"})
    if not usable:
        parser.error("No usable text remains after preprocessing")

    n = len(usable)
    model = cluster.TextClust(
        radius=args.radius,
        auto_r=False,
        real_time_fading=False,
        term_fading=False,
        # A positive fading factor avoids the zero-factor singleton-removal edge.
        # This corpus-scaled setting keeps weight decay small, while tgap exceeds
        # the replay length so old windows survive this finite experiment.
        fading_factor=0.001 / n,
        tgap=n + 1,
    )
    for counts in features:
        model.learn_one(counts)

    # Assign against the FINAL snapshot. These are not historical online labels.
    groups = defaultdict(list)
    for record, counts in zip(usable, features):
        theme_id = model.predict_one(counts)
        if theme_id is None:
            raise RuntimeError("TextClust produced no assignment for nonempty text")
        groups[theme_id].append((record, counts))

    # Explain clusters with TF-IDF terms calculated across final output groups.
    term_counts = {key: sum((Counter(x) for _, x in rows), Counter()) for key, rows in groups.items()}
    document_frequency = Counter(term for counts in term_counts.values() for term in counts)
    themes = []
    assignments = []
    for theme_id, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        counts = term_counts[theme_id]
        scores = {term: (1 + math.log(tf)) * (1 + math.log((1 + len(groups)) / (1 + document_frequency[term]))) for term, tf in counts.items()}
        top = sorted(scores, key=lambda term: (-scores[term], term_text(term)))[:args.top_terms]
        keywords = [term_text(term) for term in top]
        # Prefer examples sharing the most of the displayed cluster keywords.
        examples = sorted(rows, key=lambda row: -sum(scores[t] for t in top if t in row[1]))[:3]
        themes.append({
            "theme_id": theme_id,
            "windows": len(rows),
            "files": len({r["source"] for r, _ in rows}),
            "keywords": keywords,
            "examples": [r for r, _ in examples],
            "listed": len(rows) >= args.min_size,
        })
        assignments.extend({"theme_id": theme_id, **record} for record, _ in rows)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "river_version": river.__version__,
        "settings": {"unit": args.unit, "window": args.window, "radius": args.radius, "min_size": args.min_size},
        "input_files": len(files), "processed_windows": n, "skipped_windows": skipped,
        "cluster_count": len(themes), "themes": themes,
        "interpretation": "Vocabulary-based final clusters, ranked by window count. Keywords are descriptive terms, not generated names or salience scores.",
    }
    (args.out / "themes.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out / "assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["theme_id", "source", "location", "text"])
        writer.writeheader()
        writer.writerows(assignments)

    print(f"River {river.__version__}: {n} windows, {len(files)} files, {len(themes)} clusters; {len(skipped)} skipped.")
    for theme in themes:
        if theme["listed"]:
            print(f"\nTheme {theme['theme_id']} ({theme['windows']} windows): {', '.join(theme['keywords'])}")
            for example in theme["examples"][:2]:
                excerpt = " ".join(example["text"].split())[:180]
                print(f"  [{example['source']}, {example['location']}] {excerpt}")
    hidden = sum(not t["listed"] for t in themes)
    if hidden:
        print(f"\n{hidden} smaller clusters are included in themes.json; use --min-size 1 to print them.")
    print(f"\nOutput: {args.out.resolve()}")


if __name__ == "__main__":
    main()
