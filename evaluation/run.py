"""Run unchanged River CLI, a gold-blind adapter, or saved predictions.

    python -m evaluation.run --out evaluation/runs/baseline
"""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

from evaluation.metrics import evaluate, validate_suite

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "evaluation/fixtures/contextual-v1.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def public_input(scenario, messages):
    # An allowlist prevents future gold fields from leaking into a new adapter.
    return {"schema_version": 1, "scenario_id": scenario["id"],
            "messages": [{key: m[key] for key in ("id", "timestamp", "thread", "reply_to", "text")}
                         for m in messages]}


def verify_baseline():
    lock = json.loads((ROOT / "evaluation/baseline-lock.json").read_text())
    for path, expected in lock["sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Frozen baseline changed: {path}. Use a separate candidate adapter.")
    if version("river") != "0.26.1":
        raise ValueError("Frozen baseline requires river==0.26.1")
    return lock


def baseline_snapshot(messages, raw_dir, radius):
    raw_dir.mkdir(parents=True, exist_ok=True)
    source = raw_dir / "messages.txt"
    source.write_text("\n".join(m["text"] for m in messages) + "\n", encoding="utf-8")
    command = [sys.executable, str(ROOT / "find_themes.py"), str(source),
               "--out", str(raw_dir / "output"), "--radius", str(radius), "--min-size", "1"]
    result = subprocess.run(command, check=True, capture_output=True, text=True,
                            env={**os.environ, "PYTHONHASHSEED": "0"})
    (raw_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    if result.stderr:
        (raw_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    report = json.loads((raw_dir / "output/themes.json").read_text())
    with (raw_dir / "output/assignments.csv").open(newline="", encoding="utf-8") as handle:
        by_line = {int(row["location"].split()[1]): row for row in csv.DictReader(handle)}
    rows = [{"id": m["id"], "theme_id": by_line[i]["theme_id"] if i in by_line else None,
             "episode_id": None, "state": None} for i, m in enumerate(messages, 1)]
    return {"messages": rows,
            "current_states": {str(t["theme_id"]): None for t in report["themes"]},
            "ranking": [{"theme_id": str(t["theme_id"]), "score": t["windows"]} for t in report["themes"]],
            "skipped_message_ids": [r["id"] for r in rows if r["theme_id"] is None]}


def command_snapshot(command, scenario, messages):
    result = subprocess.run(command, input=json.dumps(public_input(scenario, messages)),
                            capture_output=True, text=True, check=True, timeout=120)
    return json.loads(result.stdout)


def generate(suite, out, radius, adapter_command):
    predictions = {"schema_version": 1, "adapter": "external_command" if adapter_command else f"river_frequency_r{radius:g}",
                   "capabilities": {"theme": True, "episode": False, "state": False, "salience": "frequency_proxy"}
                   if not adapter_command else {"declared_by": "adapter"}, "scenarios": {}}
    command = shlex.split(adapter_command) if adapter_command else None
    for scenario in suite["scenarios"]:
        def snapshot(messages, name):
            if command:
                return command_snapshot(command, scenario, messages)
            return baseline_snapshot(messages, out / "raw" / scenario["id"] / name, radius)
        checkpoints = {}
        # Each checkpoint runs in a fresh process, with only its available prefix.
        for checkpoint in scenario["checkpoints"]:
            end = [m["id"] for m in scenario["messages"]].index(checkpoint["after"]) + 1
            checkpoints[checkpoint["after"]] = snapshot(scenario["messages"][:end], checkpoint["after"])
        predictions["scenarios"][scenario["id"]] = {
            "final": snapshot(scenario["messages"], "final"), "checkpoints": checkpoints}
    return predictions


def markdown_report(report):
    a = report["aggregate"]
    def number(value):
        return f"{value:.3f}" if value is not None else "N/A"
    lines = [f"# Evaluation: {report['adapter']}", "",
             "Synthetic diagnostic suite; these figures are not real-world accuracy estimates.", "",
             "| Measure | Result |", "| --- | ---: |",
             f"| Final theme pair F1 (scenario macro mean) | {number(a['theme_pair_f1_macro'])} |",
             f"| Message-state coverage / accuracy | {a['final_state_coverage']:.1%} / {a['final_state_accuracy']:.1%} |",
             f"| Current issue-state coverage / accuracy at checkpoints | {a['current_issue_state_coverage']:.1%} / {a['current_issue_state_accuracy']:.1%} |",
             f"| Purity-discounted, duplicate-suppressed nDCG@3 | {a['ndcg_at_3_macro']:.3f} |" if a['ndcg_at_3_macro'] is not None else "| nDCG@3 | N/A (no positive relevance) |",
             f"| Critical singleton strict top-1 | {a['singleton_hits_at_1']}/{a['important_singletons']} |", "",
             "Null states and episodes mean unsupported, not inferred errors by a state model.", "",
             "| Scenario | Messages | Theme pair F1 | Theme coverage | State accuracy |", "| --- | ---: | ---: | ---: | ---: |"]
    for s in report["scenarios"]:
        lines.append(f"| {s['id']} | {s['messages']} | {number(s['theme_pairs']['f1'])} | {s['theme_coverage']:.1%} | {s['states']['accuracy']:.1%} |")
    lines += ["", "| Diagnostic category | Passed |", "| --- | ---: |"]
    for name, counts in a["probes"].items():
        lines.append(f"| {name} | {counts['passed']}/{counts['total']} |")
    lines += ["", "## Checkpoint rankings", "", "Scores are adapter outputs; River's scores are counts, not salience.",
              "Ties use the first member's input order, never a gold label. nDCG excludes all-zero relevance checkpoints.", ""]
    for s in report["scenarios"]:
        for c in s["checkpoints"]:
            ranking = c["ranking"]
            top = ", ".join(f"{r['theme_id']}→{r['dominant_gold_theme']} ({r['score']}, purity {r['purity']:.2f})" for r in ranking["ranking"][:3]) or "(empty)"
            value = ranking["ndcg_at_3"]
            score = f"{value:.3f}" if value is not None else "N/A"
            lines.append(f"- `{s['id']}/{c['after']}`: {top}; nDCG@3={score}; tied scores={ranking['tied_scores']}.")
    lines += ["", "## Failed probes", ""]
    for s in report["scenarios"]:
        for probe in s["probes"]:
            if not probe["passed"]:
                lines.append(f"- `{s['id']}/{probe['a']}→{probe['b']}` ({probe['relation']}): {probe['why']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation/runs/baseline")
    parser.add_argument("--radius", type=float, default=0.60)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adapter-command", help="Executable and arguments; JSON on stdin/stdout, no shell")
    group.add_argument("--predictions", type=Path, help="Score an existing prediction file; causality is supplier-asserted")
    args = parser.parse_args()
    if not 0 < args.radius <= 1:
        parser.error("--radius must be in (0, 1]")
    suite = json.loads(args.suite.read_text())
    validate_suite(suite)
    baseline_lock = verify_baseline() if not args.adapter_command and not args.predictions else None
    args.out = args.out.resolve()
    predictions = json.loads(args.predictions.read_text()) if args.predictions else generate(suite, args.out, args.radius, args.adapter_command)
    report = evaluate(suite, predictions)
    manifest = {"suite_id": suite["suite_id"], "suite_sha256": digest(args.suite),
                "evaluator_sha256": {f"evaluation/{name}": digest(ROOT / "evaluation" / name) for name in ("run.py", "metrics.py")},
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: version(name) for name in ("river", "numpy", "scipy", "narwhals")},
                "baseline_lock": baseline_lock, "radius": args.radius if baseline_lock else None,
                "adapter_command": args.adapter_command,
                "causality": "supplier_asserted" if args.predictions else "fresh_process_per_prefix_no_future_messages",
                "ranking_policy": "descending score; ties by first assigned input message",
                "state_support": "null = unsupported, unknown = explicit uncertainty"}
    write_json(args.out / "predictions.json", predictions)
    write_json(args.out / "metrics.json", report)
    write_json(args.out / "manifest.json", manifest)
    (args.out / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Report: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
