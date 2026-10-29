"""Evaluator contract, adversarial metrics, causality, and frozen CLI regression."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from evaluation.metrics import evaluate, pairwise, ranking_metrics, validate_snapshot, validate_suite
from evaluation.run import DEFAULT_SUITE, ROOT, baseline_snapshot, command_snapshot, markdown_report, public_input, verify_baseline


def oracle_snapshot(messages, relevance):
    rows = [{"id": m["id"], "theme_id": m["gold"]["theme"],
             "episode_id": m["gold"]["episode"], "state": m["gold"]["state"]} for m in messages]
    states = {m["gold"]["theme"]: m["gold"]["state"] for m in messages}
    return {"messages": rows, "current_states": states,
            "ranking": [{"theme_id": theme, "score": score} for theme, score in relevance.items()]}


def oracle_predictions(suite):
    outputs = {}
    for s in suite["scenarios"]:
        checkpoints = {}
        for c in s["checkpoints"]:
            prefix = s["messages"][:[m["id"] for m in s["messages"]].index(c["after"]) + 1]
            checkpoints[c["after"]] = oracle_snapshot(prefix, c["relevance"])
        outputs[s["id"]] = {"final": oracle_snapshot(s["messages"], s["checkpoints"][-1]["relevance"]),
                            "checkpoints": checkpoints}
    return {"schema_version": 1, "adapter": "oracle_metric_test_only", "scenarios": outputs}


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = json.loads(DEFAULT_SUITE.read_text())

    def test_gold_is_consistent_and_covers_required_cases(self):
        validate_suite(self.suite)
        categories = {p["category"] for s in self.suite["scenarios"] for p in s["probes"]}
        self.assertTrue({"paraphrase", "reference", "recurrence", "state_change", "singleton", "lexical_collision"} <= categories)
        self.assertEqual(sum(len(s["messages"]) for s in self.suite["scenarios"]), 64)

    def test_all_routine_singleton_has_no_undefined_metric_crash(self):
        suite = deepcopy(self.suite)
        suite["scenarios"] = [suite["scenarios"][0]]
        s = suite["scenarios"][0]
        s["messages"] = [s["messages"][0]]
        s["messages"][0]["gold"] = {"theme": "routine", "episode": None, "state": "none"}
        s["probes"] = []
        s["checkpoints"] = [{"after": "m01", "relevance": {"routine": 0}, "important_singletons": []}]
        report = evaluate(suite, oracle_predictions(suite))
        self.assertIsNone(report["aggregate"]["ndcg_at_3_macro"])
        self.assertIsNone(report["aggregate"]["theme_pair_f1_macro"])
        self.assertIn("N/A", markdown_report(report))

    def test_oracle_reaches_ceiling_on_all_capabilities(self):
        report = evaluate(self.suite, oracle_predictions(self.suite))
        aggregate = report["aggregate"]
        for metric in ("theme_pair_f1_macro", "episode_pair_f1_macro", "final_state_accuracy",
                       "current_issue_state_accuracy", "ndcg_at_3_macro"):
            self.assertAlmostEqual(aggregate[metric], 1.0, msg=metric)
        self.assertEqual(aggregate["singleton_hits_at_1"], aggregate["important_singletons"])
        self.assertTrue(all(p["passed"] == p["total"] for p in aggregate["probes"].values()))

    def test_cluster_labels_are_arbitrary(self):
        predictions = oracle_predictions(self.suite)
        before = evaluate(self.suite, predictions)
        for scenario in predictions["scenarios"].values():
            for snapshot in [scenario["final"], *scenario["checkpoints"].values()]:
                for row in snapshot["messages"]:
                    row["theme_id"] = "renamed:" + row["theme_id"]
                    if row["episode_id"] is not None:
                        row["episode_id"] = "episode:" + row["episode_id"]
                snapshot["current_states"] = {"renamed:" + k: v for k, v in snapshot["current_states"].items()}
                for item in snapshot["ranking"]:
                    item["theme_id"] = "renamed:" + item["theme_id"]
        self.assertEqual(before["aggregate"], evaluate(self.suite, predictions)["aggregate"])

    def test_overmerge_and_fragmentation_are_penalized(self):
        gold = ["a", "a", "b", "b"]
        self.assertEqual(pairwise(gold, ["a", "a", "b", "b"])["f1"], 1)
        self.assertLess(pairwise(gold, ["one"] * 4)["precision"], 1)
        self.assertEqual(pairwise(gold, ["1", "2", "3", "4"])["recall"], 0)
        self.assertEqual(pairwise(gold, [None] * 4)["f1"], 0)
        self.assertIsNone(pairwise(["single"], ["single"])["f1"])

    def test_unsupported_outputs_are_coverage_failures(self):
        predictions = oracle_predictions(self.suite)
        for output in predictions["scenarios"].values():
            for snapshot in [output["final"], *output["checkpoints"].values()]:
                for row in snapshot["messages"]:
                    row.update(state=None, episode_id=None)
                snapshot["current_states"] = dict.fromkeys(snapshot["current_states"], None)
        report = evaluate(self.suite, predictions)
        self.assertEqual(report["aggregate"]["final_state_coverage"], 0)
        self.assertEqual(report["aggregate"]["current_issue_state_accuracy"], 0)
        recurrence = report["aggregate"]["probes"]["recurrence"]
        self.assertLess(recurrence["passed"], recurrence["total"])

    def test_rare_critical_event_beats_frequency_only_with_salience(self):
        s = next(s for s in self.suite["scenarios"] if s["id"] == "singleton_critical")
        c = s["checkpoints"][-1]
        oracle = oracle_snapshot(s["messages"], c["relevance"])
        self.assertEqual(ranking_metrics(c, s["messages"], oracle)["singleton_hits_at_1"], 1)
        counts = Counter(m["gold"]["theme"] for m in s["messages"])
        for item in oracle["ranking"]:
            item["score"] = counts[item["theme_id"]]
        ranked = ranking_metrics(c, s["messages"], oracle)
        self.assertEqual(ranked["singleton_hits_at_1"], 0)
        self.assertLess(ranked["ndcg_at_3"], 1)
        for item in oracle["ranking"]:
            item["score"] = 1
        self.assertEqual(ranking_metrics(c, s["messages"], oracle)["singleton_hits_at_1"], 0)

    def test_ranking_does_not_reward_impure_or_duplicate_clusters(self):
        messages = [{"id": str(i), "gold": {"theme": t}} for i, t in enumerate(["critical", "critical", "routine"])]
        checkpoint = {"relevance": {"critical": 3, "routine": 0}, "important_singletons": []}
        split = {"messages": [{"theme_id": str(i)} for i in range(3)],
                 "ranking": [{"theme_id": str(i), "score": 3 - i} for i in range(3)]}
        result = ranking_metrics(checkpoint, messages, split)
        self.assertEqual(result["ranking"][1]["credited_gain"], 0)
        merged = {"messages": [{"theme_id": "all"}] * 3, "ranking": [{"theme_id": "all", "score": 3}]}
        self.assertLess(ranking_metrics(checkpoint, messages, merged)["ndcg_at_3"], 1)

    def test_resolved_state_must_update_without_overwriting_history(self):
        s = self.suite["scenarios"][0]
        predictions = oracle_predictions(self.suite)
        snapshot = predictions["scenarios"][s["id"]]["checkpoints"]["m12"]
        # The historical invoice failure remains active at its observation,
        # while the latest issue state must reflect the later success.
        self.assertEqual(snapshot["messages"][1]["state"], "active")
        self.assertEqual(snapshot["current_states"]["invoice"], "resolved")
        snapshot["current_states"]["invoice"] = "active"
        result = evaluate(self.suite, predictions)
        self.assertLess(result["aggregate"]["current_issue_state_accuracy"], 1)

    def test_invalid_or_incomplete_predictions_are_rejected(self):
        messages = self.suite["scenarios"][0]["messages"]
        snapshot = oracle_snapshot(messages, {"access": 2, "invoice": 0, "build": 0})
        bad = deepcopy(snapshot)
        bad["messages"].pop()
        with self.assertRaises(ValueError):
            validate_snapshot(bad, messages)
        bad = deepcopy(snapshot)
        bad["ranking"][0]["score"] = float("nan")
        with self.assertRaises(ValueError):
            validate_snapshot(bad, messages)
        bad = deepcopy(snapshot)
        bad["ranking"].append(bad["ranking"][0])
        with self.assertRaises(ValueError):
            validate_snapshot(bad, messages)

    def test_adapter_receives_only_available_text_and_metadata(self):
        s = self.suite["scenarios"][0]
        prefix = s["messages"][:3]
        payload = public_input(s, prefix)
        self.assertNotIn("gold", json.dumps(payload))
        self.assertNotIn("relevance", json.dumps(payload))
        self.assertEqual(len(payload["messages"]), 3)
        command = [sys.executable, "-c", "import json,sys; x=json.load(sys.stdin); print(json.dumps({'ids':[m['id'] for m in x['messages']]}))"]
        self.assertEqual(command_snapshot(command, s, prefix), {"ids": ["m01", "m02", "m03"]})

    def test_original_cli_partition_matches_committed_sample(self):
        verify_baseline()
        s = self.suite["scenarios"][0]
        self.assertEqual([m["text"] for m in s["messages"]], (ROOT / "sample-data/001-chat.txt").read_text().splitlines())
        with tempfile.TemporaryDirectory() as directory:
            snapshot = baseline_snapshot(s["messages"], Path(directory), 0.60)
        self.assertEqual(sorted(Counter(r["theme_id"] for r in snapshot["messages"]).values()), [4, 4, 4])
        report = evaluate(self.suite, oracle_predictions(self.suite))
        self.assertEqual(report["scenarios"][0]["theme_pairs"]["f1"], 1)
        self.assertEqual(pairwise([m["gold"]["theme"] for m in s["messages"]],
                                  [r["theme_id"] for r in snapshot["messages"]])["f1"], 1)

    def test_stopword_only_replies_are_retained_as_unassigned(self):
        s = next(s for s in self.suite["scenarios"] if s["id"] == "conversational_references")
        with tempfile.TemporaryDirectory() as directory:
            snapshot = baseline_snapshot(s["messages"][:4], Path(directory), 0.60)
        self.assertEqual(snapshot["skipped_message_ids"], ["m03", "m04"])
        self.assertEqual(len(snapshot["messages"]), 4)


if __name__ == "__main__":
    unittest.main()
