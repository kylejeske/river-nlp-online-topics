"""Label-invariant metrics. Gold labels are used only here, never by adapters."""

from collections import Counter, defaultdict
from itertools import combinations
import math


STATES = {"active", "resolved", "none", "unknown"}


def validate_suite(suite):
    if suite.get("schema_version") != 1 or not suite.get("scenarios"):
        raise ValueError("Expected nonempty suite schema_version=1")
    scenario_ids = set()
    for scenario in suite["scenarios"]:
        sid = scenario["id"]
        if not scenario["messages"] or not scenario["checkpoints"]:
            raise ValueError(f"Scenario requires messages and checkpoints: {sid}")
        if sid in scenario_ids:
            raise ValueError(f"Duplicate scenario: {sid}")
        scenario_ids.add(sid)
        seen, themes, episodes = set(), set(), {}
        previous_time = ""
        for message in scenario["messages"]:
            mid, gold = message["id"], message["gold"]
            if mid in seen or not message["text"].strip() or "\n" in message["text"]:
                raise ValueError(f"Invalid message: {sid}/{mid}")
            if message["reply_to"] is not None and message["reply_to"] not in seen:
                raise ValueError(f"Reply must target an earlier message: {sid}/{mid}")
            if message["timestamp"] < previous_time:
                raise ValueError(f"Nonchronological input: {sid}/{mid}")
            previous_time = message["timestamp"]
            if (gold["state"] not in STATES or not isinstance(gold["theme"], str) or
                    (gold["episode"] is not None and not isinstance(gold["episode"], str))):
                raise ValueError(f"Invalid gold labels: {sid}/{mid}")
            if gold["episode"] is not None:
                prior = episodes.setdefault(gold["episode"], gold["theme"])
                if prior != gold["theme"]:
                    raise ValueError("Episodes cannot span themes")
            seen.add(mid)
            themes.add(gold["theme"])
        by_id = {m["id"]: m for m in scenario["messages"]}
        checkpoint_ids = set()
        for checkpoint in scenario["checkpoints"]:
            after = checkpoint["after"]
            if after not in seen or after in checkpoint_ids:
                raise ValueError(f"Invalid checkpoint: {sid}/{after}")
            checkpoint_ids.add(after)
            prefix = scenario["messages"][:list(by_id).index(after) + 1]
            counts = Counter(m["gold"]["theme"] for m in prefix)
            if set(checkpoint["relevance"]) != set(counts):
                raise ValueError("Checkpoint relevance must cover exactly observed themes")
            if any(type(v) is not int or v not in range(4) for v in checkpoint["relevance"].values()):
                raise ValueError("Relevance must be an integer from 0 to 3")
            for theme in checkpoint["important_singletons"]:
                if counts[theme] != 1 or checkpoint["relevance"][theme] != 3:
                    raise ValueError("Critical singleton must occur once and have grade 3")
        for probe in scenario["probes"]:
            if probe["a"] not in seen or probe["b"] not in seen:
                raise ValueError("Probe references missing message")
            a, b = (by_id[probe[k]]["gold"] for k in ("a", "b"))
            relation = probe["relation"]
            if relation in {"same_theme", "different_theme", "same_episode", "different_episode"}:
                field = "theme" if relation.endswith("theme") else "episode"
                same = a[field] == b[field] and a[field] is not None and b[field] is not None
                if a[field] is None or b[field] is None or same != relation.startswith("same"):
                    raise ValueError("Probe contradicts gold grouping")
            elif relation != "state_transition" or a["state"] == b["state"]:
                raise ValueError("Invalid state-transition probe")


def validate_snapshot(snapshot, messages):
    expected = [m["id"] for m in messages]
    rows = snapshot["messages"]
    if [row["id"] for row in rows] != expected:
        raise ValueError("Predictions must cover each input ID once, in input order, including skipped text")
    themes = set()
    for row in rows:
        for key in ("theme_id", "episode_id"):
            if row[key] is not None and not isinstance(row[key], str):
                raise ValueError(f"{key} must be a string or null")
        if row["state"] is not None and row["state"] not in STATES:
            raise ValueError("Invalid predicted state")
        if row["theme_id"] is not None:
            themes.add(row["theme_id"])
    ranked = set()
    for item in snapshot["ranking"]:
        theme, score = item["theme_id"], item["score"]
        if theme not in themes or theme in ranked or type(score) not in (float, int) or not math.isfinite(score):
            raise ValueError("Ranking requires unique, observed theme IDs and finite numeric scores")
        ranked.add(theme)
    if set(snapshot["current_states"]) != themes:
        raise ValueError("current_states must cover every assigned theme ID")
    if any(state is not None and state not in STATES for state in snapshot["current_states"].values()):
        raise ValueError("Invalid current theme state")


def pairwise(gold, predicted):
    tp = fp = fn = 0
    for i, j in combinations(range(len(gold)), 2):
        actual = gold[i] == gold[j]
        proposed = predicted[i] is not None and predicted[i] == predicted[j]
        tp += actual and proposed
        fp += not actual and proposed
        fn += actual and not proposed
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "tp": tp, "fp": fp, "fn": fn}


def grouping(messages, rows, field, prediction_field):
    pairs = [(m["gold"][field], r[prediction_field]) for m, r in zip(messages, rows)
             if m["gold"][field] is not None]
    return pairwise([a for a, _ in pairs], [b for _, b in pairs])


def state_metrics(messages, rows):
    expected = [m["gold"]["state"] for m in messages]
    actual = [r["state"] for r in rows]
    covered = sum(s is not None for s in actual)
    correct = sum(a == b for a, b in zip(expected, actual))
    per_class = {}
    for state in sorted(STATES):
        support = expected.count(state)
        if support:
            per_class[state] = {"correct": sum(a == state and b == state for a, b in zip(expected, actual)),
                                "total": support}
    return {"coverage": covered / len(rows), "accuracy": correct / len(rows),
            "accuracy_when_supported": correct / covered if covered else None,
            "correct": correct, "total": len(rows), "by_class": per_class}


def probe_results(scenario, snapshot):
    predicted = {r["id"]: r for r in snapshot["messages"]}
    gold = {m["id"]: m["gold"] for m in scenario["messages"]}
    results = []
    for probe in scenario["probes"]:
        a, b = (predicted[probe[k]] for k in ("a", "b"))
        relation = probe["relation"]
        if relation == "state_transition":
            passed = (a["state"] == gold[probe["a"]]["state"] and
                      b["state"] == gold[probe["b"]]["state"] and
                      a["theme_id"] is not None and a["theme_id"] == b["theme_id"])
        else:
            field = "theme_id" if relation.endswith("theme") else "episode_id"
            supported = a[field] is not None and b[field] is not None
            same = a[field] == b[field]
            passed = supported and (same if relation.startswith("same") else not same)
        results.append({**probe, "passed": bool(passed), "predicted_a": a, "predicted_b": b})
    return results


def current_state_metrics(messages, snapshot):
    latest = {}
    for message, row in zip(messages, snapshot["messages"]):
        latest[message["gold"]["theme"]] = (message["gold"]["state"], row["theme_id"])
    covered = correct = 0
    for expected, predicted_theme in latest.values():
        actual = snapshot["current_states"].get(predicted_theme)
        covered += actual is not None
        correct += actual == expected
    return {"correct": correct, "total": len(latest), "covered": covered,
            "accuracy": correct / len(latest), "coverage": covered / len(latest)}


def ranking_metrics(checkpoint, messages, snapshot, k=3):
    """Purity-discounted, duplicate-suppressed cluster DCG vs ideal gold themes.

    Dominant gold theme is chosen by member count, then first member. This is
    evaluator-only alignment, not gold-fed retrieval. Merges lose purity credit;
    split clusters cannot earn the same theme's gain repeatedly.
    """
    members = defaultdict(list)
    first = {}
    for index, (m, row) in enumerate(zip(messages, snapshot["messages"])):
        if row["theme_id"] is not None:
            members[row["theme_id"]].append(m["gold"]["theme"])
            first.setdefault(row["theme_id"], index)
    ranked = sorted(snapshot["ranking"], key=lambda item: (-item["score"], first[item["theme_id"]]))
    credited, details, dcg = set(), [], 0.0
    for rank, item in enumerate(ranked, 1):
        labels = members[item["theme_id"]]
        counts = Counter(labels)
        dominant = max(counts, key=lambda label: (counts[label], -labels.index(label)))
        purity = counts[dominant] / len(labels)
        gain = ((2 ** checkpoint["relevance"][dominant]) - 1) * purity
        if dominant in credited:
            gain = 0.0
        credited.add(dominant)
        if rank <= k:
            dcg += gain / math.log2(rank + 1)
        details.append({**item, "rank": rank, "dominant_gold_theme": dominant,
                        "purity": purity, "credited_gain": gain})
    ideal = sorted((2 ** grade - 1 for grade in checkpoint["relevance"].values()), reverse=True)
    idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal[:k]))
    singleton_hits = 0
    for theme in checkpoint["important_singletons"]:
        # Strict top-1: a tie is not a reliable alert, nor is an impure merge.
        singleton_hits += bool(ranked and members[ranked[0]["theme_id"]] == [theme] and
                               (len(ranked) == 1 or ranked[0]["score"] > ranked[1]["score"]))
    scores = [r["score"] for r in ranked]
    return {"ndcg_at_3": dcg / idcg if idcg else None,
            "important_singletons": len(checkpoint["important_singletons"]),
            "singleton_hits_at_1": singleton_hits,
            "tied_scores": len(scores) != len(set(scores)), "ranking": details}


def evaluate(suite, predictions):
    validate_suite(suite)
    if predictions.get("schema_version") != 1:
        raise ValueError("Expected prediction schema_version=1")
    if set(predictions["scenarios"]) != {s["id"] for s in suite["scenarios"]}:
        raise ValueError("Prediction scenario IDs do not match suite")
    results, categories, ndcgs = [], defaultdict(list), []
    state_correct = state_total = state_covered = singleton_hits = singleton_total = 0
    current_correct = current_total = current_covered = 0
    for scenario in suite["scenarios"]:
        output = predictions["scenarios"][scenario["id"]]
        final = output["final"]
        validate_snapshot(final, scenario["messages"])
        if set(output["checkpoints"]) != {c["after"] for c in scenario["checkpoints"]}:
            raise ValueError("Prediction checkpoint IDs do not match suite")
        probes = probe_results(scenario, final)
        for p in probes:
            categories[p["category"]].append(p["passed"])
        state = state_metrics(scenario["messages"], final["messages"])
        checkpoints = []
        for checkpoint in scenario["checkpoints"]:
            prefix = scenario["messages"][:[m["id"] for m in scenario["messages"]].index(checkpoint["after"]) + 1]
            snapshot = output["checkpoints"][checkpoint["after"]]
            validate_snapshot(snapshot, prefix)
            ranked = ranking_metrics(checkpoint, prefix, snapshot)
            current_state = state_metrics(prefix, snapshot["messages"])
            latest_state = current_state_metrics(prefix, snapshot)
            current_correct += latest_state["correct"]
            current_total += latest_state["total"]
            current_covered += latest_state["covered"]
            state_correct += current_state["correct"]
            state_total += current_state["total"]
            state_covered += sum(r["state"] is not None for r in snapshot["messages"])
            singleton_hits += ranked["singleton_hits_at_1"]
            singleton_total += ranked["important_singletons"]
            if ranked["ndcg_at_3"] is not None:
                ndcgs.append(ranked["ndcg_at_3"])
            checkpoints.append({"after": checkpoint["after"], "ranking": ranked, "states": current_state,
                                "current_states": latest_state,
                                "theme_pairs": grouping(prefix, snapshot["messages"], "theme", "theme_id")})
        episode_messages = [(m, r) for m, r in zip(scenario["messages"], final["messages"]) if m["gold"]["episode"] is not None]
        results.append({"id": scenario["id"], "messages": len(scenario["messages"]),
                        "theme_pairs": grouping(scenario["messages"], final["messages"], "theme", "theme_id"),
                        "episode_pairs": grouping(scenario["messages"], final["messages"], "episode", "episode_id"),
                        "theme_coverage": sum(r["theme_id"] is not None for r in final["messages"]) / len(final["messages"]),
                        "episode_coverage": sum(r["episode_id"] is not None for _, r in episode_messages) / len(episode_messages) if episode_messages else None,
                        "states": state, "probes": probes, "checkpoints": checkpoints})
    def macro(field):
        values = [r[field]["f1"] for r in results if r[field]["f1"] is not None]
        return sum(values) / len(values) if values else None
    return {"adapter": predictions["adapter"], "scenarios": results,
            "aggregate": {"theme_pair_f1_macro": macro("theme_pairs"),
                          "episode_pair_f1_macro": macro("episode_pairs"),
                          "final_state_coverage": sum(r["states"]["coverage"] * r["messages"] for r in results) / sum(r["messages"] for r in results),
                          "final_state_accuracy": sum(r["states"]["correct"] for r in results) / sum(r["messages"] for r in results),
                          "checkpoint_state_coverage": state_covered / state_total,
                          "checkpoint_state_accuracy": state_correct / state_total,
                          "current_issue_state_accuracy": current_correct / current_total,
                          "current_issue_state_coverage": current_covered / current_total,
                          "ndcg_at_3_macro": sum(ndcgs) / len(ndcgs) if ndcgs else None,
                          "nonzero_relevance_checkpoints": len(ndcgs),
                          "singleton_hits_at_1": singleton_hits, "important_singletons": singleton_total,
                          "probes": {key: {"passed": sum(values), "total": len(values)} for key, values in sorted(categories.items())}}}
