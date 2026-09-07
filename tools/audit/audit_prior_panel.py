"""Independent saved-output audit. No API calls and no runtime decision writes."""
import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def ratio(n, d):
    return n / d if d else None


def audit(output):
    completed = read(output / "prediction_completion.json")
    path = output / "predictions.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == completed["prediction_sha256"]
    plan, rows, ledger = read(output / "plan.json"), read(path), read(output / "budget.json")
    assert len(rows) == len(plan["selection"]) == 8
    assert {(r["video_id"], r["frame_id"]) for r in rows} == {
        (r["video_id"], r["frame_id"]) for r in plan["selection"]}
    call_roots = [output, *(Path(p["path"]) for p in plan.get("predecessors", []))]
    for row in rows:
        key = f"{row['video_id']}_{row['frame_id']}"
        originals = [root / "calls" / key / "h0/parsed.json" for root in call_roots]
        originals = [p for p in originals if p.exists()]
        if row["h0"] is not None:
            assert len(originals) == 1, "H0 was duplicated or lacks an original API response"
            native = read(originals[0])
            expected_h0 = {q: [native[q]["selected_id"]] if q == "phase" else sorted(native[q]["selected_ids"])
                           for q in ("instrument", "verb", "target", "ivt", "phase")}
            assert row["h0"] == expected_h0
    gt_rows = read(output / "scored/no_prior_D3.json")
    truth = {(r["video_id"], r["frame_id"]): r for r in gt_rows}
    metrics = read(output / "metrics.json")
    # Recompute set counts directly from frozen prediction/GT rows; this does
    # not import the runtime scorer or its metric functions.
    for arm in ("no_prior", "prior"):
        for index in range(3):
            for task in ("instrument", "verb", "target", "ivt", "phase"):
                tp = fp = fn = exact = valid = 0
                for row in rows:
                    gt = truth[row["video_id"], row["frame_id"]]
                    if not gt["mask"][task]:
                        continue
                    valid += 1
                    prediction = row["arms"].get(arm, {}).get("snapshots", [row["h0"]] * 3)[index]
                    ids = set(prediction[task]) if prediction is not None else set()
                    expected = set(gt["gt"][task])
                    tp += len(ids.intersection(expected))
                    fp += len(ids.difference(expected))
                    fn += len(expected.difference(ids))
                    exact += prediction is not None and ids == expected
                saved = metrics[f"{arm}_D{index + 1}"]["comparison"]["arms"]["final"]["tasks"][task]
                assert (saved["tp"], saved["fp"], saved["fn"], saved["valid_targets"], saved["exact_matches"]) == (tp, fp, fn, valid, exact)
    outputs = {}
    for arm in ("no_prior", "prior"):
        judges, consensus, admission, stops = {}, Counter(), {}, Counter()
        candidates = {q: Counter() for q in ("instrument", "verb", "target", "ivt")}
        for row in rows:
            result = row["arms"].get(arm)
            if result is None:
                stops["H0_OR_BUDGET_UNAVAILABLE"] += 1
                continue
            stops[result["stop_reason"]] += 1
            assert result["h0"] == row["h0"]
            assert result["final"]["phase"] == row["h0"]["phase"]
            key = f"{row['video_id']}_{row['frame_id']}"
            u = read(output / "targets" / key / arm / "universe.json")
            props = {p["proposition_id"]: p for p in u["propositions"]}
            gt = truth[row["video_id"], row["frame_id"]]
            first = result["history"][0]
            for task, counter in candidates.items():
                if not gt["mask"][task]:
                    continue
                expected, baseline = set(gt["gt"][task]), set(row["h0"][task])
                available = {p["label_id"] for p in props.values() if p["task"] == task}
                counter["valid_targets"] += 1
                counter["h0_false_positives"] += len(baseline - expected)
                counter["h0_false_negatives"] += len(expected - baseline)
                counter["new_candidates"] += len(available - baseline)
                counter["true_new_candidates"] += len((available - baseline) & expected)
                counter["gt_labels_outside_universe"] += len(expected - available)
                if first["status"] != "VALID":
                    counter["invalid_initial_panel_targets"] += 1
                    continue
                for pid, p in props.items():
                    if p["task"] != task:
                        continue
                    c = p["label_id"]
                    vote = first["votes"][pid]
                    if c not in baseline:
                        counter["true_add_candidate_majority_supported" if c in expected else
                                "false_add_candidate_majority_supported"] += vote.get("PRESENT", 0) >= 2
                    elif c not in expected:
                        counter["h0_fp_detected_by_majority_absent"] += vote.get("ABSENT", 0) >= 2
                        counter["h0_fp_unanimously_absent"] += vote.get("ABSENT", 0) == 3
            for round_row in result["history"]:
                if round_row["status"] != "VALID":
                    continue
                rnd = round_row["round"]
                assert len(round_row["normalized"]) == 3
                collected = {pid: [] for pid in props}
                for j, review in enumerate(round_row["normalized"]):
                    counter = judges.setdefault(f"R{rnd}_J{j + 1}", Counter())
                    assert len(review["assessments"]) == len(props)
                    for item in review["assessments"]:
                        p = props[item["proposition_id"]]
                        q, c, answer = p["task"], p["label_id"], item["presence"]
                        if not gt["mask"][q]:
                            continue
                        present = c in gt["gt"][q]
                        h0_present = c in row["h0"][q]
                        current_present = c in round_row["current"][q]
                        collected[item["proposition_id"]].append(answer)
                        counter["valid_propositions"] += 1
                        if item["normalization_reasons"]:
                            counter["normalized_to_unclear"] += 1
                        if answer == "UNCLEAR":
                            counter["unclear"] += 1
                        elif (answer == "PRESENT") == present:
                            counter["correct_decisive"] += 1
                        else:
                            counter["wrong_decisive"] += 1
                        # Detection denominator includes all U ground-truth errors;
                        # UNCLEAR does not detect an error and is reported separately.
                        error = current_present != present
                        flagged = (current_present and answer == "ABSENT") or (not current_present and answer == "PRESENT")
                        counter["detection_tp" if flagged and error else "detection_fp" if flagged
                                else "detection_fn" if error else "detection_tn"] += 1
                        if h0_present != present:
                            counter["h0_error_propositions"] += 1
                for pid, answers in collected.items():
                    if len(answers) == 3 and len(set(answers)) == 1 and answers[0] != "UNCLEAR":
                        p = props[pid]
                        present = p["label_id"] in gt["gt"][p["task"]]
                        consensus["unanimous_decisive"] += 1
                        consensus["unanimous_wrong"] += (answers[0] == "PRESENT") != present
                if rnd == 1:
                    continue
                counter = admission.setdefault(f"R{rnd}", Counter())
                for q in ("instrument", "verb", "target", "ivt"):
                    if not gt["mask"][q]:
                        continue
                    h0 = set(row["h0"][q])
                    draft = set(round_row["current"][q])
                    accepted = set(round_row["last_accepted"][q])
                    expected = set(gt["gt"][q])
                    assert (h0 ^ accepted) <= (h0 ^ draft)
                    for c in h0 ^ draft:
                        helpful = (c in draft) == (c in expected)
                        allowed = (c in accepted) == (c in draft)
                        kind = "beneficial" if helpful else "harmful"
                        counter[kind + "_proposed"] += 1
                        counter[kind + "_admitted"] += allowed
        for c in judges.values():
            c["detection_precision"] = ratio(c["detection_tp"], c["detection_tp"] + c["detection_fp"])
            c["detection_recall"] = ratio(c["detection_tp"], c["detection_tp"] + c["detection_fn"])
            c["decisive_accuracy"] = ratio(c["correct_decisive"], c["correct_decisive"] + c["wrong_decisive"])
        for c in admission.values():
            for kind in ("beneficial", "harmful"):
                c[kind + "_admission_rate"] = ratio(c[kind + "_admitted"], c[kind + "_proposed"])
        outputs[arm] = {"judge_by_round": judges, "unanimous": dict(consensus), "admission_by_round": admission,
                        "stop_reasons": dict(stops), "candidate_diagnosis": candidates,
                        "rounds_are_correlated_not_independent": True}
    native = sum(Decimal(r["cost_usd"]) for r in ledger["calls"].values() if r["cost_usd"] is not None)
    unsettled = sum(Decimal(r["reserve_usd"]) for r in ledger["calls"].values() if r["cost_usd"] is None)
    carried = Decimal(plan.get("carryover_liability_usd", "0"))
    previous_native = previous_unknown = Decimal(0)
    for predecessor in plan.get("predecessors", []):
        path = Path(predecessor["path"]) / "budget.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == predecessor["ledger_sha256"]
        for row in read(path)["calls"].values():
            if row["state"] in {"NOT_SENT", "RESERVED"}:
                continue
            if row["cost_usd"] is None:
                previous_unknown += Decimal(row["reserve_usd"])
            else:
                previous_native += Decimal(row["cost_usd"])
    assert previous_native + previous_unknown == carried
    assert native + unsettled + carried <= Decimal("15.00")
    actual_calls = sum(r["state"] not in {"RESERVED", "NOT_SENT"} for r in ledger["calls"].values())
    assert actual_calls + plan.get("carryover_calls", 0) <= 188
    result = {"status": "SAVED_OUTPUT_AUDIT_PASSED", "arms": outputs,
              "accounting": {"settled_native_usd": str(native), "unsettled_liability_usd": str(unsettled),
                             "predecessor_liability_usd": str(carried), "current_requests": actual_calls,
                             "previous_requests": plan.get("carryover_calls", 0),
                             "total_settled_native_usd": str(native + previous_native),
                             "total_unsettled_reservations_usd": str(unsettled + previous_unknown)},
              "note": "Masked GT joined after completed prediction hash. No pixel truth or independence inferred from judge agreement."}
    with (output / "independent_audit.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"status": result["status"], "accounting": result["accounting"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)
