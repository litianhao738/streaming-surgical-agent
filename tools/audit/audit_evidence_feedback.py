"""Independent offline audit of the four-target shared-first-round experiment.

The entry point refuses to read any truth or score until paid inference is
closed and completion.json exists. No transport or credentials are used.
"""

import argparse
import json
import sys
from copy import deepcopy
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.check_candidate_panel_providers import redact_images
from scripts.run_evidence_feedback_trial import (
    ARMS,
    PROFILE,
    TASKS,
    proposal_wire,
    review_wire,
)
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base, gemini_h0_wire
from scripts.run_repair_revision_trial import normalize_five
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.prior_panel import COMPONENTS
from surgical_agent.research.verification.review_feedback import build_review_feedback
from surgical_agent.research.verification.semantic_coordinator import item_error
from tools.audit.audit_repair_revision import (
    PROVIDERS,
    ROUTES,
    change_counts,
    check_metrics,
    decode_boundary_json,
    identity,
    image_refs,
    indexed,
    metrics,
    money,
    read,
    require,
    sha,
)


def fingerprint(saved_wire):
    import hashlib
    return hashlib.sha256(json.dumps(saved_wire, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def same_reconstructed_input(actual, expected):
    """Ignore object key order lost by sorted state JSON, never values or lists.

    This checks reconstructed inputs only. Cache reuse still requires the saved
    full request fingerprint, including the exact original prompt text.
    """
    a, b = deepcopy(actual), deepcopy(expected)
    for body in (a, b):
        content = body["messages"][0]["content"][0]
        content["text"] = json.loads(content["text"])
    return a == b


def inspect_transition(record):
    """Independent score arithmetic and patch application on the saved state."""
    before, pool = record["before"], record["pool"]
    if not record.get("reviewed"):
        require(record["after"] == before, "unreviewed stage changed prediction")
        return
    scores, means, diagnostics = {}, {}, {}
    for proposition in pool["propositions"]:
        pid, task = proposition["id"], proposition["task"]
        values, invalid = [], {}
        for seat in SEATS:
            value = record["reviews"][seat]["judgments"].get(pid)
            error = item_error(value, task, record["image_count"])
            values.append(None if error else value["rating"])
            if error:
                invalid[seat] = error
        means[pid] = None if invalid else sum(values) / 5
        effective = [3 if value is None else value for value in values]
        diagnostics[pid] = {"scores": values, "invalid": invalid,
                            "explicit_conflict": min(effective) <= 2 and max(effective) >= 4}
        scores[task, proposition["label_id"]] = means[pid]
    require(means == record["means"] and diagnostics == record["diagnostics"], "review mean/invalid arithmetic differs")

    def support(task, label):
        value = scores[task, label]
        return value is not None and value >= 4

    def refute(task, label):
        value = scores[task, label]
        return value is not None and value <= 2

    after = deepcopy(before)
    after["ivt"] = [label for label in before["ivt"] if not refute("ivt", label)]
    for task, label in sorted(scores):
        if (task == "ivt" and label not in before["ivt"] and support(task, label)
                and all(support(t, value) for t, value in COMPONENTS[label].items())):
            after["ivt"].append(label)
    for task in TASKS[:3]:
        after[task] = sorted((set(before[task]) | {label for t, label in scores
                                                 if t == task and support(task, label)})
                            - {label for label in before[task] if refute(task, label)
                               and not any(COMPONENTS[ivt][task] == label for ivt in after["ivt"])})
    after["ivt"].sort()
    if record["status"] != "SELECTION_FAILED":
        require(after == record["after"], "saved repair differs from independently applied rules")
        issues = []
        for proposition in pool["propositions"]:
            pid, mean = proposition["id"], means[proposition["id"]]
            present = proposition["label_id"] in after[proposition["task"]]
            if mean is None or not (mean >= 4 if present else mean <= 2):
                issues.append({"candidate_id": pid, "currently_selected": present, "mean": mean,
                               "scores": diagnostics[pid]["scores"], "invalid": diagnostics[pid]["invalid"]})
        require(issues == record["issues"], "saved unresolved feedback differs")
        require(record["status"] == ("UNRESOLVED" if issues else "MODEL_PASS"), "incorrect model-pass status")
    require(record["after"]["phase"] == before["phase"], "repair changed frozen Phase")


def audit(output):
    # This guard precedes loading the dataset adapter, saved truth, or scores.
    require((output / "completion.json").is_file(), "trial has not completed; GT audit deferred")
    budget = read(output / "budget.json")
    require(budget["stopped"] is True, "paid inference is not closed; GT audit deferred")
    completion, plan, states = read(output / "completion.json"), read(output / "plan.json"), read(output / "inference_states.json")
    require(plan["profile"] == PROFILE and plan["arms"] == list(ARMS), "unexpected protocol")
    require(plan["round_cap"] == 2 and plan["threshold"] == 4 and plan["max_calls"] == 76, "protocol caps changed")
    require((output / "execution.lock").read_text().strip() == sha(output / "plan.json"), "plan execution binding changed")
    source_hashes = {}
    for path, digest in plan["source_sha256"].items():
        require(sha(output / "frozen_source" / path) == digest and sha(ROOT / path) == digest, "source drift: " + path)
        source_hashes[path] = digest
    require(sha(Path(plan["previous_budget"]) / "budget.json") == plan["previous_budget_sha256"], "previous ledger changed")
    require(sha(Path(plan["selection_source"])) == plan["selection_sha256"], "selection source changed")
    selection = indexed(plan["selection"])
    require(len(selection) == 4 and set(states) == set(selection), "four target identities must be retained")
    manifest = read(Path(plan["selection_source"]))
    require(manifest["no_gt_label_values_used_for_selection"] is True, "selection not mask-only")
    prior = []
    for path, digest in plan["earlier_selection_sha256"].items():
        require(sha(Path(path)) == digest, "earlier plan changed")
        prior.extend(read(Path(path))["selection"])
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    bases = {}
    for key, row in selection.items():
        expected = [r for r in manifest["selection"] if r["video_id"] == row["video_id"]][2]
        require(identity(expected) == key, "not the fixed time-index selection")
        require(adapter.entries[row["video_id"]].split is DatasetSplit.TRAINING, "non-Training target")
        require(set(row["gt_availability_only"]) == set(TASKS)
                and all(type(v) is bool and v for v in row["gt_availability_only"].values()), "five masks not available")
        require(not any(identity(r) == key for r in prior), "target reused from development/confirmation")
        require(not any(set(r["causal_frame_ids"]) & set(row["causal_frame_ids"])
                        for r in prior if r["video_id"] == row["video_id"]), "causal image window overlaps earlier experiment")
        frames = row["causal_frame_ids"]
        require(1 <= len(frames) <= 3 and frames[-1] == row["frame_id"]
                and all(b - a == 25 for a, b in pairwise(frames)), "noncausal or padded window")
        for im in row["images"]:
            require(sha(Path(im["path"])) == im["sha256"], "selected image hash changed")
        bases[key] = build_gemini_base(adapter, row)
        require(redact_images(gemini_h0_wire(bases[key])) == read(output / "h0_preflight" / f"{key}.json"), "H0 preflight changed")

    calls = budget["calls"]
    require(len(calls) == completion["post_calls"] <= 76, "POST count differs or cap exceeded")
    require([r["index"] for r in calls] == list(range(len(calls))), "nonsequential call ledger")
    require(len({(r["target"], r["stage"], r["seat"]) for r in calls}) == len(calls), "duplicate paid stage")
    parsed, bodies, folders, call_hashes = {}, {}, {}, {}
    for row in calls:
        key = row["target"], row["stage"], row["seat"]
        folder = output / "calls" / f"{row['index']:03d}_{'_'.join(key)}"
        require(read(folder / "record.json") == row, "record/ledger mismatch")
        body = read(folder / "request.json")
        seat = row["seat"]
        require(body["model"] == (plan["h0"] if seat == "base" else plan["models"][seat]), "model changed")
        require(image_refs(body) == image_refs(read(output / "h0_preflight" / f"{row['target']}.json")), "API images/order/detail changed")
        if seat in ROUTES:
            require(body["provider"]["only"] == [ROUTES[seat]] and not body["provider"]["allow_fallbacks"], "provider unlocked")
        parsed[key], bodies[key], folders[key] = None, body, folder
        for path in folder.glob("*.json"):
            call_hashes[str(path)] = sha(path)
        if row["status"].startswith("JSON_PARSED"):
            response = read(folder / "response.json")
            raw, choice = response["body"], response["body"]["choices"][0]
            require(response["http_status"] == 200 and raw["model"] == body["model"]
                    and choice["finish_reason"] == "stop" and not choice["message"].get("refusal"), "invalid parsed transport")
            if seat in PROVIDERS:
                require(raw["provider"] == PROVIDERS[seat], "response provider mismatch")
            parsed[key] = (json.loads(choice["message"]["content"]) if seat == "base" else
                           decode_boundary_json(choice["message"]["content"], allow_identical_core_duplicates=True)[0])
        usage = row.get("usage") or {}
        if row["charge_kind"] == "native":
            expected = (Decimal(str(usage["cost_in_usd_ticks"])) / Decimal(10**10)
                        if seat == "grok" else Decimal(str(usage["cost"])))
        elif row["charge_kind"] == "conservative_estimate":
            require(seat == "qwen", "unexpected estimated account")
            inp, out = map(Decimal, plan["rates"][seat])
            expected = Decimal(usage["prompt_tokens"]) * inp + Decimal(usage["completion_tokens"]) * out
        else:
            require(row["charge_kind"] == "unknown_reserved", "unknown charge kind")
            expected = Decimal(row["reserve"])
        require(Decimal(row["charge"]) == expected and expected >= 0, "cost mismatch")
    for account, total in budget["occupied"].items():
        expected = Decimal(plan["carried_occupied"][account]) + sum(
            (Decimal(r["charge"]) for r in calls if r["account"] == account), Decimal(0))
        require(Decimal(total) == expected and expected <= Decimal(plan["limits"][account]), "balance/limit mismatch")

    shared_panels, transitions, feedback_differences = [], 0, []
    for key, state in states.items():
        base, selected = bases[key], selection[key]
        h0_key = (key, "h0", "base")
        require(h0_key in parsed, "missing shared H0 request")
        require(bodies[h0_key] == redact_images(gemini_h0_wire(base)), "H0 wire mismatch")
        if state["h0"] is None:
            require(parsed[h0_key] is None, "valid H0 silently discarded")
            continue
        raw_h0 = parsed[h0_key]
        normalized_h0 = {t: [raw_h0[t]["selected_id"]] if t == "phase" else raw_h0[t]["selected_ids"] for t in TASKS}
        require(normalized_h0 == state["h0"], "shared H0 differs from paid response")
        require(len(state["shared"]["history"]) == 1, "first round not shared exactly once")
        first = state["shared"]["history"][0]
        require(first["before"] == state["h0"] and first["pool_before"] == make_pool(state["h0"])
                and first["input_issues"] == [], "first round starts from altered H0")
        histories = [("shared", first)] + [(name, arm["history"][0]) for name, arm in state["arms"].items() if arm["history"]]
        for name, record in histories:
            n = record["round"]
            saved_record = output / "targets" / key / f"{name}_round_{n}.json"
            require(read(saved_record) == record, "target record differs from state snapshot")
            if n == 2:
                require(record["before"] == first["after"] and record["pool_before"] == first["pool"]
                        and record["input_issues"] == first["issues"], "R2 arms do not share initial state/pool/issues")
            feedback = build_review_feedback(first["pool"], first["reviews"], first["issues"], image_count=len(base.images)) if name == ARMS[1] else None
            require(record["review_evidence_feedback"] == feedback, "feedback differs from valid first-round evidence")
            proposal_key = (key, f"{name}_proposal_{n}", "base")
            require(same_reconstructed_input(bodies[proposal_key], redact_images(proposal_wire(
                base, selected, record["before"], record["pool_before"], record["input_issues"], feedback))), "proposal input changed")
            require(parsed[proposal_key] == record["proposal"], "proposal differs from paid answer")
            if record["proposal"] is None:
                require(record["status"] == "PROPOSAL_FAILED" and record["after"] == record["before"], "failed proposal did not fall back")
                continue
            require(record["pool"] == make_pool(record["before"], record["proposal"], record["pool_before"]), "proposal pool changed")
            if not record["pool"]["propositions"]:
                require(record["status"] == "EMPTY_POOL_UNVERIFIED", "empty pool falsely passed")
                continue
            expected_bodies = {s: redact_images(review_wire(s, base, selected, record["pool"])) for s in SEATS}
            source_name = record.get("shared_review", {}).get("source_arm", name)
            raw = {}
            if "shared_review" in record:
                reuse = record["shared_review"]
                source_record = Path(reuse["source_record"])
                require(n == 2 and source_name in ARMS and source_name != name, "review shared across wrong round/arm")
                require(sha(source_record) == reuse["source_record_sha256"], "shared record hash changed")
                origin = read(source_record)
                require(origin["pool"] == record["pool"] and origin["raw_reviews"] == record["raw_reviews"], "shared review raw/pool differ")
                require(record["new_review_calls"] == 0 and not any((key, f"{name}_review_2", s) in parsed for s in SEATS), "shared panel billed twice")
                shared_panels.append({"target": key, "source_arm": source_name, "recipient_arm": name})
            for seat in SEATS:
                review_key = key, f"{source_name}_review_{n}", seat
                require(review_key in parsed and same_reconstructed_input(bodies[review_key], expected_bodies[seat]), "review inputs changed")
                require(record["review_request_fingerprints"][seat] == fingerprint(bodies[review_key]), "actual full request fingerprint differs")
                if "shared_review" in record:
                    require(record["shared_review"]["request_fingerprints"][seat] == record["review_request_fingerprints"][seat]
                            == origin["review_request_fingerprints"][seat], "shared full request differs")
                raw[seat] = parsed[review_key]
                if "shared_review" in record:
                    ref = record["shared_review"]["responses"][seat]
                    response_path = folders[review_key] / "response.json"
                    require(Path(ref["path"]) == response_path and ref["sha256"] == (sha(response_path) if response_path.exists() else None), "shared response hash changed")
            require(raw == record["raw_reviews"], "saved raw review differs, including failure response")
            normalized, normalizer_diagnostics = normalize_five(raw, record["pool"], len(base.images))
            require(normalized == record["reviews"] and normalizer_diagnostics == record["format_diagnostics"], "review normalization changed")
            inspect_transition({**record, "image_count": len(base.images)})
            transitions += 1
        second = {name: arm["history"][0] for name, arm in state["arms"].items() if arm["history"]}
        if len(second) == 2:
            control = deepcopy(bodies[key, f"{ARMS[0]}_proposal_2", "base"])
            treated = deepcopy(bodies[key, f"{ARMS[1]}_proposal_2", "base"])
            packet = json.loads(treated["messages"][0]["content"][0]["text"])
            require(packet.pop("review_evidence_feedback") == second[ARMS[1]]["review_evidence_feedback"], "treatment feedback field missing")
            treated["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
            require(control == treated, "R2 proposer differs beyond feedback field")
            feedback_differences.append(key)
        if first["status"] == "MODEL_PASS":
            require(not second and all(a["status"] == "SHARED_MODEL_PASS" for a in state["arms"].values()), "forced second round after pass")

    rounds = {}
    for number in (1, 2):
        path = output / f"round_{number}_predictions.json"
        rows, saved = read(path), read(output / "scores" / f"round_{number}.json")
        require(sha(path) == saved["prediction_sha256"] and set(indexed(rows)) == set(selection), "snapshot identity/hash changed")
        for row in rows:
            state = states[identity(row)]
            require(row["h0"] == state["h0"], "snapshot altered shared H0")
            shared = state["shared"]
            require(row["shared_first_round"] == {"final": shared["current"], "status": shared["status"],
                                                  "reviewed": shared["reviewed"]}, "snapshot altered shared R1")
            for name in ARMS:
                snapshot_arm = row["arms"][name]
                expected = shared["current"] if number == 1 else state["arms"][name]["current"]
                require(snapshot_arm["final"] == expected, "arm snapshot differs from real terminal state")
                if number == 1:
                    require(not snapshot_arm["round2_attempted"] and not snapshot_arm["round2_reviewed"]
                            and snapshot_arm["actual_rounds"] == int(bool(shared["history"])), "R1 snapshot contains R2 work")
                else:
                    arm_state = state["arms"][name]
                    for field in ("status", "round2_attempted", "round2_reviewed"):
                        require(snapshot_arm[field] == arm_state[field], "R2 snapshot status differs")
                    require(snapshot_arm["actual_rounds"] == int(bool(shared["history"])) + int(arm_state["round2_attempted"])
                            == completion["actual_rounds"][identity(row)][name], "actual round count differs")
        arms = {}
        for arm in ("shared_first_round", *ARMS):
            truth = read(output / "scores" / f"round_{number}_{arm}_truth.json")
            require(set(indexed(truth)) == set(selection), "truth discarded target")
            for row in truth:
                source_row = indexed(rows)[identity(row)]
                expected = source_row[arm]["final"] if arm == "shared_first_round" else source_row["arms"][arm]["final"]
                require(row["h0"] == source_row["h0"] and row["final"] == expected, "scoring uses altered prediction")
                require(all(row["mask"].values()), "fresh scoring lacks predeclared five masks")
            _, fresh = score_saved(adapter, truth)
            require(fresh == truth, "saved GT or masks differ from fresh dataset")
            actual = metrics(truth, "final")
            check_metrics(actual, saved["reports"][arm]["arms"]["final"]["tasks"])
            h0_metrics = metrics(truth, "h0")
            check_metrics(h0_metrics, saved["reports"][arm]["arms"]["h0"]["tasks"])
            changes = change_counts(truth)
            for field, value in saved["reports"][arm]["paired"]["h0_to_final"]["frames"].items():
                require(changes["frames"][field] == value, "paired change category mismatch")
            arms[arm] = {"metrics": actual, "changes_from_h0": changes,
                         "total_label_errors": sum(m["fp"] + m["fn"] for m in actual.values())}
        control_truth = read(output / "scores" / f"round_{number}_{ARMS[0]}_truth.json")
        treatment_truth = indexed(read(output / "scores" / f"round_{number}_{ARMS[1]}_truth.json"))
        paired = [{**r, "h0": r["final"], "final": treatment_truth[identity(r)]["final"]} for r in control_truth]
        rounds[str(number)] = {"arms": arms, "h0_metrics": h0_metrics,
                              "treatment_vs_control": change_counts(paired)}
    groups = {}
    shared_sources = {(r["target"], f"{r['source_arm']}_review_2") for r in shared_panels}
    for row in calls:
        group = ("shared_h0" if row["stage"] == "h0" else "shared_first_round" if row["stage"].startswith("shared_")
                 else "shared_round2_review" if (row["target"], row["stage"]) in shared_sources
                 else next(a for a in ARMS if row["stage"].startswith(a + "_")))
        groups.setdefault(group, []).append(row)
    grouped = {name: {"calls": len(values), "costs": money(values)} for name, values in groups.items()}
    require(grouped == completion["cost_groups"], "cost attribution changed or doubled shared review")
    require(len(shared_panels) == completion["shared_second_round_panels"], "shared panel count mismatch")
    result = {"verified": True, "completed": completion["completed"], "plan_sha256": sha(output / "plan.json"),
              "source_hashes_verified": len(source_hashes), "selection_training_causal_masks_nonoverlap_verified": True,
              "four_targets": list(selection), "shared_h0_r1_verified": True,
              "r2_feedback_only_verified_targets": feedback_differences, "shared_r2_panels": shared_panels,
              "independent_transition_checks": transitions, "post_calls": len(calls), "post_call_cap": 76,
              "new_costs": money(calls), "cost_groups": grouped, "rounds": rounds,
              "raw_call_sha256": call_hashes, "audit_source_sha256": sha(Path(__file__)),
              "gt_read_only_after_paid_inference_closed": True,
              "caveat": "Four Training targets; development comparison, not Test or independent-surgery generalization."}
    destination = output / "independent_audit.json"
    require(not destination.exists(), "independent audit already exists; preserve it")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": True, "post_calls": len(calls), "new_costs": money(calls),
                      "final_f1": {a: {t: m["micro_f1"] for t, m in values["metrics"].items()}
                                   for a, values in rounds["2"]["arms"].items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
