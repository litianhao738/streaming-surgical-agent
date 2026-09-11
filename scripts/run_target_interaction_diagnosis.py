"""Paired original-score / explicit-interaction diagnosis; no H0 or pool changes."""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_candidate_panel_trial as transport
from scripts.check_candidate_panel_providers import body_for
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_prior_panel_trial import build_base, now, read, save
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.research.verification.candidate_coordinator import (
    SEATS,
    aggregate,
    score_schema,
    select,
)

SOURCE = ROOT / "artifacts/preflight/candidate_panel_development_20260907_v2"
ALLOWANCES = {"openrouter_usd": Decimal("0.40"), "xai_usd": Decimal("0.10"), "aliyun_cny": Decimal("0.20")}


def evidence_schema(pool):
    schema = score_schema(pool)
    props = {}
    for p in pool["propositions"]:
        if p["task"] != "target":
            continue
        fields = {
            "visible_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "interaction_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "instrument": {"type": "string", "maxLength": 120},
            "action": {"type": "string", "maxLength": 120},
            "contact_bbox_xyxy": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "maxItems": 4},
            "observation": {"type": "string", "maxLength": 1000},
            "uncertainty": {"type": "string", "maxLength": 1000},
        }
        props[p["id"]] = {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}
    schema["properties"]["target_evidence"] = {"type": "object", "properties": props,
                                                  "required": list(props), "additionalProperties": False}
    schema["required"].append("target_evidence")
    return schema


def diagnostic_body(seat, base, selected, pool):
    body = body_for(seat, base, selected, pool, gemini_json_object=True)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    packet["task"] = ("Rate dataset candidate labels in the CURRENT target frame. For every anatomical Target, "
                      "separately report whether it is visible and whether ANY instrument is interacting with it.")
    packet["rating_scale"] = {"1": "clearly refuted", "2": "likely false", "3": "uncertain/insufficient evidence",
                              "4": "likely supported", "5": "clearly supported"}
    packet["instructions"] = (
        "Instrument scores concern visible instrument identity. Verb scores require an action actually performed by an instrument. "
        "Target scores concern the anatomical recipient of an instrument's current action, not all visible anatomy. "
        "Any instrument and any actual interaction may establish a Target; rejecting one named IVT does not refute that Target. "
        "Do not require the interaction to match a proposed IVT or invent an interaction to justify a visible organ. "
        "For each target, visible_score rates visibility; interaction_score rates actual instrument interaction and must equal its scores entry. "
        "Describe only observable evidence briefly, not hidden reasoning. Identify the instrument/action if supported; otherwise write NONE or UNCERTAIN. "
        "Give the interaction site in the CURRENT image (index 2) as normalized [left,top,right,bottom] coordinates; "
        "use [] when no defensible site is observable. The box must mark the claimed contact region, not merely the organ. "
        "Mention occlusion or competing tissue interpretation in uncertainty. Do not infer certainty from candidate membership. "
        "History images are causal motion context only. No future image is available. Return the specified JSON object only.")
    packet["response_schema"] = evidence_schema(pool)
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    body["max_tokens"] = 3072
    if body["response_format"]["type"] == "json_schema":
        body["response_format"]["json_schema"].update(name="target_interaction_diagnosis_v1", schema=evidence_schema(pool))
    return body


def validate_evidence(raw, pool):
    Draft202012Validator(evidence_schema(pool)).validate(raw)
    for pid, entry in raw["target_evidence"].items():
        if entry["interaction_score"] != raw["scores"][pid]:
            raise ValueError("target interaction rating differs from candidate score")
        box = entry["contact_bbox_xyxy"]
        if len(box) not in (0, 4) or (box and not (box[0] < box[2] and box[1] < box[3])):
            raise ValueError("invalid contact box geometry")
    return raw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("new output required")
    old_plan = read(SOURCE / "plan.json")
    selected = old_plan["selection"][:2]
    sources = [Path(__file__), ROOT / "scripts/run_candidate_panel_trial.py",
               ROOT / "scripts/check_candidate_panel_providers.py",
               ROOT / "src/surgical_agent/research/verification/candidate_coordinator.py"]
    hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    carried = read(SOURCE / "budget.json")["occupied"]
    transport.LIMITS = {k: min(transport.LIMITS[k], Decimal(carried[k]) + v) for k, v in ALLOWANCES.items()}
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    prepared = []
    for s in selected:
        original = read(SOURCE / "targets" / s["key"] / "result.json")
        base = build_base(adapter, s)
        pool = original["history"][0]["pool"]
        prepared.append((s, base, pool, original["h0"]))
    plan = {"created_utc": now(), "scope": "diagnostic_replay_of_two_known_errors_not_validation_set",
            "selection": selected, "max_post_calls": 20, "models": transport.MODELS,
            "arms": ["original_repeat", "explicit_interaction"], "source_sha256": hashes,
            "carried_occupied": carried, "additional_allowances": {k: str(v) for k, v in ALLOWANCES.items()},
            "limits": {k: str(v) for k, v in transport.LIMITS.items()},
            "gt_hidden_until_scoring": True, "predictions_published": False,
            "caveat": "Task wording and observable-evidence output both change; not a pure single-factor wording ablation."}
    save(args.output / "plan.json", plan)
    for f in sources:
        dest = args.output / "frozen_source" / f.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.read_bytes())
    if not args.execute:
        print("Prepared only")
        return
    calls, records = transport.Calls(args.output, carried), []
    for s, base, pool, h0 in prepared:
        record = {"video_id": s["video_id"], "frame_id": s["frame_id"], "h0": h0, "pool": pool, "arms": {}}
        for arm in plan["arms"]:
            def one(seat, arm=arm, base=base, s=s, pool=pool):
                if arm == "original_repeat":
                    body = body_for(seat, base, s, pool, gemini_json_object=True)
                    body["max_tokens"] = 1536
                else:
                    body = diagnostic_body(seat, base, s, pool)
                return calls.call(s["key"], arm, seat, body)
            with ThreadPoolExecutor(max_workers=5) as workers:
                raw = dict(zip(SEATS, workers.map(one, SEATS), strict=True))
            out = {"raw": raw, "status": "FAILED", "final": deepcopy(h0)}
            try:
                means, clean = aggregate(raw, pool)
                out.update(means=means, normalized=clean, final=select(h0, pool, means), status="VALID_NUMERIC")
                if arm == "explicit_interaction":
                    out["evidence_validation"] = {}
                    for seat, answer in raw.items():
                        try:
                            validate_evidence(answer, pool)
                            out["evidence_validation"][seat] = "VALID"
                        except (ValidationError, ValueError, TypeError, KeyError) as exc:
                            # Report schema failure separately; don't discard valid numerical scores.
                            out["evidence_validation"][seat] = type(exc).__name__
            except (ValueError, TypeError, KeyError):
                pass
            record["arms"][arm] = out
            save(args.output / "targets" / s["key"] / f"{arm}.json", {"pool": pool, **out})
            print(json.dumps({"target": s["key"], "arm": arm, "status": out["status"],
                              "target_means": {k: v for k, v in out.get("means", {}).items() if k.startswith("target_")}}), flush=True)
        records.append(record)
        save(args.output / "diagnostic_predictions.json", records)
    calls.stopped = True
    calls.persist()
    summary = {"records": [], "costs_native": {}, "post_calls": len(calls.rows), "semantic_generalization_proven": False}
    for arm in plan["arms"]:
        rows = [{"video_id": r["video_id"], "frame_id": r["frame_id"], "h0": r["h0"], "h1": None,
                 "final": r["arms"][arm]["final"]} for r in records]
        comparison, scored = score_saved(adapter, rows)
        save(args.output / f"{arm}_comparison.json", comparison)
        save(args.output / f"{arm}_scored.json", scored)
        summary[arm] = comparison
    for r in records:
        values = {"video_id": r["video_id"], "frame_id": r["frame_id"], "reviewers": {}}
        for seat in SEATS:
            raw = r["arms"]["explicit_interaction"]["raw"].get(seat) or {}
            values["reviewers"][seat] = {"original_scores": (r["arms"]["original_repeat"]["raw"].get(seat) or {}).get("scores"),
                                         "target_evidence": raw.get("target_evidence")}
        summary["records"].append(values)
    for account in transport.LIMITS:
        summary["costs_native"][account] = str(sum(Decimal(r["charge"]) for r in calls.rows
                                                    if r["account"] == account and r["charge_kind"] == "native"))
    save(args.output / "summary.json", summary)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest() == h for f, h in hashes.items())
    print(json.dumps({"post_calls": len(calls.rows), "native_costs": summary["costs_native"]}), flush=True)


if __name__ == "__main__":
    main()
