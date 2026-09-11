"""Follow-up diagnosis removing duplicate target ratings; no prediction publication."""
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
from scripts.run_prior_panel_trial import build_base, now, read, save
from scripts.run_target_interaction_diagnosis import (
    SOURCE,
    diagnostic_body,
    evidence_schema,
)
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.research.verification.candidate_coordinator import SEATS

PREVIOUS = ROOT / "artifacts/preflight/target_interaction_diagnosis_20260908_v1"


def single_body(seat, base, selected, pool):
    body = diagnostic_body(seat, base, selected, pool)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    packet["task"] = "For each anatomical Target candidate, distinguish visibility from actual interaction by ANY instrument in the CURRENT frame."
    packet["instructions"] = (
        "Return target_evidence ONLY. There is no separate scores map. For each target, visible_score rates whether the tissue is visible; "
        "interaction_score is the sole rating of whether it is the anatomical recipient of any instrument's current action. "
        "A visible organ is not necessarily an interaction target. Any instrument and any actual interaction may establish a Target; "
        "rejecting one named IVT does not refute that Target. Do not require the interaction to match a proposed IVT, "
        "and do not invent an interaction to justify a visible organ. "
        "Report observable evidence briefly, not hidden reasoning. Identify the instrument and action if supported; otherwise write NONE or UNCERTAIN. "
        "Give the claimed contact site in CURRENT image index 2 as normalized [left,top,right,bottom] fractions in [0,1], not pixel coordinates. "
        "Use [] when no defensible contact site is observable. Mark the contact region, not the whole organ. "
        "Describe occlusion and competing tissue interpretations in uncertainty. History is causal motion context only; no future images. "
        "Do not infer truth from candidate membership. Do not output scores for Instrument, Verb or IVT. Return only the specified JSON.")
    schema = evidence_schema(pool)
    schema["properties"].pop("scores")
    schema["required"] = ["target_evidence"]
    packet["response_schema"] = schema
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    if body["response_format"]["type"] == "json_schema":
        body["response_format"]["json_schema"].update(name="single_target_interaction_v1", schema=schema)
    return body, schema


def analyze(raw, pool, schema):
    if not isinstance(raw, dict) or not isinstance(raw.get("target_evidence"), dict):
        raise TypeError("missing target evidence")
    evidence = raw["target_evidence"]
    expected = {p["id"] for p in pool["propositions"] if p["task"] == "target"}
    if set(evidence) != expected:
        raise ValueError("target IDs differ")
    for entry in evidence.values():
        if any(type(entry.get(k)) is not int or not 1 <= entry[k] <= 5 for k in ("visible_score", "interaction_score")):
            raise ValueError("invalid target rating")
    result = {"numeric_status": "VALID", "evidence_status": "VALID", "raw": raw}
    try:
        Draft202012Validator(schema).validate(raw)
        for entry in evidence.values():
            box = entry["contact_bbox_xyxy"]
            if len(box) not in (0, 4) or (box and not (box[0] < box[2] and box[1] < box[3])):
                raise ValueError("invalid box geometry")
    except (ValidationError, ValueError) as exc:
        result["evidence_status"] = type(exc).__name__
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-output-tokens", type=int, default=3072)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new output required")
    old_plan = read(PREVIOUS / "plan.json")
    selected = old_plan["selection"]
    budget = read((args.resume or PREVIOUS) / "budget.json")
    transport.LIMITS = {k: Decimal(v) for k, v in budget["limits"].items()}
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    reused = []
    if args.resume:
        reused = [r for r in read(args.resume / "results.json") if all(v["numeric_status"] == "VALID" for v in r["reviews"].values())]
    reused_ids = {(r["video_id"], r["frame_id"]) for r in reused}
    prepared = []
    for s in selected:
        if (s["video_id"], s["frame_id"]) in reused_ids:
            continue
        original = read(SOURCE / "targets" / s["key"] / "result.json")
        prepared.append((s, build_base(adapter, s), original["history"][0]["pool"], original["h0"]))
    sources = [Path(__file__), ROOT / "scripts/run_target_interaction_diagnosis.py",
               ROOT / "scripts/run_candidate_panel_trial.py", ROOT / "scripts/check_candidate_panel_providers.py"]
    hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    save(args.output / "plan.json", {"created_utc": now(), "selection": selected, "max_post_calls": 5*len(prepared),
         "resume": str(args.resume) if args.resume else None, "reused_ids": sorted(reused_ids),
         "max_output_tokens": args.max_output_tokens,
         "source_sha256": hashes, "previous": str(PREVIOUS), "limits": budget["limits"],
         "carried_occupied": budget["occupied"], "scope": "single_authoritative_interaction_rating_diagnosis",
         "predictions_published": False, "gt_and_previous_answers_hidden": True})
    for f in sources:
        dest = args.output / "frozen_source" / f.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.read_bytes())
    if not args.execute:
        print("Prepared only")
        return
    calls = transport.Calls(args.output, budget["occupied"])
    results = reused[:]
    for s, base, pool, h0 in prepared:
        def one(seat, s=s, base=base, pool=pool):
            body, schema = single_body(seat, base, s, pool)
            body["max_tokens"] = args.max_output_tokens
            raw = calls.call(s["key"], "single_interaction", seat, body)
            try:
                return analyze(raw, pool, schema)
            except (ValueError, TypeError, KeyError):
                return {"numeric_status": "INVALID", "evidence_status": "INVALID", "raw": raw}
        with ThreadPoolExecutor(max_workers=5) as workers:
            reviews = dict(zip(SEATS, workers.map(one, SEATS), strict=True))
        record = {"video_id": s["video_id"], "frame_id": s["frame_id"], "pool": pool, "h0": h0, "reviews": reviews}
        if all(r["numeric_status"] == "VALID" for r in reviews.values()):
            ids = [p["id"] for p in pool["propositions"] if p["task"] == "target"]
            means = {pid: sum(r["raw"]["target_evidence"][pid]["interaction_score"] for r in reviews.values())/5 for pid in ids}
            record["interaction_means"] = means
            # Exploratory Target-only counterfactual: other heads stay frozen.
            draft = deepcopy(h0)
            draft["target"] = sorted(c for c in h0["target"] if means[f"target_{c}"] > 2)
            draft["target"] += [int(pid.split("_")[1]) for pid in ids if int(pid.split("_")[1]) not in h0["target"] and means[pid] >= 4]
            draft["target"].sort()
            record["target_only_numeric_counterfactual"] = draft
        save(args.output / "targets" / s["key"] / "result.json", record)
        results.append(record)
        print(json.dumps({"target": s["key"], "means": record.get("interaction_means"),
                          "evidence_status": {k:v["evidence_status"] for k,v in reviews.items()}}), flush=True)
    calls.stopped = True
    calls.persist()
    save(args.output / "results.json", results)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest() == h for f,h in hashes.items())


if __name__ == "__main__":
    main()
