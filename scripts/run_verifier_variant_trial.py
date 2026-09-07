"""Bounded verifier-only replay with a shared, append-only six-hour-goal budget.

This separate runner does not change historical experiments or generate H0.
It checks frozen H0/locator/proposal bindings, emits paired neutral reviews, and
persists all predictions before optional task-masked scoring. Testing is banned.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_diff_review_trial import _restore_images
from scripts.run_presence_review_trial import (
    DirectCalls,
    _mask_only,
    accounting,
    build_review_request,
    safe_wire,
    wire_body,
)
from surgical_agent.api.contracts import ApiRequest, canonical_json_bytes, thaw_json
from surgical_agent.api.credentials import assert_secret_absent, resolve_api_key
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.schema import validator_for
from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.joint_api_vlm import JointPerceptionRequestBuilder
from surgical_agent.perception.main_h0 import load_main_h0_prompt
from surgical_agent.perception.ontology_prompt import _TASK_NAMES, _ivt_rows
from surgical_agent.research.verification.diff_review import build_change_claims
from surgical_agent.research.verification.final_only_grounded import (
    final_labels,
    prepare_grounded_review,
)
from surgical_agent.research.verification.grounded_pipeline import run_grounded_target
from surgical_agent.research.verification.grounded_repair import make_contact_crops
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
    evaluate_presence_review,
)

MODEL = "qwen/qwen3.8-max-0902"
GOAL_CAP_USD = Decimal("12.00")
DEVELOPMENT_CAP_USD = Decimal("8.00")
PER_CALL_RESERVE_USD = Decimal("2.60")
PRICING_URL = "https://openrouter.ai/api/v1/models/qwen/qwen3.8-max-0902/endpoints"
VARIANTS = ("numeric1000", "names1000")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def checked_pricing(snapshot):
    response = snapshot.get("response", snapshot)
    if response["data"]["id"] != MODEL:
        raise ValueError("pricing snapshot model mismatch")
    endpoint = next(e for e in response["data"]["endpoints"] if e["tag"] == "alibaba")
    if endpoint["status"] != 0 or endpoint.get("provider_name") != "Alibaba":
        raise ValueError("strict Alibaba endpoint is unavailable")
    rates = endpoint["pricing"]
    if not {"prompt", "completion", "input_cache_read", "input_cache_write"} <= set(rates):
        raise ValueError("pricing metadata omits an input/output rate")
    if any(not Decimal(str(value)).is_finite() or Decimal(str(value)) < 0 for value in rates.values()):
        raise ValueError("pricing rates must be finite and nonnegative")
    allowed = {"prompt", "completion", "input_cache_read", "input_cache_write", "discount"}
    if any(Decimal(str(value)) != 0 for key, value in rates.items() if key not in allowed):
        raise ValueError("additional priced service invalidates the reservation envelope")
    input_rate = max(Decimal(rates.get(key, "0")) for key in
                     ("prompt", "input_cache_read", "input_cache_write"))
    output_rate = Decimal(rates["completion"])
    upper = Decimal(endpoint["context_length"]) * input_rate + Decimal(4096) * output_rate
    if (type(endpoint["context_length"]) is not int or not 1 <= endpoint["context_length"] <= 1000000
            or not 1 <= endpoint["max_prompt_tokens"] <= endpoint["context_length"]
            or input_rate > Decimal(".0000025") or output_rate > Decimal(".000006")
            or upper > PER_CALL_RESERVE_USD or endpoint["max_completion_tokens"] < 4096):
        raise ValueError("pricing/context exceeds the approved $2.60 call reservation")
    return endpoint, {"context_tokens": endpoint["context_length"],
                      "maximum_input_rate": str(input_rate), "output_token_cap": 4096,
                      "output_rate": str(output_rate), "calculated_upper_usd": str(upper),
                      "reserved_usd": str(PER_CALL_RESERVE_USD)}


def _source_files():
    paths = [Path(__file__), ROOT / "scripts/run_presence_review_trial.py",
             ROOT / "scripts/run_diff_review_trial.py", ROOT / "scripts/run_grounded_api_pipeline.py",
             ROOT / "configs/perception/joint_openrouter_h0.yaml",
             ROOT / "docs/VERIFIER_NAMED_ABLATION_PROTOCOL_2026-09-07.md"]
    paths += [ROOT / "src/surgical_agent" / name for name in (
        "api/schema.py", "perception/ontology_prompt.py", "perception/final_only.py",
        "perception/main_h0.py", "perception/joint_api_vlm.py", "perception/context_builder.py",
        "perception/prompts/perception_prompt_main_h0.txt",
        "research/signals/resources/ivt_components_v1.csv", "research/verification/diff_review.py",
        "research/verification/presence_review.py", "research/verification/final_only_grounded.py",
        "research/verification/grounded_pipeline.py", "research/verification/grounded_repair.py",
        "evaluation/repair_comparison.py")]
    return paths


def require_split(source_split, *, confirmation, variants=None):
    if source_split not in ("Training", "Validation"):
        raise ValueError("Testing is forbidden; only Training or Validation can run")
    if source_split == "Validation" and (not confirmation or variants is not None and len(variants) != 1):
        raise ValueError("Validation requires explicit confirmation and one preselected variant")
    if source_split == "Training" and confirmation:
        raise ValueError("Training development cannot be reported as Validation confirmation")


def save_plan(output, plan, source_files):
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    if (output / "plan.json").exists():
        if read(output / "plan.json") != plan:
            raise ValueError("preflight changed; use a fresh directory")
        return False
    if output.exists():
        raise ValueError("fresh output directory required")
    output.mkdir(parents=True)
    atomic_write_json(output / "plan.json", plan)
    for path in source_files:
        destination = output / "frozen_source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    return True


def prepare_collect(args):
    import torch

    torch.set_num_threads(2)
    manifest = read(args.targets_json)
    selection = deepcopy(manifest["selection"])
    require_split(manifest["source_split"], confirmation=args.confirmation, variants=args.variants)
    keys = [f"{s['video_id']}_{s['frame_id']}" for s in selection]
    if not 1 <= len(keys) <= 24 or len(keys) != len(set(keys)) or 3 * len(keys) > args.max_calls:
        raise ValueError("fixed collection must contain 1..24 targets within its three-call cap")
    if (args.output / "execution.lock").exists():
        raise ValueError("collection already dispatched; never repeat")
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    config = load_api_config(ROOT / "configs/perception/joint_openrouter_h0.yaml")
    builder, loader = JointPerceptionRequestBuilder(config=config), CausalApiMediaLoader()
    context = CausalPerceptionContextBuilder(max_frames=3, max_images=3, selection_strategy="fixed_all",
                                            history_image_detail="low", target_image_detail="high")
    hashes = {str(args.targets_json.resolve()): sha(args.targets_json),
              str(args.pricing_snapshot.resolve()): sha(args.pricing_snapshot)}
    bases = {}
    for video in dict.fromkeys(s["video_id"] for s in selection):
        desired = {s["frame_id"] for s in selection if s["video_id"] == video}
        available = {s.target_frame_id: s for s in adapter.iter_inference_video(video)
                     if s.target_frame_id in desired}
        if set(available) != desired:
            raise ValueError("frozen collection frame unavailable; no resampling")
        for selected in (s for s in selection if s["video_id"] == video):
            sample = available[selected["frame_id"]]
            if (sample.source_split not in (DatasetSplit.TRAINING, DatasetSplit.VALIDATION)
                    or sample.source_split is not DatasetSplit.parse(manifest["source_split"])
                    or selected["source_split"] != manifest["source_split"]
                    or list(sample.causal_frame_ids) != selected["causal_frame_ids"]):
                raise ValueError("frozen split or causal window differs from dataset")
            loaded = loader.load(sample)
            base = builder.build(context.build(loaded.runtime_sample, loaded.frames, workflow_snapshot={},
                memory_snapshot={}, prior_finalized_prediction=None))
            if base.payload["system_text"] != load_main_h0_prompt():
                raise ValueError("frozen H0 prompt drift")
            check_request_envelope(base)
            key = f"{video}_{selected['frame_id']}"
            bases[key] = base
            source_images = {str(p): sha(p) for p in sample.media_refs}
            if "images" in selected and any(source_images.get(i["path"]) != i["sha256"] for i in selected["images"]):
                raise ValueError("manifest source-image SHA differs from the dataset")
            selected.update(key=key, source_images=source_images,
                            request_metadata=canonical_request_metadata(base).to_mapping())
            hashes.update(source_images)
        masks_seen = set()
        for resolved in adapter.iter_video(video, frame_ids=sorted(desired)):
            frame = resolved.inference.target_frame_id
            selected = next(s for s in selection if s["video_id"] == video and s["frame_id"] == frame)
            if _mask_only(resolved) != selected["task_masks"]:
                raise ValueError("task availability changed after frozen selection")
            masks_seen.add(frame)
            for attr in ("annotation_source", "phase_source", "frame_action_source", "manifest_source"):
                path = getattr(resolved.provenance, attr)
                if path:
                    path = Path(path)
                    path = path if path.is_absolute() else args.dataset_root / path
                    if path.is_file():
                        hashes[str(path.resolve())] = sha(path)
        if masks_seen != desired:
            raise ValueError("task masks did not cover every selected target")
    endpoint, reservation = checked_pricing(read(args.pricing_snapshot))
    sources = _source_files()
    plan = {"schema_version": "verifier_candidate_collection_plan_v1", "model": MODEL,
            "source_split": manifest["source_split"], "selection": selection, "target_count": len(selection),
            "max_provider_calls": args.max_calls, "required_call_cap": 3 * len(selection),
            "retries": 0, "protocol": "frozen H0 -> blind locator -> proposal; no contrast or presence API call",
            "goal_id": args.goal_id, "goal_budget_usd": str(GOAL_CAP_USD),
            "development_budget_usd": str(DEVELOPMENT_CAP_USD),
            "budget_ledger": str(args.budget_ledger.resolve()), "reservation": reservation,
            "timeout_seconds": args.timeout_seconds, "confirmation": args.confirmation,
            "variants": list(args.variants),
            "pricing": endpoint["pricing"], "source_artifact_sha256": hashes,
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
            "gt_values_used_in_requests_or_selection": False}
    if save_plan(args.output, plan, sources):
        for key, request in bases.items():
            atomic_write_json(args.output / "requests" / f"{key}.json", {
                "metadata": canonical_request_metadata(request).to_mapping(), "payload": thaw_json(request.payload)})
    print(json.dumps({"status": "PREFLIGHT_PASSED", "mode": "collect", "targets": len(selection),
                      "max_calls": args.max_calls, "reservation": reservation, "provider_calls": 0}), flush=True)
    return adapter, plan, bases


class GoalBudgetStop(RuntimeError):
    """The shared goal cannot authorize another provider request."""


class GoalBudgetLedger:
    """Cross-process reservations, settlements, and unknown-cost liabilities.

    The $2.60 reservation covers the full endpoint context at its highest input
    rate plus the fixed output cap, using checked public pricing. It is not a
    provider-enforced price ceiling. Any over-reservation native cost halts the goal.
Crashes and unknown charges keep their reservations; nothing silently refunds.
"""

    def __init__(self, path, goal_id):
        self.path, self.goal_id = Path(path).resolve(), goal_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with self._locked():
            events = self._read_events()
            if not events:
                self._append(events, {"event": "INIT", "goal_id": goal_id,
                                      "budget_usd": str(GOAL_CAP_USD),
                                      "development_budget_usd": str(DEVELOPMENT_CAP_USD)})
            elif (events[0].get("goal_id") != goal_id
                  or Decimal(events[0]["budget_usd"]) != GOAL_CAP_USD
                  or Decimal(events[0].get("development_budget_usd", "0")) != DEVELOPMENT_CAP_USD):
                raise GoalBudgetStop("ledger belongs to another goal or budget")

    @contextmanager
    def _locked(self):
        with self.lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            deadline = time.monotonic() + 10
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt

                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise GoalBudgetStop("shared budget lock unavailable") from None
                    time.sleep(.05)
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_events(self):
        if not self.path.exists():
            return []
        events, previous = [], None
        for index, line in enumerate(self.path.read_text(encoding="utf-8").splitlines()):
            event = json.loads(line)
            digest = event.pop("sha256")
            if (event.get("sequence") != index or event.get("previous_sha256") != previous
                    or hashlib.sha256(canonical_json_bytes(event)).hexdigest() != digest):
                raise GoalBudgetStop("budget ledger hash chain is invalid")
            event["sha256"] = digest
            events.append(event)
            previous = digest
        return events

    def _append(self, events, fields):
        event = {"sequence": len(events), "at_utc": utc_now(),
                 "previous_sha256": events[-1]["sha256"] if events else None, **fields}
        event["sha256"] = hashlib.sha256(canonical_json_bytes(event)).hexdigest()
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        events.append(event)
        return event

    def _state(self, events):
        reservations, settlements, unknown, halts = {}, {}, set(), []
        for event in events:
            kind = event["event"]
            identity = event.get("reservation_id")
            if kind == "RESERVE":
                if identity in reservations:
                    raise GoalBudgetStop("duplicate reservation in ledger")
                reservations[identity] = event
            elif kind == "SETTLE":
                if identity not in reservations or identity in settlements:
                    raise GoalBudgetStop("invalid or duplicate settlement")
                settlements[identity] = event
            elif kind == "UNKNOWN":
                if identity not in reservations or identity in settlements:
                    raise GoalBudgetStop("unknown charge is not bound to an open reservation")
                unknown.add(identity)
            elif kind == "HALT":
                halts.append(event["reason"])
            elif kind != "INIT":
                raise GoalBudgetStop("unrecognized budget event")
        known = sum((Decimal(e["cost_usd"]) for e in settlements.values()), Decimal(0))
        held = sum((Decimal(e["reserve_usd"]) for key, e in reservations.items()
                    if key not in settlements), Decimal(0))
        occupied_by_phase = {phase: sum((
            Decimal(settlements[key]["cost_usd"]) if key in settlements else Decimal(event["reserve_usd"])
            for key, event in reservations.items() if event["phase"] == phase), Decimal(0))
            for phase in ("development", "validation")}
        return {"reservations": reservations, "settlements": settlements,
                "known_cost_usd": known, "held_reserve_usd": held,
                "occupied_usd": known + held, "available_usd": GOAL_CAP_USD - known - held,
                "unknown_open": sorted(unknown - set(settlements)), "halt_reasons": halts,
                "occupied_by_phase": occupied_by_phase}

    def snapshot(self):
        with self._locked():
            events = self._read_events()
            state = self._state(events)
            return {"goal_id": self.goal_id, "budget_usd": str(GOAL_CAP_USD),
                    "event_count": len(events), "ledger_sha256": sha(self.path),
                    **{key: str(state[key]) for key in (
                        "known_cost_usd", "held_reserve_usd", "occupied_usd", "available_usd")},
                    "open_reservations": sorted(set(state["reservations"]) - set(state["settlements"])),
                    "development_budget_usd": str(DEVELOPMENT_CAP_USD),
                    "occupied_by_phase": {key: str(value) for key, value in state["occupied_by_phase"].items()},
                    "unknown_open": state["unknown_open"], "halt_reasons": state["halt_reasons"]}

    def reserve(self, reservation_id, *, run_id, key, variant, phase="development"):
        if phase not in ("development", "validation"):
            raise GoalBudgetStop("unsupported budget phase")
        with self._locked():
            events = self._read_events()
            state = self._state(events)
            if state["halt_reasons"]:
                raise GoalBudgetStop("shared goal is halted")
            if reservation_id in state["reservations"]:
                raise GoalBudgetStop("reservation already exists; no redispatch")
            if state["occupied_usd"] + PER_CALL_RESERVE_USD > GOAL_CAP_USD:
                raise GoalBudgetStop("shared goal budget cannot reserve another call")
            if (phase == "development" and state["occupied_by_phase"][phase] + PER_CALL_RESERVE_USD
                    > DEVELOPMENT_CAP_USD):
                raise GoalBudgetStop("development budget cannot reserve another call")
            self._append(events, {"event": "RESERVE", "reservation_id": reservation_id,
                                  "run_id": run_id, "key": key, "variant": variant,
                                  "phase": phase,
                                  "reserve_usd": str(PER_CALL_RESERVE_USD)})

    def settle(self, reservation_id, cost, *, evidence_path, provider_request_id=None):
        amount = Decimal(str(cost))
        if not amount.is_finite() or amount < 0:
            raise GoalBudgetStop("native cost must be finite and nonnegative")
        with self._locked():
            events = self._read_events()
            state = self._state(events)
            if reservation_id not in state["reservations"] or reservation_id in state["settlements"]:
                raise GoalBudgetStop("settlement does not name an open reservation")
            self._append(events, {"event": "SETTLE", "reservation_id": reservation_id,
                                  "cost_usd": str(amount), "evidence_path": str(evidence_path),
                                  "evidence_sha256": sha(evidence_path),
                                  "provider_request_id": provider_request_id})
            reserve = Decimal(state["reservations"][reservation_id]["reserve_usd"])
            if amount > reserve:
                self._append(events, {"event": "HALT", "reason": "NATIVE_COST_EXCEEDED_RESERVATION"})

    def unknown(self, reservation_id, *, evidence_path, reason):
        with self._locked():
            events = self._read_events()
            state = self._state(events)
            if reservation_id not in state["reservations"] or reservation_id in state["settlements"]:
                raise GoalBudgetStop("unknown charge does not name an open reservation")
            self._append(events, {"event": "UNKNOWN", "reservation_id": reservation_id,
                                  "reason": reason, "evidence_path": str(evidence_path),
                                  "evidence_sha256": sha(evidence_path)})

    def halt(self, reason):
        with self._locked():
            events = self._read_events()
            self._append(events, {"event": "HALT", "reason": reason})


def check_request_envelope(request):
    """Limit the local reservation envelope; never claim a provider hard cap."""
    import io

    from PIL import Image

    text_bytes = len(request.payload["system_text"].encode()) + len(request.payload["input_text"].encode())
    if text_bytes > 40000 or not 1 <= len(request.images) <= 6:
        raise ValueError("request exceeds reserved text/image-count envelope")
    if sum(len(image.content) for image in request.images) > 12 * 1024 * 1024:
        raise ValueError("request exceeds reserved image-byte envelope")
    for image in request.images:
        with Image.open(io.BytesIO(image.content)) as decoded:
            if max(decoded.size) > 2048 or decoded.width * decoded.height > 2048 * 2048:
                raise ValueError("request exceeds reserved image-pixel envelope")
    wire_body(request)  # strict model, route, schema and 4096-output-token cap


class BudgetedCalls(DirectCalls):
    def __init__(self, output, cap, secret, ledger, *, phase="development", timeout=180, sender=None):
        super().__init__(output, cap, secret, timeout=timeout, sender=sender)
        self.ledger = ledger
        self.phase = phase
        self.run_id = str(Path(output).resolve())

    def call(self, key, stage, request):
        if self.stopped or self.used >= self.cap:
            return None
        check_request_envelope(request)
        reservation_id = hashlib.sha256(canonical_json_bytes({
            "run_id": self.run_id, "key": key, "stage": stage,
            "request_hash": canonical_request_metadata(request).request_hash})).hexdigest()
        try:
            self.ledger.reserve(reservation_id, run_id=self.run_id, key=key, variant=stage, phase=self.phase)
        except GoalBudgetStop:
            self.stopped = "SHARED_GOAL_BUDGET_STOP"
            return None
        payload = super().call(key, stage, request)
        record = self.records[-1]
        evidence = self.output / "calls" / key / stage / "result.json"
        if record["cost_usd"] is None:
            self.ledger.unknown(reservation_id, evidence_path=evidence,
                                reason=record.get("error", "MISSING_NATIVE_COST"))
        else:
            self.ledger.settle(reservation_id, record["cost_usd"], evidence_path=evidence,
                               provider_request_id=record.get("provider_request_id"))
        if record.get("error") == "model_or_provider_identity_mismatch":
            self.stopped = "MODEL_OR_PROVIDER_IDENTITY_MISMATCH"
        if self.stopped:
            self.ledger.halt(self.stopped)
        if self.ledger.snapshot()["halt_reasons"]:
            self.stopped = self.stopped or "SHARED_GOAL_HALTED"
        return payload


def restore_request(saved, images):
    metadata = saved["metadata"]
    request = ApiRequest(provider=metadata["provider"],
        model_identifier=metadata["requested_model_identifier"],
        endpoint_identifier=metadata["endpoint_identifier"],
        prompt_version=metadata["prompt_version"], response_schema_version=metadata["response_schema_version"],
        generation_parameters=metadata["generation_parameters"], payload=saved["payload"], images=tuple(images))
    if canonical_request_metadata(request).to_mapping() != metadata:
        raise ValueError("frozen request/image canonical binding mismatch")
    wire_body(request)
    return request


def _stage(source, key, stage, images, expected, expected_hash, hashes):
    directory = source / "calls" / key / stage
    for name in ("request.json", "result.json", "http_response.json"):
        hashes[str(directory / name)] = sha(directory / name)
    request = restore_request(read(directory / "request.json"), images)
    result, http = read(directory / "result.json"), read(directory / "http_response.json")
    native = json.loads(http["body"])
    if (result["status"] != "OK" or result.get("payload") != expected
            or result["request_hash"] != expected_hash
            or canonical_request_metadata(request).request_hash != expected_hash
            or http["status_code"] != 200 or native.get("model") != MODEL
            or native.get("provider") != "Alibaba" or not native.get("id")
            or native.get("id") != result.get("provider_request_id")
            or json.loads(native["choices"][0]["message"]["content"]) != expected):
        raise ValueError("frozen stage response is not bound to its request")
    validator_for(request.response_schema_version)(expected)
    return request


def restore_target(source, selected, row, hashes):
    key = selected["key"]
    original = source / "requests" / f"{key}.json"
    hashes[str(original)] = sha(original)
    saved = read(original)
    images = _restore_images(selected, saved["metadata"], hashes)
    base = restore_request(saved, images)
    if row["h0"] is None:
        if row.get("h1") is not None or row.get("h0_payload") is not None:
            raise ValueError("failed H0 cannot contain a valid candidate or hidden payload")
        failure = source / "calls" / key / "h0" / "result.json"
        if failure.exists():
            hashes[str(failure)] = sha(failure)
            if read(failure)["status"] == "OK":
                raise ValueError("failed H0 row contradicts a successful source response")
        return base, None
    _stage(source, key, "h0", images, row["h0_payload"], row["request_hashes"]["h0"], hashes)
    if final_labels(row["h0_payload"]) != row["h0"]:
        raise ValueError("saved H0 differs from its original response")
    if row.get("h1") is None:
        return base, None
    _stage(source, key, "locator", images, row["locator"], row["request_hashes"]["locator"], hashes)
    crops, manifest = make_contact_crops(images[-1], row["locator"], target_frame_id=row["frame_id"])
    if manifest != row["crop_manifest"]:
        raise ValueError("reconstructed crop geometry or SHA mismatch")
    request = _stage(source, key, "proposal", images + crops, row["proposal"],
                      row["request_hashes"]["proposal"], hashes)
    body = json.loads(request.payload["input_text"])
    if body["crop_manifest"] != manifest or body["predicted_instance_regions"] != row["locator"]:
        raise ValueError("candidate did not receive the recorded locator/crops")
    prepared = prepare_grounded_review(row["h0"], row["locator"], row["proposal"],
                                        proposal_slot=row["proposal_slot"])
    if prepared["h1"] != row["h1"]:
        raise ValueError("saved candidate cannot be derived from original proposal")
    if not build_change_claims(row["h0"], row["h1"]):
        return base, None
    return base, build_review_request(base, row, schema_version=PRESENCE_REVIEW_1000_VERSION)


def variant_request(request, variant):
    if variant not in VARIANTS:
        raise ValueError("unsupported frozen verifier variant")
    if variant == "numeric1000":
        return request
    payload = thaw_json(request.payload)
    body = json.loads(payload["input_text"])
    for proposition in body["propositions"]:
        task, label = proposition["task"], proposition["label_id"]
        if task == "ivt":
            _, instrument, verb, target = _ivt_rows()[label]
            name = {"instrument": _TASK_NAMES["instrument"][instrument],
                    "verb": _TASK_NAMES["verb"][verb], "target": _TASK_NAMES["target"][target]}
            proposition["decoded_components"] = name
        else:
            proposition["decoded_label_name"] = _TASK_NAMES[task][label]
    payload["input_text"] = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return replace(request, payload=payload, prompt_version=request.prompt_version + "_decoded_names_v1")


def prepare(args):
    if not 1 <= args.max_calls <= 72:
        raise ValueError("max_calls must be 1..72")
    if not 1 <= args.timeout_seconds <= 600:
        raise ValueError("timeout must be 1..600 seconds")
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        raise ValueError("new trial output must be separate from frozen source")
    if (output / "execution.lock").exists():
        raise ValueError("run already dispatched; never repeat")
    source_plan, source_rows = read(source / "plan.json"), read(source / "predictions.json")
    completion = read(source / "summary.json")
    unsigned_plan = {key: value for key, value in source_plan.items() if key != "plan_sha256"}
    if (hashlib.sha256(canonical_json_bytes(unsigned_plan)).hexdigest() != source_plan["plan_sha256"]
            or completion.get("plan_sha256") != source_plan["plan_sha256"]
            or completion.get("predictions_sha256") != sha(source / "predictions.json")):
        raise ValueError("source collection is incomplete or its completion hashes do not match")
    if source_plan.get("model") != MODEL:
        raise ValueError("source H0 model mismatch")
    selected = {s["key"]: s for s in source_plan["selection"]}
    rows_by_key = {f"{r['video_id']}_{r['frame_id']}": r for r in source_rows}
    keys = read(args.targets_json)["keys"]
    if not keys or len(keys) != len(set(keys)) or set(keys) - set(selected) or len(keys) > 24:
        raise ValueError("one to 24 distinct frozen source keys required")
    variants = list(args.variants)
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
        raise ValueError("distinct supported variants required")
    require_split(source_plan["source_split"], confirmation=args.confirmation, variants=variants)
    if source_plan["source_split"] == "Validation" and source_plan.get("variants") != variants:
        raise ValueError("Validation variant must be frozen before collecting any predictions")
    hashes = {str(source / n): sha(source / n) for n in ("plan.json", "predictions.json", "summary.json")}
    hashes[str(args.targets_json.resolve())] = sha(args.targets_json)
    hashes[str(args.pricing_snapshot.resolve())] = sha(args.pricing_snapshot)
    endpoint, reservation = checked_pricing(read(args.pricing_snapshot))
    requests_by_key, frozen_rows, selection = {}, [], []
    for index, key in enumerate(keys):
        item, row = selected[key], rows_by_key[key]
        if item["source_split"] != source_plan["source_split"]:
            raise ValueError("target split differs from declared source split")
        _, rebuilt = restore_target(source, item, row, hashes)
        frozen_rows.append({"video_id": row["video_id"], "frame_id": row["frame_id"],
                            "h0": row["h0"], "h1": row["h1"], "variants": {}})
        if rebuilt is None:
            continue
        request, evidence, full_ref = rebuilt
        order = variants if index % 2 == 0 else list(reversed(variants))
        for variant in order:
            changed = variant_request(request, variant)
            check_request_envelope(changed)
            requests_by_key[(key, variant)] = changed
            selection.append({"key": key, "variant": variant, "evidence_manifest": evidence,
                              "full_frame_ref": full_ref,
                              "request_metadata": canonical_request_metadata(changed).to_mapping()})
    if len(selection) > args.max_calls:
        raise ValueError("paired review count exceeds this batch call cap")
    source_files = _source_files()
    plan = {"schema_version": "verifier_variant_trial_plan_v1", "source": str(source),
            "source_split": source_plan["source_split"], "confirmation": args.confirmation,
            "model": MODEL, "variants": variants, "keys": keys, "selection": selection,
            "target_count": len(keys), "review_call_count": len(selection),
            "max_provider_calls": args.max_calls, "retries": 0,
            "goal_id": args.goal_id, "goal_budget_usd": str(GOAL_CAP_USD),
            "development_budget_usd": str(DEVELOPMENT_CAP_USD),
            "per_call_reserve_usd": str(PER_CALL_RESERVE_USD),
            "reservation": reservation, "timeout_seconds": args.timeout_seconds,
            "budget_ledger": str(args.budget_ledger.resolve()),
            "reservation_is_provider_enforced": False,
            "unknown_cost_policy": "retain full reservation; native-cost evidence may settle later",
            "pricing": endpoint["pricing"], "source_artifact_sha256": hashes,
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_files},
            "gt_values_used_in_requests_or_selection": False,
            "interpretation": "Training development" if not args.confirmation else "locked Validation confirmation"}
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    if (output / "plan.json").exists():
        if read(output / "plan.json") != plan:
            raise ValueError("preflight changed; use a fresh directory")
    elif output.exists():
        raise ValueError("fresh output directory required")
    else:
        output.mkdir(parents=True)
        atomic_write_json(output / "plan.json", plan)
        atomic_write_json(output / "frozen_predictions.json", frozen_rows)
        for selected_call in selection:
            key, variant = selected_call["key"], selected_call["variant"]
            request = requests_by_key[(key, variant)]
            atomic_write_json(output / "requests" / key / f"{variant}.json", {
                "metadata": canonical_request_metadata(request).to_mapping(),
                "payload": thaw_json(request.payload), "wire": safe_wire(wire_body(request))})
        for path in source_files:
            destination = output / "frozen_source" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    print(json.dumps({"status": "PREFLIGHT_PASSED", "targets": len(keys), "review_calls": len(selection),
                      "provider_calls": 0, "goal_budget_usd": str(GOAL_CAP_USD)}), flush=True)
    return plan, requests_by_key, frozen_rows


def assert_frozen(plan):
    for name, digest in plan["source_artifact_sha256"].items():
        if sha(name) != digest:
            raise ValueError("source artifact changed after preflight")
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("source code changed after preflight")


def score_saved_rows(args, rows, variants):
    # Import only here: all inference is already closed and predictions durable.
    from scripts.run_grounded_api_pipeline import score_saved

    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    _, targets = score_saved(adapter, [{**r, "final": r["h0"]} for r in rows])
    truth = {(r["video_id"], r["frame_id"]): r for r in targets}
    scored = [{**r, "gt": truth[(r["video_id"], r["frame_id"])]["gt"],
               "mask": truth[(r["video_id"], r["frame_id"])]["mask"]} for r in rows]
    comparisons = {variant: {arm: compute_repair_comparison([
        {**r, "final": r["variants"][variant][arm]} for r in scored])
        for arm in ("final_a", "final_b")} for variant in variants}
    return scored, comparisons


def start_execution(args, plan):
    """One public price recheck, then the shared budget for every paid stage."""
    assert_frozen(plan)
    response = requests.get(PRICING_URL, timeout=(10, 30), allow_redirects=False)
    response.raise_for_status()
    native = response.json()
    endpoint, reservation = checked_pricing(native)
    for name, old_rate in plan["pricing"].items():
        if Decimal(str(endpoint["pricing"].get(name, "0"))) > Decimal(str(old_rate)):
            raise ValueError("provider price increased after the frozen preflight")
    atomic_write_json(args.output / "pricing_execution.json", {
        "fetched_at_utc": utc_now(), "source": PRICING_URL, "response": native,
        "reservation": reservation})
    secret = resolve_api_key(api_key=None, api_key_file=args.api_key_file)
    ledger = GoalBudgetLedger(args.budget_ledger, args.goal_id)
    with (args.output / "execution.lock").open("x", encoding="utf-8") as marker:
        marker.write(plan["plan_sha256"] + "\nBUDGETED_PAID_NO_RETRIES\n")
    phase = "development" if plan["source_split"] == "Training" else "validation"
    calls = BudgetedCalls(args.output, args.max_calls, secret, ledger,
                          phase=phase, timeout=args.timeout_seconds)
    return secret, ledger, calls


def collect_target(base, calls, key):
    # The existing checked pipeline exposes review through its callback. Returning
    # None there skips that optional HTTP request entirely; only these three
    # stages may reach the one shared budgeted network boundary.
    def call(stage, request):
        return None if stage == "review" else calls.call(key, stage, request)
    result = run_grounded_target(base, call, proposal_slot="FIRST")
    result["review_execution"] = "SKIPPED_COLLECTION_ONLY_NO_API_CALL"
    return result


def run_collect(args):
    if not 1 <= args.max_calls <= 72 or not 1 <= args.timeout_seconds <= 600:
        raise ValueError("collection requires max_calls 1..72 and timeout 1..600")
    _adapter, plan, bases = prepare_collect(args)
    if not args.execute:
        return plan
    if args.api_key_file is None:
        raise ValueError("--execute requires an external key file")
    secret, ledger, calls = start_execution(args, plan)
    rows = [{"video_id": s["video_id"], "frame_id": s["frame_id"], "status": "NOT_ATTEMPTED",
             "h0": None, "h0_payload": None, "h1": None, "final": None,
             "locator": None, "proposal": None, "request_hashes": {}, "crop_manifest": []}
            for s in plan["selection"]]
    atomic_write_json(args.output / "predictions.json", rows)
    for index, selected in enumerate(plan["selection"]):
        if calls.stopped:
            break
        rows[index] = collect_target(bases[selected["key"]], calls, selected["key"])
        atomic_write_json(args.output / "predictions.json", rows)
    calls.stopped = calls.stopped or "INFERENCE_FINISHED"
    predictions_hash = sha(args.output / "predictions.json")
    assert_frozen(plan)
    summary = {"schema_version": "verifier_candidate_collection_result_v1", "model": MODEL,
               "source_split": plan["source_split"], "targets": len(rows),
               "h0_successes": sum(r["h0"] is not None for r in rows),
               "candidate_available": sum(r["h1"] is not None for r in rows),
               "stop_reason": calls.stopped, "plan_sha256": plan["plan_sha256"],
               "predictions_sha256": predictions_hash, "gt_scoring_performed": False,
               "gt_policy": "wait until every preselected verifier output is persisted",
               "goal_budget": ledger.snapshot(), **accounting(calls.records)}
    if sha(args.output / "predictions.json") != predictions_hash:
        raise ValueError("predictions changed during scoring")
    atomic_write_json(args.output / "summary.json", summary)
    assert_secret_absent(secret, (p for p in args.output.rglob("*") if p.is_file()))
    print(json.dumps(summary), flush=True)
    return summary


def run(args):
    if args.mode == "collect":
        return run_collect(args)
    if args.source is None:
        raise ValueError("replay mode requires --source")
    plan, requests_by_key, rows = prepare(args)
    if not args.execute:
        return plan
    if args.api_key_file is None:
        raise ValueError("--execute requires an external key file")
    secret, ledger, calls = start_execution(args, plan)
    requested = {(s["key"], s["variant"]) for s in plan["selection"]}
    for row in rows:
        key = f"{row['video_id']}_{row['frame_id']}"
        for variant in plan["variants"]:
            row["variants"][variant] = {"status": "NOT_ATTEMPTED" if (key, variant) in requested else "NO_CHANGED_CANDIDATE",
                "final_a": deepcopy(row["h0"]), "final_b": deepcopy(row["h0"]),
                "decision_a": "KEEP", "decision_b": "KEEP", "review": None}
    atomic_write_json(args.output / "predictions.json", rows)
    by_key = {f"{r['video_id']}_{r['frame_id']}": r for r in rows}
    for selected in plan["selection"]:
        if calls.stopped:
            break
        key, variant = selected["key"], selected["variant"]
        row = by_key[key]
        review = calls.call(key, variant, requests_by_key[(key, variant)])
        result = evaluate_presence_review(row["h0"], row["h1"], review,
            allowed_evidence_refs=[e["ref"] for e in selected["evidence_manifest"]],
            full_frame_ref=selected["full_frame_ref"], schema_version=PRESENCE_REVIEW_1000_VERSION)
        row["variants"][variant] = {**result, "review": review,
            "status": "OK" if review is not None and result["reason_a"] != "INVALID_REVIEW" else "REVIEW_FAILURE_KEEP"}
        atomic_write_json(args.output / "predictions.json", rows)
    calls.stopped = calls.stopped or "INFERENCE_FINISHED"
    predictions_hash = sha(args.output / "predictions.json")
    assert_frozen(plan)
    scored, comparisons = score_saved_rows(args, rows, plan["variants"])
    atomic_write_json(args.output / "scored_predictions.json", scored)
    summary = {"schema_version": "verifier_variant_trial_result_v1", "model": MODEL,
               "source_split": plan["source_split"], "targets": len(rows),
               "variants": plan["variants"], "stop_reason": calls.stopped,
               "plan_sha256": plan["plan_sha256"], "predictions_sha256": predictions_hash,
               "comparisons": comparisons, "goal_budget": ledger.snapshot(), **accounting(calls.records)}
    if sha(args.output / "predictions.json") != predictions_hash:
        raise ValueError("predictions changed during scoring")
    atomic_write_json(args.output / "summary.json", summary)
    assert_secret_absent(secret, (p for p in args.output.rglob("*") if p.is_file()))
    print(json.dumps({k: v for k, v in summary.items() if k != "comparisons"}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("collect", "replay"), default="replay")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--targets-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pricing-snapshot", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument("--goal-id", required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--max-calls", type=int, default=24)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())
