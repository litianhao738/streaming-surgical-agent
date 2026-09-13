"""Tracker review-evidence paired pilot, run 3 (docs/PGP_TRACKER_REVIEW_EVIDENCE_PROTOCOL_R3_2026-09-14.md).

Same frozen 64-frame paired design, prompts, caps and scoring as scripts/assess_tracker_review_evidence.py
(r2). One pre-declared amendment: on the official GLM/DeepSeek/Qwen route, a provider content refusal
(Aliyun `data_inspection_failed`, GLM `1301`) is recorded as API_FAILED and treated as an invalid seat, as
the production first-attempt fallback does; it no longer stops the whole batch. Every other global-stop
condition (other 4xx, OpenRouter 4xx, identity mismatch, budget) is unchanged. No retries, no replacement.
Only `collect --allow-paid` sends requests.
"""
import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import assess_tracker_review_evidence as base
from scripts import full_official_reviewer_transport as T
from scripts.train_pgp_gate_no_tracker import sha, write
from surgical_agent.research.gate.collection_budget import (
    AmbiguousDispatch,
    Budget,
    BudgetStop,
)

DEFAULT = ROOT / "artifacts/training/gate/tracker_review_evidence_20260914_r3"
PROTOCOL = ROOT / "docs/PGP_TRACKER_REVIEW_EVIDENCE_PROTOCOL_R3_2026-09-14.md"
REFUSAL_CODES = frozenset(("data_inspection_failed", "1301"))


class RefusalTolerantCalls(T.Calls):
    """Line-for-line copy of Calls.call; only the official-route refusal predicate differs."""

    def call(self, target, stage, seat, body):
        if target != self.s["key"]:
            raise RuntimeError("target identity mismatch")
        identity = (target, stage, seat)
        with self.lock:
            if identity in self.seen:
                raise RuntimeError("duplicate request in target")
            self.seen.add(identity)
        if seat not in T.CHANGED:
            found, result = self.reuse(target, stage, seat, body)
            if found:
                return result
            with self.lock:
                if self.delegate is None:
                    if self.replay_only:
                        self.delegate = T.old.trial.Replay(self.folder / "run", "gemini")
                    else:
                        self.delegate = T.old.ResumeCalls(self.folder / "run", self.plan, self.budget, self.stop)
            if self.replay_only and identity not in self.delegate.rows:
                return None
            return self.delegate.call(target, stage, seat, body)
        wire = T.routes.route_body(seat, body, self.plan["reviewer_config"])
        if seat == "qwen":
            found, result = self.reuse(target, stage, seat, wire)
            if found:
                return result
        safe = T.old.old.joint.roster.transport.redact_images(wire)
        folder = self.folder / "changed" / f"{stage}_{seat}"
        if (folder / "record.json").exists():
            record = T.old.old.read(folder / "record.json")
            if T.old.old.read(folder / "request.json") != safe:
                raise RuntimeError("resumed changed request differs")
            if record["status"] == "DISPATCHED" or "finished_utc" not in record:
                raise AmbiguousDispatch("changed request has uncertain outcome: " + str(identity))
            if record.get("response_sha256") and T.old.digest(folder / "response.json") != record["response_sha256"]:
                raise RuntimeError("changed response drift")
            self.rows.append(record)
            if record["status"] == "JSON_PARSED":
                raw = T.old.old.read(folder / "response.json")
                T.routes.check_response(seat, raw, self.plan["reviewer_config"])
                return T.parse_changed(raw, wire)
            return None
        if self.replay_only:
            raise RuntimeError("replay missing changed dispatch")
        if self.stop.is_set():
            raise BudgetStop("global stop before dispatch")
        url, key = T.routes.credentials(seat, self.plan["reviewer_config"])
        if url != self.plan["endpoints"][seat]:
            raise RuntimeError("endpoint drift")
        account = {"grok": "glm_requests", "deepseek": "deepseek_requests", "qwen": "aliyun_cny"}[seat]
        reserve = (T.old.old.joint.roster.transport.envelope(seat, wire, self.plan["call_rates"])
                   if seat == "qwen" else Decimal(1))
        key_id = Budget.key(*identity)
        try:
            self.budget.reserve(key_id, account, reserve)
        except BudgetStop:
            self.stop.set()
            raise
        record = {"target": target, "stage": stage, "seat": seat, "model": wire["model"], "endpoint": url,
                  "account": account, "reserve": str(reserve), "charge": str(reserve),
                  "charge_kind": "unknown_reserved" if seat == "qwen" else "request_count_not_money",
                  "status": "DISPATCHED", "started_utc": T.old.old.now()}
        T.old.write(folder / "request.json", safe)
        T.old.write(folder / "record.json", record)
        parsed = None
        try:
            response = requests.post(url.rstrip("/") + "/chat/completions", json=wire,
                                     headers={"Authorization": "Bearer " + key}, timeout=(15, 120),
                                     allow_redirects=False)
            try:
                raw = response.json()
            except ValueError:
                raw = {"non_json_response": response.text}
            raw = json.loads(json.dumps(raw).replace(key, "[REDACTED]"))
            T.old.write(folder / "response.json", raw)
            usage = raw.get("usage") or {}
            record.update(http_status=response.status_code, usage=usage,
                          response_sha256=T.old.digest(folder / "response.json"), returned_model=raw.get("model"))
            if seat == "qwen":
                charge, kind = T.qwen_charge(usage, reserve)
                record.update(charge=str(charge), charge_kind=kind)
            if response.ok and not raw.get("error"):
                parsed = T.parse_changed(raw, wire)
                T.routes.check_response(seat, raw, self.plan["reviewer_config"])
                record["status"] = "JSON_PARSED"
            else:
                record.update(status="API_FAILED", provider_error=raw.get("error"))
                err = raw.get("error", {})
                # r3 amendment: GLM `1301` joins Aliyun `data_inspection_failed` as a content refusal.
                refusal = isinstance(err, dict) and str(err.get("code")) in REFUSAL_CODES
                if refusal:
                    record["content_refusal_invalid_seat"] = True
                if response.status_code in (400, 401, 402, 403, 404) and not refusal:
                    self.stop.set()
                    record["global_stop"] = True
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError, RuntimeError) as exc:
            record.update(status="FAILED", error_type=type(exc).__name__, error=str(exc).replace(key, "[REDACTED]")[:200])
            if "identity mismatch" in str(exc) or "disabled policy" in str(exc):
                self.stop.set()
                record["global_stop"] = True
        finally:
            record["finished_utc"] = T.old.old.now()
            T.old.write(folder / "record.json", record)
            self.budget.settle(key_id, Decimal(record["charge"]))
            with self.lock:
                self.rows.append(record)
        return parsed


def prepare(out):
    base.prepare(out)
    plan = base.read(out / "plan.json")
    plan["amendment"] = {"run": "r3", "protocol": str(PROTOCOL),
                         "content_refusal_codes_treated_as_invalid_seat": sorted(REFUSAL_CODES),
                         "unchanged": "sampling, arms, prompts, models, caps, primary checks, all other global stops"}
    for p in (PROTOCOL, Path(__file__).resolve()):
        plan["source_sha256"][str(p)] = sha(p)
    # base.prepare just created these two files in this process; `write` refuses to overwrite.
    (out / "plan.json").unlink()
    write(out / "plan.json", plan)
    pre = base.read(out / "preflight.json")
    pre.update(plan_sha256=sha(out / "plan.json"), amendment=plan["amendment"])
    (out / "preflight.json").unlink()
    write(out / "preflight.json", pre)
    print(json.dumps({"plan_sha256": pre["plan_sha256"], "amendment": plan["amendment"]}), flush=True)


def collect(out, limit, workers, allow_paid):
    base.Calls = RefusalTolerantCalls
    base.collect(out, limit, workers, allow_paid)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "collect", "score"])
    p.add_argument("--output", type=Path, default=DEFAULT)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--allow-paid", action="store_true")
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.output)
    elif a.command == "collect":
        collect(a.output, a.limit, a.workers, a.allow_paid)
    else:
        base.score(a.output)
