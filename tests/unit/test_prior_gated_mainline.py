"""No-provider regression tests for the release runner and concurrent ledger."""
import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from threading import RLock, get_ident

import pytest

from scripts import run_prior_gated_joint_mainline as mainline
from surgical_agent.research.verification import serialized_ledger


class Ledger(serialized_ledger.SerializedLedgerMixin):
    def __init__(self, output):
        self.output, self.lock, self.rows = output, RLock(), []
        self.limits = {"usd": Decimal(10)}
        self.carried = {"usd": "0"}
        self.occupied = {"usd": Decimal(0)}
        self.stopped = False


def test_many_concurrent_updates_have_one_writer_and_no_lost_rows(tmp_path, monkeypatch):
    writer_threads = set()
    original = serialized_ledger.atomic_write_json

    def write(path, snapshot):
        writer_threads.add(get_ident())
        original(path, snapshot)

    monkeypatch.setattr(serialized_ledger, "atomic_write_json", write)
    ledger = Ledger(tmp_path)

    def update(index):
        with ledger.lock:
            ledger.rows.append({"index": index, "status": "COMPLETE"})
            ledger.occupied["usd"] += Decimal(".01")
            ledger.persist()
        ledger.persist()  # External persist is also serialized with mutations.

    with ThreadPoolExecutor(max_workers=12) as workers:
        list(workers.map(update, range(100)))
    ledger.close_ledger()
    result = json.loads((tmp_path / "budget.json").read_text())
    assert len(result["calls"]) == 100
    assert {r["index"] for r in result["calls"]} == set(range(100))
    assert result["occupied"]["usd"] == "1.00"
    assert len(writer_threads) == 1
    assert not ledger._ledger_thread.is_alive()


def test_writer_error_is_propagated_without_retry(tmp_path, monkeypatch):
    calls = []

    def fail(*args):
        calls.append(1)
        raise PermissionError("simulated writer failure")

    monkeypatch.setattr(serialized_ledger, "atomic_write_json", fail)
    ledger = Ledger(tmp_path)
    with pytest.raises(PermissionError):
        ledger.persist()
    ledger.close_ledger()
    assert len(calls) == 1


def test_frozen_mainline_has_no_blind_phase_stage():
    assert mainline.CALLS_PER_TARGET == 13
    assert "control_phase" not in mainline.STAGES
    assert mainline.ARMS == ("h0", "gated_control", "gated_control_jointphase")


def test_failed_target_stays_in_metric_denominator():
    from scripts import prior_gated_mainline_release as release

    correct = {t: [0] for t in mainline.old.TASKS}
    rows = [{"key": "ok", "predictions": {a: correct for a in mainline.ARMS}},
            {"key": "failed", "predictions": {a: None for a in mainline.ARMS}}]
    truth = {r["key"]: {"mask": dict.fromkeys(mainline.old.TASKS, True), "gt": correct} for r in rows}
    result = release.metrics(rows, truth)
    assert result[mainline.PRIMARY]["ivt"]["tp"] == 1
    assert result[mainline.PRIMARY]["ivt"]["fn"] == 1
    assert result[mainline.PRIMARY]["errors"] == 5
    assert result[mainline.PRIMARY]["mean_f1"] == pytest.approx(200 / 3)
