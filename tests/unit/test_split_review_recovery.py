from decimal import Decimal

from scripts import recover_split_review_trial as recovery


def test_proxy_failure_stops_subsequent_dispatch_without_retry(tmp_path, monkeypatch):
    recovery.trial.save(tmp_path / "cached_ledger.json", [])
    recovery.trial.save(tmp_path / "budget.json", {"occupied": {a: "0" for a in recovery.trial.LIMITS}})
    monkeypatch.setattr(recovery, "SOURCE", tmp_path)
    calls = recovery.RecoveryCalls(tmp_path)
    def fail_once(self, target, stage, seat, body):
        self.rows.append({"target": target, "stage": stage, "seat": seat, "exception_type": "ProxyError"})
    monkeypatch.setattr(recovery.RecoveryCalls.__mro__[1], "call", fail_once)
    assert calls.call("fixture", "components", "qwen", {}) is None
    assert calls.stopped
    assert len(calls.rows) == 1
    assert (tmp_path / "network_circuit_breaker.json").exists()


def test_recovery_retains_original_unknown_reserves_in_cumulative_limit(tmp_path, monkeypatch):
    recovery.trial.save(tmp_path / "cached_ledger.json", [])
    recovery.trial.save(tmp_path / "budget.json", {"occupied": {a: "2.75" for a in recovery.trial.LIMITS}})
    monkeypatch.setattr(recovery, "SOURCE", tmp_path)
    calls = recovery.RecoveryCalls(tmp_path)
    assert all(value == Decimal("0.25") for value in calls.limits.values())
