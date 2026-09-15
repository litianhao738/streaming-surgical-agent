"""Durable single-dispatch cache with separate generation/evaluation accounting."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
from urllib.request import Request, urlopen

from .contracts import digest


class CachedCalls:
    def __init__(self, path, caller=None, *, max_calls=0, mock=False):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, stage TEXT, request TEXT, status TEXT, response TEXT, seconds REAL, mock INTEGER)")
        self.db.commit()
        self.caller, self.max_calls, self.dispatched, self.mock = caller, max_calls, 0, mock
        self.hits = {"report_generation": 0, "offline_evaluation": 0}
        self.session_dispatches = dict.fromkeys(self.hits, 0)

    def call(self, stage, request):
        if stage not in self.hits:
            raise ValueError("invalid cost stage")
        key = digest({"stage": stage, "request": request, "mock": self.mock})
        row = self.db.execute("SELECT status,response FROM calls WHERE key=?", (key,)).fetchone()
        if row:
            if row[0] != "COMPLETE":
                raise RuntimeError("previous dispatch unresolved/failed; no automatic retry")
            self.hits[stage] += 1
            return json.loads(row[1]), key, True
        if self.caller is None or self.dispatched >= self.max_calls:
            raise RuntimeError("uncached call not authorized or request cap reached")
        self.db.execute("INSERT INTO calls VALUES (?,?,?,'PENDING',NULL,NULL,?)", (key, stage, json.dumps(request), int(self.mock)))
        self.db.commit()
        self.dispatched += 1
        self.session_dispatches[stage] += 1
        tick = time.perf_counter()
        try:
            response = self.caller(request)
            if not isinstance(response, dict) or not isinstance(response.get("text"), str):
                raise ValueError("transport must return text and optional usage/cost")
            encoded = json.dumps(response, allow_nan=False)
            self.db.execute("UPDATE calls SET status='COMPLETE',response=?,seconds=? WHERE key=?", (encoded, time.perf_counter()-tick, key))
            self.db.commit()
            return response, key, False
        except Exception:
            self.db.execute("UPDATE calls SET status='FAILED',seconds=? WHERE key=?", (time.perf_counter()-tick, key))
            self.db.commit()
            raise

    def costs(self):
        out = {}
        for stage in self.hits:
            rows = self.db.execute("SELECT status,response,seconds,mock FROM calls WHERE stage=?", (stage,)).fetchall()
            amounts, usage, unknown = {}, {}, 0
            for status, raw, _, mock in rows:
                response = json.loads(raw) if raw else {}
                cost = response.get("cost")
                if not mock and not cost:
                    unknown += 1
                if isinstance(cost, dict):
                    for currency, value in cost.items():
                        amounts[currency] = amounts.get(currency, 0) + value
                for name, value in response.get("usage", {}).items():
                    if type(value) in (int, float):
                        usage[name] = usage.get(name, 0) + value
            out[stage] = {"ledger_dispatches": len(rows), "mock_dispatches": sum(r[3] for r in rows),
                          "session_dispatches": self.session_dispatches[stage],
                          "paid_dispatch_attempts": sum(not r[3] for r in rows), "cache_hits_this_session": self.hits[stage],
                          "unpriced_dispatches": unknown, "reported_cost_by_currency": amounts,
                          "usage": usage, "seconds": sum(r[2] or 0 for r in rows)}
        return out

    def close(self):
        self.db.close()


class OpenAICompatibleCaller:
    """Explicit endpoint/key adapter; never auto-selects credentials or retries."""
    def __init__(self, api_key, *, timeout=90):
        self.api_key, self.timeout = api_key, timeout

    def __call__(self, request):
        c = request["model_config"]
        if not c["endpoint"].startswith("https://"):
            raise ValueError("explicit HTTPS chat-completions endpoint required")
        payload = {"model": c["model"], "temperature": c["temperature"], "max_tokens": c["max_tokens"],
                   "messages": [{"role": "user", "content": request["prompt"]}]}
        policy = request.get("wire_policy", {})
        if set(policy) - {"reasoning", "response_format", "stream"}:
            raise ValueError("unsupported wire policy")
        payload.update(policy)
        if request.get("omit_temperature"):
            payload.pop("temperature")
        if request.get("provider_tag"):
            tag = request["provider_tag"]
            payload["provider"] = {"only": [tag], "order": [tag], "allow_fallbacks": False, "require_parameters": True}
        req = Request(c["endpoint"], data=json.dumps(payload).encode(),
                      headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"})
        with urlopen(req, timeout=self.timeout) as response:
            obj = json.load(response)
        return {"text": obj["choices"][0]["message"]["content"], "usage": obj.get("usage", {}),
                "provider_request_id": obj.get("id"), "cost": None}
