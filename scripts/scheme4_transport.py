"""Reuse durable official routes, with narrow refusal and unknown-outcome checks."""
from scripts.assess_tracker_review_evidence_r3 import RefusalTolerantCalls
from scripts import full_official_reviewer_transport as transport
from surgical_agent.research.gate.collection_budget import AmbiguousDispatch


class GuardedCalls(RefusalTolerantCalls):
    def call(self, target, stage, seat, body):
        answer = super().call(target,stage,seat,body)
        rows = self.rows if seat in transport.CHANGED else self.delegate.rows
        row = next((r for r in reversed(rows) if (r['target'],r['stage'],r['seat'])==(target,stage,seat)),None)
        fatal = row is None
        if row is not None:
            status = row.get('http_status'); error = row.get('provider_error') or {}
            refusal = status==400 and isinstance(error,dict) and (
                seat=='grok' and str(error.get('code'))=='1301' or
                seat=='qwen' and error.get('code')=='data_inspection_failed')
            fatal = (status is None or not row.get('finished_utc') or row.get('status')=='DISPATCHED'
                or bool(row.get('global_stop')) or status in (400,401,402,403,404) and not refusal
                or row.get('error_type') in ('ReadTimeout','ConnectTimeout','Timeout','ConnectionError'))
        if fatal or self.stop.is_set():
            self.stop.set()
            raise AmbiguousDispatch('stopped request; inspect durable records before continuation')
        return answer
