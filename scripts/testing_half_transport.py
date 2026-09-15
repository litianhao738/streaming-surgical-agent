"""Half-Testing policy: network/HTTP failures are invalid seats; billing stops."""
from scripts.assess_tracker_review_evidence_r3 import RefusalTolerantCalls
from scripts import full_official_reviewer_transport as transport
from surgical_agent.research.gate.collection_budget import BudgetStop, AmbiguousDispatch


NETWORK_ERRORS = frozenset({'RequestException', 'ConnectionError', 'ProxyError',
    'SSLError', 'Timeout', 'ConnectTimeout', 'ReadTimeout', 'ChunkedEncodingError',
    'ContentDecodingError', 'RetryError'})


def network_failure(row):
    return (row.get('status') == 'FAILED' and bool(row.get('finished_utc'))
            and (row.get('error_type') or row.get('exception_type')) in NETWORK_ERRORS)


def billing_failure(status, error):
    error = error if isinstance(error, dict) else {}
    code = str(error.get('code', '')).lower()
    message = str(error.get('message', '')).lower()
    return status == 402 or code in {'1113', 'arrearage', 'insufficientbalance',
        'insufficient_balance', 'insufficient_credits', 'insufficient_quota'} or any(
        text in message for text in ('insufficient balance', 'insufficient credits', '余额不足', '欠费'))


class ClassifiedStop:
    """Legacy transport stop requests are classified by the outer call before propagation.

    The shared event is never cleared: another worker's billing/budget stop is preserved.
    """
    def __init__(self, shared):
        self.shared = shared
        self.requested = False

    def is_set(self):
        return self.shared.is_set()

    def set(self):
        self.requested = True


class GuardedCalls(RefusalTolerantCalls):
    def __init__(self, run, plan, selected, budget, stop, **kwargs):
        self.global_stop = stop
        self.last_http_failure = None
        super().__init__(run, plan, selected, budget, ClassifiedStop(stop), **kwargs)

    def call(self, target, stage, seat, body):
        self.last_http_failure = None
        answer = super().call(target, stage, seat, body)
        rows = self.rows if seat in transport.CHANGED else self.delegate.rows
        if isinstance(rows, dict):
            rows = rows.values()
        row = next((r for r in reversed(rows) if (r['target'], r['stage'], r['seat']) == (target, stage, seat)), None)
        if row is None or not row.get('finished_utc') or row.get('status') == 'DISPATCHED':
            self.global_stop.set()
            raise AmbiguousDispatch('missing/unfinished request record; inspect before continuation')
        status = row.get('http_status')
        error = row.get('provider_error')
        if seat not in transport.CHANGED:
            path = self.folder/'run/calls'/f"{row['index']:03d}_{target}_{stage}_{seat}"/'response.json'
            if path.exists():
                error = transport.old.old.read(path).get('body', {}).get('error')
        if billing_failure(status, error):
            self.global_stop.set()
            raise BudgetStop('provider balance/credits exhausted: '+seat)
        if self.global_stop.is_set():
            raise BudgetStop('another worker stopped the run')
        if (status is not None and status >= 400) or network_failure(row):
            self.last_http_failure = dict(target=target, stage=stage, seat=seat, http_status=status,
                                          provider_error=error,
                                          error_type=row.get('error_type') or row.get('exception_type'))
            # Original journals stay unchanged, including historical global_stop flags.
            transport.old.write(self.run/'request_policy'/target/(stage+'_'+seat+'.json'),
                dict(policy='nonbilling_network_or_http_failure_is_invalid_seat_v2',
                     **self.last_http_failure, continued=True, original_status=row['status'],
                     automatic_retry=False))
            if self.delegate is not None and hasattr(self.delegate, 'persist'):
                self.delegate.stopped = False
                self.delegate.persist()
            return None
        if (status is None and not row.get('recovered_successful_response')) or row.get('global_stop'):
            self.global_stop.set()
            raise AmbiguousDispatch('transport/integrity failure; inspect durable records')
        if self.global_stop.is_set():
            raise BudgetStop('another worker stopped the run')
        return answer
