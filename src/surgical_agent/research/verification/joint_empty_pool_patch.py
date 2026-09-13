"""Versioned local empty-four-head fallback; never relax reviewer validation.

The original module stays frozen. A private function namespace replaces only
the empty compact aggregation, leaving Phase validation and all other code intact.
"""
from types import FunctionType
from surgical_agent.research.verification import prior_gated_joint as original

VERSION = 'prior-gated-joint-mainline-v1.0.1-empty-pool'


def compact_aggregate(reviews, pool, *, image_count=3):
    if not pool['propositions']:
        return {}, {'empty_pool_fallback': 'No four-head candidates; no model pass inferred.'}
    return original.compact_aggregate(reviews, pool, image_count=image_count)


_globals = dict(original.decide.__globals__, compact_aggregate=compact_aggregate)
decide = FunctionType(original.decide.__code__, _globals, 'decide_empty_pool_patch', original.decide.__defaults__)
decide.__kwdefaults__ = original.decide.__kwdefaults__
