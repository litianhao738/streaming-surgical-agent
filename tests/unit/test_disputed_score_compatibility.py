"""Exact numeric equality must not weaken immutable inference audit checks."""
from decimal import Decimal

import pytest

from scripts.score_disputed_relation_trial import exact_same


def test_exact_decimal_amounts_and_empty_account_sum():
    exact_same(Decimal("0.100"), Decimal("0.1"), "account")
    exact_same(0, Decimal(0), "empty account")


def test_changed_amount_or_wrong_type_is_rejected():
    for value in (Decimal("0.10000001"), "0.1", True, 0.1):
        with pytest.raises(ValueError):
            exact_same(value, Decimal("0.1"), "account")


def test_nonnumeric_frozen_evidence_check_remains_exact():
    exact_same({"ids": [1, 2]}, {"ids": [1, 2]}, "wire")
    with pytest.raises(ValueError):
        exact_same({"ids": [1, 2]}, {"ids": [2, 1]}, "wire")
