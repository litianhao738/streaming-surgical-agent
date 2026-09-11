"""Recover recorded review rejection reasons without changing votes or predictions."""
from copy import deepcopy


def explain_invalid_items(diagnostics, formatting):
    """Normalize-report errors are evidence; absent reports remain explicitly unknown.

    The original admission diagnostics are retained verbatim. A semantic rejection
    must not be relabelled as broken JSON simply because normalization omitted it.
    This function never makes an invalid opinion valid or alters a numeric score.
    """
    explained = {}
    for candidate, item in diagnostics.items():
        details = {}
        for seat, admission_error in item.get("invalid", {}).items():
            source = formatting.get(seat, {})
            reasons = source.get("errors", {}).get(candidate, [])
            if not isinstance(reasons, list) or not reasons or not all(isinstance(r, str) for r in reasons):
                reasons = [admission_error]
            details[seat] = {"admission_error": admission_error, "recorded_reasons": list(reasons),
                "semantic_rejection": any(r in {"LOCAL_ABSENCE_CANNOT_DELETE_FRAME_LABEL",
                    "RATING_FINDING_CONFLICT", "TARGET_ONLY_FINDING", "NO_CURRENT_FRAME_EVIDENCE",
                    "UNCERTAIN_SUPPORT"} for r in reasons),
                "valid_vote": False}
        if details:
            explained[candidate] = details
    return {"original_diagnostics": deepcopy(diagnostics), "invalid_item_details": explained,
            "votes_changed": False, "predictions_changed": False}
