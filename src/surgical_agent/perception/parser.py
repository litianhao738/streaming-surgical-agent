"""Conversion of one validated API response into joint-perception contracts."""

from __future__ import annotations

from collections.abc import Mapping

from surgical_agent.api.contracts import ApiResponseRecord
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.contracts import (
    ApiCallProvenance,
    EvidenceReference,
    JointPerceptionResult,
    PerceptionEvidence,
    RankedCandidate,
)
from surgical_agent.perception.schema import (
    TASK_LAYOUT,
    validate_joint_perception_payload,
)


def _invalid() -> None:
    raise ApiSchemaError("Joint perception response violates the strict schema")


def _dense_scores(candidates: object, *, class_count: int) -> tuple[float, ...]:
    if not isinstance(candidates, tuple):
        _invalid()
    dense = [0.0] * class_count
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            _invalid()
        dense[candidate["id"]] = float(candidate["score"])
    return tuple(dense)


def _selected_ids(payload: Mapping[str, object], task: str) -> tuple[int, ...]:
    values = payload[task]["selected_ids"]
    if not isinstance(values, tuple):
        _invalid()
    return tuple(sorted(values))


def parse_joint_perception_response(
    response: ApiResponseRecord,
    *,
    video_id: str,
    frame_id: int,
    backend: str,
) -> JointPerceptionResult:
    """Fail closed while turning a safe API record into frozen result contracts."""

    try:
        if not isinstance(response, ApiResponseRecord):
            _invalid()
        if not isinstance(video_id, str) or not video_id.strip():
            _invalid()
        if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 0:
            _invalid()
        if not isinstance(backend, str) or not backend.strip():
            _invalid()

        payload = response.parsed_payload
        validate_joint_perception_payload(payload)
        task_payloads = {task: payload[task] for task, _count in TASK_LAYOUT}
        dense = {
            task: _dense_scores(
                task_payloads[task]["topk"],
                class_count=TASK_CLASS_COUNTS[task],
            )
            for task, _count in TASK_LAYOUT
        }
        ranked = {
            task: tuple(
                RankedCandidate(class_id=item["id"], score=float(item["score"]))
                for item in task_payloads[task]["topk"]
            )
            for task, _count in TASK_LAYOUT
        }
        confidences = {
            task: float(payload["self_reported_confidence"][task])
            for task, _count in TASK_LAYOUT
        }
        references = tuple(
            EvidenceReference(frame_id=item["frame_id"], code=item["code"])
            for item in payload["evidence_refs"]
        )
        prediction = InitialPrediction(
            instrument_ids=_selected_ids(task_payloads, "instrument"),
            verb_ids=_selected_ids(task_payloads, "verb"),
            target_ids=_selected_ids(task_payloads, "target"),
            triplet_ids=_selected_ids(task_payloads, "ivt"),
            phase_id=task_payloads["phase"]["selected_id"],
            probabilities=dense,
            score_semantics="uncalibrated_rank_v1",
            backend=backend,
        )
        return JointPerceptionResult(
            prediction=prediction,
            raw_evidence=PerceptionEvidence(
                source=backend,
                ranked_candidates=ranked,
                self_reported_confidence=confidences,
                evidence_refs=references,
                source_max_frame_id=frame_id,
            ),
            api_provenance=ApiCallProvenance.from_response(response),
        )
    except ApiSchemaError:
        raise
    except (KeyError, OverflowError, TypeError, ValueError):
        _invalid()
