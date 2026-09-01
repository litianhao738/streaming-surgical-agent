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
    FieldUncertainty,
    JointPerceptionResult,
    PerceptionEvidence,
    RankedCandidate,
)
from surgical_agent.perception.schema import (
    RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    task_layout_for_schema_version,
    validate_joint_perception_payload_by_version,
)


def _invalid() -> None:
    raise ApiSchemaError("Joint perception response violates the strict schema")


def _dense_scores(
    candidates: object,
    *,
    class_count: int,
    confidence_key: str,
) -> tuple[float, ...]:
    if not isinstance(candidates, tuple):
        _invalid()
    dense = [0.0] * class_count
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            _invalid()
        dense[candidate["id"]] = float(candidate[confidence_key])
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
        validate_joint_perception_payload_by_version(payload)
        schema_version = payload["schema_version"]
        task_layout = task_layout_for_schema_version(schema_version)
        confidence_key = (
            "confidence"
            if schema_version == RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
            else "score"
        )
        task_payloads = {task: payload[task] for task, _count in task_layout}
        dense = {
            task: _dense_scores(
                task_payloads[task]["topk"],
                class_count=TASK_CLASS_COUNTS[task],
                confidence_key=confidence_key,
            )
            for task, _count in task_layout
        }
        ranked = {
            task: tuple(
                RankedCandidate(
                    class_id=item["id"], score=float(item[confidence_key])
                )
                for item in task_payloads[task]["topk"]
            )
            for task, _count in task_layout
        }
        if schema_version == RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION:
            confidences = {task: None for task, _count in task_layout}
            references = ()
            uncertainties = tuple(
                FieldUncertainty(
                    path=item["path"],
                    reason=item["reason"],
                    alternative_ids=tuple(item["alternative_ids"]),
                )
                for item in payload["uncertainty"]
            )
        else:
            confidences = {
                task: float(payload["self_reported_confidence"][task])
                for task, _count in task_layout
            }
            references = tuple(
                EvidenceReference(frame_id=item["frame_id"], code=item["code"])
                for item in payload["evidence_refs"]
            )
            uncertainties = ()
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
                field_uncertainties=uncertainties,
            ),
            api_provenance=ApiCallProvenance.from_response(response),
        )
    except ApiSchemaError:
        raise
    except (KeyError, OverflowError, TypeError, ValueError):
        _invalid()
