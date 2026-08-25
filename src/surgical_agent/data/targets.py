"""Gold-free runtime and GT-bearing evaluation partition."""

from __future__ import annotations

from collections.abc import Sequence

from surgical_agent.data.schemas import (
    CanonicalFrameAnnotation,
    EvaluationInstanceTarget,
    EvaluationTarget,
    InferenceSample,
)


def build_inference_sample(
    frame: CanonicalFrameAnnotation,
    *,
    causal_frame_ids: Sequence[int],
    media_refs: Sequence[str],
    alignment_version: str,
) -> InferenceSample:
    """Create the runtime branch without copying any canonical labels."""

    return InferenceSample(
        video_id=frame.video_id,
        target_frame_id=frame.frame_id,
        causal_frame_ids=tuple(causal_frame_ids),
        media_refs=tuple(media_refs),
        source_split=frame.provenance.split,
        alignment_version=alignment_version,
    )


def build_evaluation_target(frame: CanonicalFrameAnnotation) -> EvaluationTarget:
    """Create the isolated GT branch consumed only by evaluation."""

    instances = tuple(
        EvaluationInstanceTarget(
            instrument_id=instance.instrument_id,
            verb_id=instance.verb_id,
            target_id=instance.target_id,
            triplet_id=instance.triplet_id,
            phase_id=instance.phase_id,
            operator_id=instance.operator_id,
            bbox=instance.bbox,
            tracks=instance.tracks,
            mask=instance.mask,
        )
        for instance in frame.instances
    )
    return EvaluationTarget(
        video_id=frame.video_id,
        frame_id=frame.frame_id,
        instances=instances,
    )
