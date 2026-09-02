"""Checkpoint-backed local Joint Perception runtime backend."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.models.baseline import FrameLogits
from surgical_agent.models.perception import (
    CausalJointPerceptionModel,
    JointPerceptionModelConfig,
)
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.perception.contracts import (
    ApiCallProvenance,
    JointPerceptionResult,
    PerceptionEvidence,
    RankedCandidate,
)
from surgical_agent.training.perception_data import preprocess_causal_frames

LOCAL_JOINT_CHECKPOINT_VERSION = "local_joint_perception_checkpoint_v1"
_TOPK = {"instrument": 3, "verb": 4, "target": 5, "ivt": 8, "phase": 3}


def _task_tensor(logits: FrameLogits, task: str) -> torch.Tensor:
    return getattr(logits, task)


class LocalJointPerceptionBackend:
    """Run the trainable local model without API or GT access."""

    def __init__(
        self,
        model: CausalJointPerceptionModel,
        *,
        device: torch.device,
        thresholds: Mapping[str, float],
        image_size: int,
        window_size: int,
    ) -> None:
        if set(thresholds) != {"instrument", "verb", "target", "ivt"}:
            raise ValueError("thresholds must contain the four multi-label tasks")
        if any(not 0.0 <= float(value) <= 1.0 for value in thresholds.values()):
            raise ValueError("local perception thresholds must lie in [0,1]")
        self.model = model.to(device).eval()
        self.device = device
        self.thresholds = {task: float(value) for task, value in thresholds.items()}
        self.image_size = image_size
        self.window_size = window_size
        self.backend_name = f"local_joint_{model.config.architecture}"

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device,
    ) -> LocalJointPerceptionBackend:
        selected_device = torch.device(device)
        payload = torch.load(
            Path(path).expanduser().resolve(),
            map_location=selected_device,
            weights_only=False,
        )
        if not isinstance(payload, dict) or payload.get("schema_version") != (
            LOCAL_JOINT_CHECKPOINT_VERSION
        ):
            raise ValueError("unsupported local Joint Perception checkpoint")
        raw_config = payload.get("model_config")
        raw_thresholds = payload.get("thresholds")
        if not isinstance(raw_config, dict) or not isinstance(raw_thresholds, dict):
            raise TypeError("local Joint Perception checkpoint metadata is invalid")
        config = JointPerceptionModelConfig(**raw_config)
        model = CausalJointPerceptionModel(config)
        state = payload.get("model_state")
        if not isinstance(state, dict):
            raise TypeError("local Joint Perception checkpoint state is invalid")
        model.load_state_dict(state, strict=True)
        return cls(
            model,
            device=selected_device,
            thresholds=raw_thresholds,
            image_size=int(payload["image_size"]),
            window_size=int(payload["window_size"]),
        )

    def predict(self, context: PerceptionContext) -> JointPerceptionResult:
        if not isinstance(context, PerceptionContext):
            raise TypeError("context must be a PerceptionContext")
        frames = preprocess_causal_frames(
            context.frames,
            image_size=self.image_size,
            window_size=self.window_size,
        ).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(frames.to(self.device))
        probabilities = {
            task: (
                torch.softmax(_task_tensor(logits, task), dim=-1)
                if task == "phase"
                else torch.sigmoid(_task_tensor(logits, task))
            )[0]
            .detach()
            .cpu()
            for task in TASK_CLASS_COUNTS
        }
        selected = {
            task: tuple(
                index
                for index, value in enumerate(probabilities[task])
                if float(value) >= self.thresholds[task]
            )
            for task in ("instrument", "verb", "target", "ivt")
        }
        prediction = InitialPrediction(
            instrument_ids=selected["instrument"],
            verb_ids=selected["verb"],
            target_ids=selected["target"],
            triplet_ids=selected["ivt"],
            phase_id=int(torch.argmax(probabilities["phase"]).item()),
            probabilities={
                task: tuple(float(value) for value in values)
                for task, values in probabilities.items()
            },
            backend=self.backend_name,
            score_semantics="probability_v1",
        )
        rankings = {
            task: tuple(
                RankedCandidate(class_id, float(probabilities[task][class_id]))
                for class_id in sorted(
                    range(TASK_CLASS_COUNTS[task]),
                    key=lambda index: (-float(probabilities[task][index]), index),
                )[: _TOPK[task]]
            )
            for task in TASK_CLASS_COUNTS
        }
        return JointPerceptionResult(
            prediction=prediction,
            raw_evidence=PerceptionEvidence(
                source=self.backend_name,
                ranked_candidates=rankings,
                self_reported_confidence={task: None for task in TASK_CLASS_COUNTS},
                evidence_refs=(),
                source_max_frame_id=context.sample.target_frame_id,
                field_uncertainties=(),
            ),
            api_provenance=ApiCallProvenance(
                source="local_joint_perception",
                provider="local",
                endpoint_identifier=None,
                request_hash=None,
                requested_model_identifier=None,
                returned_model_identifier=None,
                cache_hit=None,
                provider_call_count=0,
            ),
        )


def checkpoint_payload(
    *,
    model: CausalJointPerceptionModel,
    optimizer_state: Mapping[str, Any],
    epoch: int,
    optimizer_steps: int,
    thresholds: Mapping[str, float],
    image_size: int,
    window_size: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the trusted local checkpoint envelope used by training/runtime."""

    return {
        "schema_version": LOCAL_JOINT_CHECKPOINT_VERSION,
        "model_config": {
            "architecture": model.config.architecture,
            "visual_feature_dim": model.config.visual_feature_dim,
            "temporal_hidden_dim": model.config.temporal_hidden_dim,
            "temporal_layers": model.config.temporal_layers,
            "dropout": model.config.dropout,
            "pretrained": False,
        },
        "model_state": model.state_dict(),
        "optimizer_state": dict(optimizer_state),
        "epoch": int(epoch),
        "optimizer_steps": int(optimizer_steps),
        "thresholds": dict(thresholds),
        "image_size": int(image_size),
        "window_size": int(window_size),
        "metadata": dict(metadata),
    }


__all__ = [
    "LOCAL_JOINT_CHECKPOINT_VERSION",
    "LocalJointPerceptionBackend",
    "checkpoint_payload",
]
