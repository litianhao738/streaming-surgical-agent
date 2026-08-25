"""Gold-free ordered per-video inference boundary. Runtime arrives in P11."""

from __future__ import annotations

from typing import Protocol

from surgical_agent.data.schemas import InferenceSample


class InferenceConsumer(Protocol):
    """Future runtime interfaces may accept only Gold-free samples."""

    def predict(self, sample: InferenceSample) -> object:
        """Predict from visual/context input without access to evaluation targets."""


def require_inference_sample(sample: object) -> InferenceSample:
    """Fail closed if a GT-bearing object is passed to runtime inference."""

    if not isinstance(sample, InferenceSample):
        raise TypeError("Runtime inference accepts only InferenceSample")
    return sample
