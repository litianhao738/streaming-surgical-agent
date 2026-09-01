"""Gold-free contracts for joint perception evidence and API provenance."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from surgical_agent.data.constants import TASK_CLASS_COUNTS, TASK_ID_BOUNDS

if TYPE_CHECKING:
    from surgical_agent.api.contracts import ApiResponseRecord
    from surgical_agent.inference.schemas import InitialPrediction
    from surgical_agent.perception.context_builder import PerceptionContext


TASK_NAMES = tuple(TASK_CLASS_COUNTS)
EVIDENCE_REF_CODES = frozenset(
    {
        "CURRENT_VISUAL_SUPPORT",
        "CAUSAL_VISUAL_TREND",
        "PRIOR_STATE_SUPPORT",
        "AMBIGUOUS_VISUAL_SUPPORT",
    }
)
FIELD_UNCERTAINTY_PATHS = {
    "/instrument/selected_ids": "instrument",
    "/verb/selected_ids": "verb",
    "/target/selected_ids": "target",
    "/ivt/selected_ids": "ivt",
    "/phase/selected_id": "phase",
}
FIELD_UNCERTAINTY_REASONS = frozenset(
    {
        "LOW_VISUAL_CONFIDENCE",
        "CLOSE_ALTERNATIVES",
        "OCCLUSION",
        "MOTION_BLUR",
        "TEMPORAL_AMBIGUITY",
        "OTHER_VISUAL_AMBIGUITY",
    }
)


def _require_nonempty_string(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_nonnegative_int(value: object, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_score(value: object, *, name: str = "score") -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be finite values in [0, 1]")


def _require_optional_score(value: object, *, name: str) -> None:
    if value is not None:
        _require_score(value, name=name)


def _require_sha256(value: object, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class RankedCandidate:
    """One ontology class and its bounded ranking score."""

    class_id: int
    score: float

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.class_id, name="class_id")
        _require_score(self.score)

    @property
    def confidence(self) -> float:
        """Read the v2 wire-name without renaming the established score field."""

        return self.score


@dataclass(frozen=True)
class FieldUncertainty:
    """One bounded, field-scoped visual ambiguity declared by the provider."""

    path: str
    reason: str
    alternative_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or self.path not in FIELD_UNCERTAINTY_PATHS:
            raise ValueError("field uncertainty path is unsupported")
        if (
            not isinstance(self.reason, str)
            or self.reason not in FIELD_UNCERTAINTY_REASONS
        ):
            raise ValueError("field uncertainty reason is outside the closed vocabulary")
        alternatives = tuple(self.alternative_ids)
        task = FIELD_UNCERTAINTY_PATHS[self.path]
        lower, upper = TASK_ID_BOUNDS[task]
        if any(
            not isinstance(candidate, int)
            or isinstance(candidate, bool)
            or not lower <= candidate <= upper
            for candidate in alternatives
        ):
            raise ValueError("field uncertainty alternative ID is outside its ontology")
        if len(set(alternatives)) != len(alternatives):
            raise ValueError("field uncertainty alternative IDs must be unique")
        object.__setattr__(self, "alternative_ids", alternatives)


@dataclass(frozen=True)
class EvidenceReference:
    """A closed-vocabulary, causal frame reference supporting a prediction."""

    frame_id: int
    code: str

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.frame_id, name="evidence frame_id")
        if self.code not in EVIDENCE_REF_CODES:
            raise ValueError("evidence reference code is outside the closed vocabulary")


@dataclass(frozen=True)
class PerceptionEvidence:
    """Strict five-head evidence emitted alongside every joint prediction."""

    source: str
    ranked_candidates: Mapping[str, tuple[RankedCandidate, ...]]
    self_reported_confidence: Mapping[str, float | None]
    evidence_refs: tuple[EvidenceReference, ...]
    source_max_frame_id: int
    field_uncertainties: tuple[FieldUncertainty, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.source, name="evidence source")
        _require_nonnegative_int(self.source_max_frame_id, name="source_max_frame_id")

        if not isinstance(self.ranked_candidates, Mapping):
            raise TypeError("ranked_candidates must be a mapping")
        if set(self.ranked_candidates) != set(TASK_NAMES):
            raise ValueError("ranked_candidates must contain exactly the five task heads")

        normalized_candidates: dict[str, tuple[RankedCandidate, ...]] = {}
        for task in TASK_NAMES:
            values = tuple(self.ranked_candidates[task])
            if any(not isinstance(candidate, RankedCandidate) for candidate in values):
                raise TypeError("ranked_candidates must contain RankedCandidate values")
            class_ids = tuple(candidate.class_id for candidate in values)
            if len(set(class_ids)) != len(class_ids):
                raise ValueError(f"{task} candidate IDs must be unique")
            lower, upper = TASK_ID_BOUNDS[task]
            if any(class_id < lower or class_id > upper for class_id in class_ids):
                raise ValueError(f"{task} candidate ID is outside {lower}..{upper}")
            if any(
                values[index].score < values[index + 1].score
                for index in range(len(values) - 1)
            ):
                raise ValueError(f"{task} candidate scores must be descending")
            normalized_candidates[task] = values
        object.__setattr__(
            self, "ranked_candidates", MappingProxyType(normalized_candidates)
        )

        if not isinstance(self.self_reported_confidence, Mapping):
            raise TypeError("self_reported_confidence must be a mapping")
        if set(self.self_reported_confidence) != set(TASK_NAMES):
            raise ValueError(
                "self_reported_confidence must contain exactly the five task heads"
            )
        normalized_confidence: dict[str, float | None] = {}
        for task in TASK_NAMES:
            value = self.self_reported_confidence[task]
            _require_optional_score(value, name=f"{task} confidence")
            normalized_confidence[task] = value
        object.__setattr__(
            self,
            "self_reported_confidence",
            MappingProxyType(normalized_confidence),
        )

        refs = tuple(self.evidence_refs)
        if any(not isinstance(ref, EvidenceReference) for ref in refs):
            raise TypeError("evidence_refs must contain EvidenceReference values")
        if any(ref.frame_id > self.source_max_frame_id for ref in refs):
            raise ValueError("evidence references must use causal frame IDs")
        object.__setattr__(self, "evidence_refs", refs)

        uncertainties = tuple(self.field_uncertainties)
        if len(uncertainties) > 5:
            raise ValueError("field_uncertainties must contain at most five findings")
        if any(not isinstance(item, FieldUncertainty) for item in uncertainties):
            raise TypeError("field_uncertainties must contain FieldUncertainty values")
        paths = tuple(item.path for item in uncertainties)
        if len(set(paths)) != len(paths):
            raise ValueError("field uncertainty paths must be unique")
        for item in uncertainties:
            task = FIELD_UNCERTAINTY_PATHS[item.path]
            topk_ids = {candidate.class_id for candidate in normalized_candidates[task]}
            if not set(item.alternative_ids) <= topk_ids:
                raise ValueError(
                    "field uncertainty alternatives must occur in the task ranking"
                )
        object.__setattr__(self, "field_uncertainties", uncertainties)

    @classmethod
    def local_unavailable(cls, frame_id: int) -> PerceptionEvidence:
        """Represent unavailable evidence from the local smoke backend."""

        _require_nonnegative_int(frame_id, name="frame_id")
        return cls(
            source="local_smoke",
            ranked_candidates={task: () for task in TASK_NAMES},
            self_reported_confidence={task: None for task in TASK_NAMES},
            evidence_refs=(),
            source_max_frame_id=frame_id,
        )


@dataclass(frozen=True)
class ApiCallProvenance:
    """Allowlisted API accounting and identity fields only."""

    source: str
    provider: str
    endpoint_identifier: str | None
    request_hash: str | None
    requested_model_identifier: str | None
    returned_model_identifier: str | None
    cache_hit: bool | None
    provider_call_count: int

    def __post_init__(self) -> None:
        _require_nonempty_string(self.source, name="provenance source")
        _require_nonempty_string(self.provider, name="provenance provider")
        for name in (
            "endpoint_identifier",
            "requested_model_identifier",
            "returned_model_identifier",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_nonempty_string(value, name=name)
        if self.request_hash is not None:
            _require_sha256(self.request_hash, name="request_hash")
        if self.cache_hit is not None and type(self.cache_hit) is not bool:
            raise TypeError("cache_hit must be a boolean when available")
        _require_nonnegative_int(
            self.provider_call_count, name="provider_call_count"
        )

    @classmethod
    def from_response(
        cls,
        response: ApiResponseRecord,
        *,
        source: str | None = None,
    ) -> ApiCallProvenance:
        """Copy only safe API identity/accounting fields from an API response."""

        from surgical_agent.api.contracts import ApiResponseRecord

        if not isinstance(response, ApiResponseRecord):
            raise TypeError("response must be an ApiResponseRecord")
        return cls(
            source=response.provider if source is None else source,
            provider=response.provider,
            endpoint_identifier=response.endpoint_identifier,
            request_hash=response.request_hash,
            requested_model_identifier=response.requested_model_identifier,
            returned_model_identifier=response.returned_model_identifier,
            cache_hit=response.cache_hit,
            provider_call_count=response.provider_call_count,
        )

    @classmethod
    def local(cls) -> ApiCallProvenance:
        """Represent a local backend run with no API call or model identity."""

        return cls(
            source="local_smoke",
            provider="local",
            endpoint_identifier=None,
            request_hash=None,
            requested_model_identifier=None,
            returned_model_identifier=None,
            cache_hit=None,
            provider_call_count=0,
        )


@dataclass(frozen=True)
class JointPerceptionResult:
    """Backend-neutral prediction, evidence, and safe API provenance."""

    prediction: InitialPrediction
    raw_evidence: PerceptionEvidence
    api_provenance: ApiCallProvenance

    def __post_init__(self) -> None:
        from surgical_agent.inference.schemas import InitialPrediction

        if not isinstance(self.prediction, InitialPrediction):
            raise TypeError("prediction must be an InitialPrediction")
        if not isinstance(self.raw_evidence, PerceptionEvidence):
            raise TypeError("raw_evidence must be PerceptionEvidence")
        if not isinstance(self.api_provenance, ApiCallProvenance):
            raise TypeError("api_provenance must be ApiCallProvenance")


class PerceptionBackend(Protocol):
    """Backend boundary for causal context to joint perception results."""

    def predict(self, context: PerceptionContext) -> JointPerceptionResult:
        raise NotImplementedError
