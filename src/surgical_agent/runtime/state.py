"""Minimal observation and per-video state contracts for the final Pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass

from surgical_agent.data.schemas import InferenceSample


@dataclass(frozen=True, order=True)
class ObservationIdentity:
    """Stable identity of one accepted stream observation."""

    video_id: str
    segment_id: str
    frame_id: int
    observation_time: float

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be non-empty")
        if not isinstance(self.segment_id, str) or not self.segment_id.strip():
            raise ValueError("segment_id must be non-empty")
        if not isinstance(self.frame_id, int) or isinstance(self.frame_id, bool) or self.frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer")
        if (
            not isinstance(self.observation_time, (int, float))
            or isinstance(self.observation_time, bool)
            or not math.isfinite(float(self.observation_time))
            or float(self.observation_time) < 0.0
        ):
            raise ValueError("observation_time must be finite and non-negative")
        object.__setattr__(self, "video_id", self.video_id.strip())
        object.__setattr__(self, "segment_id", self.segment_id.strip())
        object.__setattr__(self, "observation_time", float(self.observation_time))

    @property
    def key(self) -> str:
        return f"{self.video_id}:{self.segment_id}:{self.frame_id}"

    @classmethod
    def from_sample(
        cls,
        sample: InferenceSample,
        *,
        segment_id: str = "segment-0",
        observation_time: float | None = None,
    ) -> ObservationIdentity:
        if not isinstance(sample, InferenceSample):
            raise TypeError("sample must be InferenceSample")
        return cls(
            video_id=sample.video_id,
            segment_id=segment_id,
            frame_id=sample.target_frame_id,
            observation_time=(
                float(sample.target_frame_id)
                if observation_time is None
                else observation_time
            ),
        )


class ObservationClock:
    """Reject cross-video, backward-time and duplicate stream observations."""

    def __init__(self) -> None:
        self._video_id: str | None = None
        self._last: ObservationIdentity | None = None

    def reset(self, video_id: str) -> None:
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("video_id must be non-empty")
        self._video_id = video_id.strip()
        self._last = None

    def accept(self, observation: ObservationIdentity) -> bool:
        """Accept an observation; return True when its segment starts anew."""

        if not isinstance(observation, ObservationIdentity):
            raise TypeError("observation must be ObservationIdentity")
        if self._video_id != observation.video_id:
            raise ValueError("observation clock crossed a video boundary without reset")
        previous = self._last
        if previous is not None:
            if observation.observation_time <= previous.observation_time:
                raise ValueError("observation time must be strictly increasing")
            if (
                observation.segment_id == previous.segment_id
                and observation.frame_id <= previous.frame_id
            ):
                raise ValueError("frame IDs must increase within one segment")
        new_segment = previous is None or observation.segment_id != previous.segment_id
        self._last = observation
        return new_segment


__all__ = ["ObservationClock", "ObservationIdentity"]
