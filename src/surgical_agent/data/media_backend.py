"""Explicit frame-folder and MP4 media alignment contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from surgical_agent.data.schemas import DatasetSplit


class AlignmentStatus(str, Enum):
    EXACT = "EXACT"
    EXTRA_MEDIA_ONLY = "EXTRA_MEDIA_ONLY"
    MISSING_ANNOTATED_MEDIA = "MISSING_ANNOTATED_MEDIA"
    NONTRIVIAL_ALIGNMENT = "NONTRIVIAL_ALIGNMENT"
    UNRESOLVED = "UNRESOLVED"


class MediaAlignmentError(LookupError):
    """Raised instead of applying an undocumented media fallback."""


@dataclass(frozen=True)
class MediaReference:
    """One auditable visual source resolved from a canonical frame ID."""

    video_id: str
    split: DatasetSplit
    canonical_frame_id: int
    media_path: str
    media_kind: str
    original_media_id: int
    decoder_frame_index: int | None
    alignment_rule: str
    alignment_version: str
    status: AlignmentStatus


class ExactFrameFolderResolver:
    """Resolve only equal numeric PNG stems; never choose a nearby frame."""

    def __init__(
        self,
        *,
        video_id: str,
        split: DatasetSplit,
        frames_dir: str | Path,
        alignment_version: str = "ct20_exact_png_stem_v1",
    ) -> None:
        self.video_id = video_id
        self.split = split
        self.frames_dir = Path(frames_dir).expanduser().resolve()
        self.alignment_version = alignment_version
        self._paths: dict[int, Path] = {}
        for path in sorted(self.frames_dir.glob("*.png")):
            try:
                frame_id = int(path.stem)
            except ValueError as exc:
                raise MediaAlignmentError(f"Non-numeric PNG filename: {path}") from exc
            if frame_id in self._paths:
                raise MediaAlignmentError(f"Duplicate numeric media ID {frame_id}")
            self._paths[frame_id] = path

    @property
    def available_frame_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._paths))

    def resolve(self, frame_id: int) -> MediaReference:
        path = self._paths.get(frame_id)
        if path is None:
            raise MediaAlignmentError(
                f"No exact PNG for {self.video_id} frame {frame_id}; fallback is forbidden"
            )
        return MediaReference(
            video_id=self.video_id,
            split=self.split,
            canonical_frame_id=frame_id,
            media_path=str(path),
            media_kind="png",
            original_media_id=frame_id,
            decoder_frame_index=None,
            alignment_rule="canonical_frame_id == numeric PNG stem",
            alignment_version=self.alignment_version,
            status=AlignmentStatus.EXACT,
        )


class Mp4FrameResolver:
    """Resolve MP4 frames only when the decoder offset is explicitly verified."""

    def __init__(
        self,
        *,
        video_id: str,
        split: DatasetSplit,
        media_path: str | Path,
        frame_count: int,
        decoder_index_offset: int | None,
        alignment_version: str,
    ) -> None:
        self.video_id = video_id
        self.split = split
        self.media_path = Path(media_path).expanduser().resolve()
        self.frame_count = frame_count
        self.decoder_index_offset = decoder_index_offset
        self.alignment_version = alignment_version

    def resolve(self, frame_id: int) -> MediaReference:
        if self.decoder_index_offset is None:
            raise MediaAlignmentError(
                f"MP4 decoder index offset is unresolved for {self.video_id}"
            )
        decoder_index = frame_id + self.decoder_index_offset
        if decoder_index < 0 or decoder_index >= self.frame_count:
            raise MediaAlignmentError(
                f"Resolved decoder index {decoder_index} is outside [0, {self.frame_count})"
            )
        return MediaReference(
            video_id=self.video_id,
            split=self.split,
            canonical_frame_id=frame_id,
            media_path=str(self.media_path),
            media_kind="mp4_frame",
            original_media_id=frame_id,
            decoder_frame_index=decoder_index,
            alignment_rule=f"decoder_index = frame_id + {self.decoder_index_offset}",
            alignment_version=self.alignment_version,
            status=AlignmentStatus.EXACT,
        )
