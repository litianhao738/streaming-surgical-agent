"""Exact, causal loading of local media referenced by API runtime samples."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from surgical_agent.data.dataset import (
    MP4_ALIGNMENT_VERSION,
    PNG_ALIGNMENT_VERSION,
    DatasetContractError,
)
from surgical_agent.data.schemas import DatasetSplit, InferenceSample


@dataclass(frozen=True)
class LoadedApiWindow:
    """Path-free runtime sample paired with its normalized causal frame tensor."""

    runtime_sample: InferenceSample
    frames: Tensor


class VideoFrameReader(Protocol):
    """Boundary for reading exact RGB frames from a video decoder."""

    def read_many_rgb(
        self,
        path: Path,
        decoder_indices: tuple[int, ...],
    ) -> tuple[np.ndarray, ...]: ...


class OpenCvVideoFrameReader:
    """OpenCV decoder that opens one video capture for a causal window."""

    def read_many_rgb(
        self,
        path: Path,
        decoder_indices: tuple[int, ...],
    ) -> tuple[np.ndarray, ...]:
        try:
            import cv2
        except ImportError as exc:
            raise DatasetContractError("OpenCV MP4 decoding is unavailable") from exc

        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise DatasetContractError("Unable to open MP4 causal media")
            frames: list[np.ndarray] = []
            for decoder_index in decoder_indices:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, decoder_index):
                    raise DatasetContractError("Unable to decode an exact MP4 frame")
                positioned_index = capture.get(cv2.CAP_PROP_POS_FRAMES)
                if (
                    not np.isfinite(positioned_index)
                    or abs(positioned_index - decoder_index) > 0.5
                ):
                    raise DatasetContractError("Unable to decode an exact MP4 frame")
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise DatasetContractError("Unable to decode an exact MP4 frame")
                next_index = capture.get(cv2.CAP_PROP_POS_FRAMES)
                if (
                    not np.isfinite(next_index)
                    or abs(next_index - (decoder_index + 1)) > 0.5
                ):
                    raise DatasetContractError("Unable to decode an exact MP4 frame")
                frames.append(np.ascontiguousarray(frame[:, :, ::-1]))
            return tuple(frames)
        finally:
            capture.release()


class CausalApiMediaLoader:
    """Load a short causal window while removing local paths from runtime data."""

    def __init__(self, *, video_reader: VideoFrameReader | None = None) -> None:
        self._video_reader = (
            OpenCvVideoFrameReader() if video_reader is None else video_reader
        )

    def load(self, sample: InferenceSample) -> LoadedApiWindow:
        if not isinstance(sample, InferenceSample):
            raise TypeError("Causal API media loading accepts only InferenceSample")
        if len(sample.causal_frame_ids) > 6:
            raise DatasetContractError("Causal API media windows may contain at most six frames")

        if sample.source_split is DatasetSplit.TESTING:
            arrays = self._load_test_mp4(sample)
        else:
            arrays = self._load_png(sample)
        frames = self._normalize_rgb_window(arrays)
        runtime_sample = replace(
            sample,
            media_refs=tuple(
                f"cholectrack20:{sample.video_id}:frame:{frame_id}"
                for frame_id in sample.causal_frame_ids
            ),
        )
        return LoadedApiWindow(runtime_sample=runtime_sample, frames=frames)

    def _load_png(self, sample: InferenceSample) -> tuple[np.ndarray, ...]:
        if sample.alignment_version != PNG_ALIGNMENT_VERSION:
            raise DatasetContractError("PNG sample has an unsupported media alignment")
        arrays: list[np.ndarray] = []
        for media_ref in sample.media_refs:
            path = Path(media_ref)
            if path.suffix.lower() != ".png":
                raise DatasetContractError("PNG sample requires PNG media paths")
            try:
                with Image.open(path) as image:
                    arrays.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
            except (OSError, ValueError) as exc:
                raise DatasetContractError("Unable to load PNG causal media") from exc
        return tuple(arrays)

    def _load_test_mp4(self, sample: InferenceSample) -> tuple[np.ndarray, ...]:
        if sample.alignment_version != MP4_ALIGNMENT_VERSION:
            raise DatasetContractError("MP4 sample has an unsupported media alignment")
        if any(frame_id <= 0 for frame_id in sample.causal_frame_ids):
            raise DatasetContractError("Test annotation frame IDs must be positive")
        paths = tuple(Path(media_ref).expanduser().resolve() for media_ref in sample.media_refs)
        if any(path.suffix.lower() != ".mp4" for path in paths):
            raise DatasetContractError("Testing samples require one MP4 media path")
        if len(set(paths)) != 1:
            raise DatasetContractError("Testing causal windows require one common MP4 path")
        arrays = self._video_reader.read_many_rgb(
            paths[0],
            tuple(frame_id - 1 for frame_id in sample.causal_frame_ids),
        )
        if len(arrays) != len(sample.causal_frame_ids):
            raise DatasetContractError("MP4 decoder did not return every causal frame")
        return arrays

    @staticmethod
    def _normalize_rgb_window(arrays: tuple[np.ndarray, ...]) -> Tensor:
        expected_shape: tuple[int, int, int] | None = None
        frames: list[Tensor] = []
        for array in arrays:
            rgb = np.asarray(array)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise DatasetContractError("Causal media frame must be HWC RGB")
            if expected_shape is None:
                expected_shape = rgb.shape
            elif rgb.shape != expected_shape:
                raise DatasetContractError("Causal media frames have mismatched geometry")
            if not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 255):
                raise DatasetContractError("Causal media RGB values must be finite uint8-range")
            normalized = np.ascontiguousarray(rgb, dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(normalized).permute(2, 0, 1).contiguous())
        if not frames:
            raise DatasetContractError("Causal media window is empty")
        return torch.stack(frames)
