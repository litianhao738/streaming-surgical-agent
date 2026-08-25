"""CholecT50 VID31 alignment and frame-label contract tests."""

from pathlib import Path

import pytest

from tools.data_resolution.cholect50_vid31.audit_vid31 import (
    parse_cholect50_frame,
    parse_label_mapping,
    track20_to_cholect50_frame,
)


def test_track20_to_cholect50_frame_uses_verified_origin_and_stride() -> None:
    assert track20_to_cholect50_frame(6201) == 0
    assert track20_to_cholect50_frame(6226) == 1
    with pytest.raises(ValueError, match="not aligned"):
        track20_to_cholect50_frame(6202)


def test_label_mapping_requires_complete_100_class_relation(tmp_path: Path) -> None:
    path = tmp_path / "mapping.txt"
    path.write_text(
        "# IVT, I, V, T, IV, IT\n"
        + "\n".join(f"{index},0,0,0,0,0" for index in range(100)),
        encoding="utf-8",
    )
    mapping = parse_label_mapping(path)
    assert len(mapping) == 100
    assert mapping[99] == (0, 0, 0)


def test_frame_parser_preserves_frame_level_multilabel_and_null_row() -> None:
    mapping = {19: (0, 1, 8)}
    positive = [19, 0, 1.0, -1.0, -1.0, -1.0, -1.0, 1, 8, 1.0, -1, -1, -1, -1, 0]
    parsed = parse_cholect50_frame([positive], mapping)
    assert parsed["triplet_ids"] == [19]
    assert parsed["instrument_ids"] == [0]
    assert parsed["action_present"] is True

    negative = [-1, -1, 1.0, -1, -1, -1, -1, -1, -1, 1.0, -1, -1, -1, -1, 0]
    parsed_negative = parse_cholect50_frame([negative], mapping)
    assert parsed_negative["triplet_ids"] == []
    assert parsed_negative["negative_sentinel_row_count"] == 1
