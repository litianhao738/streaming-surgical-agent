"""Read-only Phase error decomposition on a completed development experiment."""
import hashlib
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/preflight/phase_temporal_ratings_same32_20260910_v2"
DATA = Path("D:/cholec_dataset/Training")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    rows = read(OUTPUT / "predictions.json")
    truths = {f"{r['video_id']}_{r['frame_id']}": r for r in read(OUTPUT / "scored_truth.json")}
    phase_maps, annotations, source_hashes = {}, {}, {}
    for video in sorted({r["video_id"] for r in rows}):
        path = DATA / video / ("vid31_phase_repaired.json" if video == "VID31" else f"{video.lower()}.json")
        raw = read(path)
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if video == "VID31":
            phase_maps[video] = {int(k): v["phase_id"] for k, v in raw["phase_by_track20_image_frame_id"].items()}
        else:
            annotations[video] = raw["annotations"]
            mapping = {}
            for frame, items in raw["annotations"].items():
                values = {x["phase"] for x in items if x.get("phase") in range(7)}
                assert len(values) <= 1
                if values:
                    mapping[int(frame)] = next(iter(values))
            phase_maps[video] = mapping
    support = Counter(t["gt"]["phase"][0] for t in truths.values() if t["mask"]["phase"])
    arms = ("h0", "cached_default", "compact_repeat", "revised_phase")
    confusions, per_class = {}, {}
    for arm in arms:
        pairs = Counter((truths[r["key"]]["gt"]["phase"][0], r[arm]["phase"][0]) for r in rows)
        confusions[arm] = {f"{a}->{b}": n for (a, b), n in sorted(pairs.items()) if a != b}
        per_class[arm] = {str(p): {"support": support[p], "correct": pairs[p, p]} for p in range(7)}
    errors, seats, citations = [], {}, {}
    for row in rows:
        key, video, frame = row["key"], row["video_id"], row["frame_id"]
        gt, h0 = truths[key]["gt"]["phase"][0], row["h0"]["phase"][0]
        assert phase_maps[video][frame] == gt
        branches = read(OUTPUT / f"targets/{key}/result.json")["branches"]
        for arm in ("compact_repeat", "revised_phase"):
            citations.setdefault(arm, Counter())
            seats.setdefault(arm, {})
            for seat, raw in branches[arm]["raw"].items():
                citations[arm][str(raw["image_indices"])] += 1
                count = seats[arm].setdefault(seat, Counter())
                maximum = max(raw["ratings"].values())
                winners = [int(p) for p, score in raw["ratings"].items() if score == maximum]
                count["unique_top_correct" if winners == [gt] else "tied_top" if len(winners) > 1 else "unique_top_wrong"] += 1
                if gt != h0:
                    count["on_h0_errors_correct_top" if winners == [gt] else "on_h0_errors_other"] += 1
        if gt == row["revised_phase"]["phase"][0]:
            continue
        series = sorted(phase_maps[video].items())
        transitions = [b[0] for a, b in pairwise(series) if b[0]-a[0] == 25 and b[1] != a[1]]
        # Distances only to observed adjacent-label transitions, not inferred across gaps.
        distance = min((abs(f-frame)/25 for f in transitions), default=None)
        flags = {flag: sorted({r.get(flag) for r in annotations.get(video, {}).get(str(frame), [])})
                 for flag in ("blurred", "smoke", "reflection", "stainedlens", "undercoverage")}
        errors.append({"key": key, "h0": h0, "gt": gt, "nearest_observed_transition_seconds": distance,
            "annotation_flags": flags, "branches": {a: branches[a] for a in ("compact_repeat", "revised_phase")}})
    result = {"development_only": True, "new_api_calls": 0, "source_sha256": source_hashes,
        "all_32_phase_labels_match_source": True, "class_support": dict(support),
        "confusions": confusions, "per_class": per_class, "reviewer_argmax_diagnostic": seats,
        "image_citations": citations, "errors": errors,
        "limitations": ["Per-reviewer argmax is diagnostic, not a tested replacement selector.",
            "Ground truth is used only for posthoc analysis of sealed predictions.",
            "Observed transition distances exclude missing-label gaps; raw annotation correctness is not established."]}
    path = OUTPUT / "failure_cause_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("errors", "source_sha256")}, ensure_ascii=False))
    for error in errors:
        print(json.dumps({k: v for k, v in error.items() if k != "branches"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
