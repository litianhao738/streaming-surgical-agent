"""Offline audit of saved review requests; no credentials or inference calls.

Requires tiktoken on PYTHONPATH. Both encodings are text-only proxies, not
verified tokenizers for the five deployed models or provider usage counts.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from surgical_agent.research.verification.compact_prompt import (
    PROFILE,
    compact_review_wire,
)


def audit(source):
    import tiktoken

    encoders = {name: tiktoken.get_encoding(name) for name in ("cl100k_base", "o200k_base")}
    rows = []
    for branch in ("graph", "phase"):
        paths = sorted((source / "branches" / branch / "calls").glob(f"*_{branch}_review_*/request.json"))
        if not paths:
            raise ValueError(f"no {branch} review requests found")
        for path in paths:
            body = json.loads(path.read_text(encoding="utf-8"))
            before = body["messages"][0]["content"][0]["text"]
            revised = None
            counts = {}
            for name, encoder in encoders.items():
                candidate = compact_review_wire(body, branch, count_tokens=lambda s, enc=encoder: len(enc.encode(s)))
                after = candidate["messages"][0]["content"][0]["text"]
                if revised is not None and revised != candidate:
                    raise AssertionError("token counter changed the candidate")
                revised = candidate
                counts[name] = {"before": len(encoder.encode(before)), "after": len(encoder.encode(after))}
            restored = deepcopy(revised)
            restored["messages"][0]["content"][0]["text"] = before
            if restored != body:
                raise AssertionError("non-prompt request change")
            old_packet, new_packet = json.loads(before), json.loads(after)
            allowed = {"instructions"}
            if branch == "graph":
                allowed |= {"proposition_semantics", "output_contract_clarification"}
            if ({k: v for k, v in old_packet.items() if k not in allowed}
                    != {k: v for k, v in new_packet.items() if k not in allowed}):
                raise AssertionError("evidence or schema changed")
            rows.append({"source": str(path.relative_to(source)), "branch": branch,
                         "model": body["model"], "text_tokens_proxy": counts,
                         "characters": {"before": len(before), "after": len(after)},
                         "before_text_sha256": hashlib.sha256(before.encode()).hexdigest(),
                         "after_text_sha256": hashlib.sha256(after.encode()).hexdigest()})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["branch"]].append(row)
    summary = {}
    for branch, group in grouped.items():
        summary[branch] = {"requests": len(group), "text_tokens_proxy": {}}
        for name in encoders:
            before = sum(r["text_tokens_proxy"][name]["before"] for r in group)
            after = sum(r["text_tokens_proxy"][name]["after"] for r in group)
            summary[branch]["text_tokens_proxy"][name] = {
                "before": before, "after": after, "reduction_percent": round(100 * (1 - after / before), 2),
                "min_tokens_saved_per_request": min(r["text_tokens_proxy"][name]["before"]
                                                    - r["text_tokens_proxy"][name]["after"] for r in group)}
    return {"profile": PROFILE, "status": "OFFLINE_CANDIDATE_ONLY", "source": str(source.resolve()),
            "inference_calls": 0, "schema_images_transport_unchanged": True,
            "provider_token_counts_verified": False, "quality_improvement_verified": False,
            "limitations": "Text proxy counts only; excludes image tokens and provider framing. "
                           "Exact deployed-model token counts and quality require separate validation.",
            "summary": summary, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/preflight/parallel_phase_eight_20260909_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new report path required")
    report = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
