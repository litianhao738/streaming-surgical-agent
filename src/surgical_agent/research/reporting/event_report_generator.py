import json
import re

from . import VERSION
from .contracts import ONTOLOGY_HASH, ReportInput, digest, strict_json

PROMPT_VERSION = VERSION + "_generator_1"
PROMPT = """Generate a concise intraoperative surgical event summary.
Use only the supplied canonical surgical state.
Describe the observed surgical interaction(s) and the current phase.
Use the supplied canonical instrument, verb, target and phase names.
Do not introduce any instrument, action, anatomical target, triplet, phase, diagnosis, complication or clinical claim that is not explicitly supplied.
Do not infer missing information. Do not provide medical advice.
Return JSON only, exactly one field: {"report": "..."}.
Write 1-2 sentences, at most 60 words. Do not include reasoning.
Use explicit instrument-action-target clauses and 'during <phase>' (underscores may be spaces).
For a null interaction say '<instrument> has no identified action or target'.
For an empty IVT list state that no interaction is identified; do not combine independent I/V/T labels into new triplets.
Canonical surgical state:
"""


def parse_report(raw):
    try:
        obj = strict_json(raw)
    except (ValueError, TypeError):
        return "", False, "invalid_json"
    text = obj.get("report", "") if isinstance(obj, dict) else ""
    if not isinstance(text, str):
        return "", False, "invalid_report_type"
    valid = set(obj) == {"report"} and 0 < len(re.findall(r"\S+", text)) <= 60
    return text.strip(), valid, "valid" if valid else "invalid_schema_or_length"


class ReportGenerator:
    def __init__(self, config, cache):
        self.config, self.cache = config, cache

    def generate(self, source: ReportInput):
        if not isinstance(source, ReportInput):
            raise TypeError("ReportInput required; no raw frame/evaluation payload accepted")
        state = source.state()
        request = {"model_config": self.config.to_dict(), "prompt_version": PROMPT_VERSION,
                   "ontology_hash": ONTOLOGY_HASH, "structured_prediction_hash": digest(source.appendix()),
                   "prompt": PROMPT + json.dumps(state, sort_keys=True)}
        response, key, hit = self.cache.call("report_generation", request)
        text, valid, status = parse_report(response["text"])
        return {"video_id": source.video_id, "frame_id": source.frame_id, "source_prediction": source.source_prediction,
                "structured_prediction": state, "canonical_appendix": source.appendix(), "report": text,
                "raw_response": response["text"], "format_valid": valid, "generation_status": status,
                "generator_provider": self.config.provider, "generator_model": self.config.model,
                "generation_parameters": self.config.to_dict(), "prompt_version": PROMPT_VERSION,
                "ontology_hash": ONTOLOGY_HASH, "cache_key": key, "cache_hit": hit, "mock": self.cache.mock}
