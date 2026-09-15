"""Explicit version selection; v1 remains available for historical reproduction."""
from .contracts import ModelConfig
from .llm_judge import LLMJudge
from .llm_judge_calibrated import CalibratedLLMJudge


class JudgePolicyCache:
    def __init__(self, cache):
        self.cache = cache
        self.mock = cache.mock

    def call(self, stage, request):
        if stage != "offline_evaluation":
            raise ValueError("judge cache cannot dispatch generation")
        return self.cache.call(stage, {**request,
            "wire_policy": {"reasoning": {"effort": "low"}, "response_format": {"type": "json_object"}, "stream": False},
            "omit_temperature": True, "provider_tag": "openai"})


def build_judge(config, cache):
    model = ModelConfig(**config["judge"])
    version = config.get("judge_version", "v1")
    if version == "v1":
        return LLMJudge(model, cache)
    if version == "v2_low":
        if (model.provider, model.model, model.max_tokens) != ("openrouter", "openai/gpt-5.6-luna", 4096):
            raise ValueError("v2_low calibration requires its frozen model and token limit")
        return CalibratedLLMJudge(model, JudgePolicyCache(cache))
    raise ValueError("unknown judge version")
