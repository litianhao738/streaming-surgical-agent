"""Opt-in prompt correction; retains the calibrated v2 parser and score formula."""
from .llm_judge_calibrated import CalibratedLLMJudge, PROMPT as LEGACY_PROMPT
from .llm_judge_calibrated import PROMPT_VERSION as LEGACY_VERSION
from .judge_selection import JudgePolicyCache
from .contracts import ModelConfig

VERSION = 'v2_low_strict'
PROMPT_VERSION = 'gsr_report_v1_judge_2_strict_evidence_1'
PROMPT = LEGACY_PROMPT.replace(
    'For omissions only, report_quote may be empty.',
    'For an omission, mark coverage as error and use an empty report_quote only for '
    'that coverage evidence entry. Every other check requires a nonempty exact '
    'substring of the candidate report. Do not mark another check as error solely '
    'because a fact is omitted; record that omission under coverage. '
    'Copy report_quote character-for-character, preserving case and punctuation; '
    'never paraphrase it or insert ellipses. Each errors entry must contain all '
    'three fields check, report_quote, and gt_fact in the same object.'
)


class StrictEvidenceCache:
    def __init__(self, cache):
        self.cache = cache

    def call(self, stage, request):
        if (stage != 'offline_evaluation' or request['prompt_version'] != LEGACY_VERSION
                or not request['prompt'].startswith(LEGACY_PROMPT)):
            raise ValueError('Unexpected legacy Judge request')
        request = {**request, 'prompt_version': PROMPT_VERSION,
                   'prompt': PROMPT + request['prompt'][len(LEGACY_PROMPT):]}
        return self.cache.call(stage, request)


class StrictEvidenceJudge(CalibratedLLMJudge):
    def __init__(self, config, cache):
        super().__init__(config, StrictEvidenceCache(JudgePolicyCache(cache)))

    def evaluate(self, report, truth):
        result = super().evaluate(report, truth)
        if 'judge_prompt_version' in result:
            result['judge_prompt_version'] = PROMPT_VERSION
        return result


def build_strict_judge(config, cache):
    model = ModelConfig(**config['judge'])
    if config.get('judge_version') != VERSION or (
            model.provider, model.model, model.max_tokens) != (
            'openrouter', 'openai/gpt-5.6-luna', 4096):
        raise ValueError('Strict Judge requires the v2/low model and token limit')
    return StrictEvidenceJudge(model, cache)
