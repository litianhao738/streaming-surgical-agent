# OpenRouter prompt v3 fixed-20 A/B decision

Date: 2026-09-02 (Asia/Shanghai)

This is a prompt-selection diagnostic, not a paper result. Both prompts used
the exact first 20 points of the existing frozen 80-frame Validation manifest:
VID110 frames 4026 through 4501 at the manifest's fixed 25-frame spacing. No
sample was redrawn or replaced.

- Frozen manifest SHA-256: `94cb9d1347c14394be9d516a81f07ccd3295305d1c76d85b80a29fffb179b8bf`
- Baseline v2 prompt SHA-256: `9acd4c97e9947e1f72462d057efcce0367914c82b0546475dbb92c5832b0a1c3`
- Final compact v3 candidate SHA-256: `645764e4ae0bb3b258cf15ef3919949154768d3f06618413174883acd07727a8`
- Provider/model: OpenRouter / `openai/gpt-5.6-sol`
- Response schema: `joint_perception_reliability_compact_v2`
- Reasoning effort: `none`; measured reasoning tokens: 0
- Images and generation parameters were identical across variants.

Recognition metrics below use the 12 points successfully returned by both
prompts. API success reports all 20 attempted frozen points.

| Prompt | API success | Instrument F1 | Verb F1 | Target F1 | IVT F1 | Phase accuracy | Five-task macro | Mean new-call latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| production v2 | 20/20 | 0.6389 | 0.1389 | 0.1389 | 0.0972 | 0.5833 | 0.3194 | 22.03 s |
| compact v3 candidate | 12/20 | 0.4861 | 0.1306 | 0.1111 | 0.0423 | 0.5000 | 0.2540 | 21.99 s |
| v3 minus v2 | -8 points | -0.1528 | -0.0083 | -0.0278 | -0.0549 | -0.0833 | -0.0654 | -0.04 s |

The compact v3 candidate had eight `content_moderation` failures. Its latency
was effectively unchanged, while every recognition metric and API success rate
was worse on this fixed pilot. Therefore the candidate is rejected, the 80-point
rollout is not run, and the v2 initial-perception prompt remains production.

The separate field-targeted verifier prompt is still upgraded because it only
rechecks Gate-flagged fields and does not replace the full initial prediction.

## Targeted-verifier follow-up

A six-frame OpenRouter diagnostic used VID110 frames 4076, 4326, 4401, 4676,
6976, and 7251. The Learned Gate verified 1/6 frames; the deterministic
reliability Gate verified 6/6. Both produced the same aggregate recognition
scores: Instrument F1 0.8889, Verb F1 0.0000, Target F1 0.1944, IVT F1 0.0000,
and Phase accuracy 0.8333. The rule Gate made one IVT repair, but it did not
improve GT agreement. This isolates the current bottleneck to visual
interaction recognition and candidate quality rather than verification rate.

The diagnostic also exposed and fixed a real parser defect: confidence-ordered
multi-label selections such as `[10, 0]` are now normalized to canonical
ascending order before constructing the prediction. A short null-aware v3
targeted prompt and coupled-field expansion were tested on frames 4076 and
4326; neither improved the output and mean verification latency rose from about
20--21 seconds to about 27 seconds. That candidate was rejected and removed.
Production therefore remains initial prompt v2 plus targeted verifier v2, and
no 80-frame rerun is justified by these changes.
