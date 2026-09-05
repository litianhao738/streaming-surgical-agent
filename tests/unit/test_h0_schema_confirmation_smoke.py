from scripts import run_h0_prompt_refinement_smoke as original
from scripts import run_h0_schema_confirmation_smoke as confirmation


def test_confirmation_does_not_change_original_runner_configuration():
    assert original.ARMS == ("baseline", "schema_only", "tuned")
    assert original.POSITIONS == (9, 12, 15, 18)
    assert original.CONCURRENCY == 3
    assert confirmation.prior is not original
    assert confirmation.prior.ARMS == ("baseline", "schema_only")
    assert confirmation.prior.CONCURRENCY == 4
    assert original.prepare is not confirmation.prepare
    assert original.analyze is not confirmation.analyze


def test_confirmation_positions_are_fixed_and_disjoint_from_prior_cohorts():
    prior_positions = set(range(8)) | {8, 14} | {9, 12, 15, 18}
    assert not set(confirmation.POSITIONS) & prior_positions
    assert set(confirmation.POSITIONS) | prior_positions == set(range(20))
    source = {"selection": {v: {"selected_targets": list(range(20))}
                            for v in confirmation.study.VIDEOS}}
    samples = confirmation.selected_samples(source)
    assert len(samples) == len({(s["video_id"], s["frame_id"]) for s in samples}) == 24
