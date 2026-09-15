from scripts.testing_half_progress import PipelineProgress
from scripts import testing_half_progress as module


def test_main_console_shows_tqdm_and_stops_monitor_thread(tmp_path, capsys):
    with PipelineProgress(tmp_path, interval=.01) as progress:
        print('startup check')
        print('Completed 320/9058 targets')
    output = capsys.readouterr()
    assert 'Startup checks' in output.err and '8578' in output.err
    assert 'Reports H0/FULL' not in output.err and 'Report scoring' not in output.err
    assert '\x1b[A' not in output.err  # No cursor-up control for stacked bars.
    assert 'Completed' not in output.err + output.out
    assert 'Completed' in tmp_path.with_name(tmp_path.name+'.console.log').read_text(encoding='utf-8')
    assert not progress.thread.is_alive()


def test_stage_change_reuses_one_bar(tmp_path, monkeypatch):
    state = module.snapshot(tmp_path)
    monkeypatch.setattr(module, 'snapshot', lambda _: state)
    with PipelineProgress(tmp_path, interval=60) as progress:
        original = progress.bar
        state['state'] = 'REPORT_GENERATION'
        progress.refresh()
        assert progress.bar is original and progress.bar.total == 17156
        state['state'] = 'RULE_JUDGE_SCORING'
        progress.refresh()
        assert progress.bar is original and progress.bar.total == 15646
