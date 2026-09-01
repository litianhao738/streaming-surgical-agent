from __future__ import annotations

from io import StringIO

from surgical_agent.cli.progress import progress_bar


def test_progress_bar_updates_one_terminal_line() -> None:
    """Replacing carriage-return refreshes with per-item prints would flood logs."""

    sink = StringIO()
    with progress_bar(
        total=2,
        description="Tracker train",
        unit="batch",
        enabled=True,
        file=sink,
    ) as bar:
        bar.update()
        bar.set_postfix(loss="1.2500", refresh=False)
        bar.update()

    rendered = sink.getvalue()
    assert "Tracker train" in rendered
    assert "2/2" in rendered
    assert "loss=1.2500" in rendered
    assert rendered.count("\n") == 1
