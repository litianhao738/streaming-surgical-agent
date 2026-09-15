"""Live tqdm progress in the complete pipeline's own console."""
from contextlib import redirect_stdout
from pathlib import Path
import sys
import shutil
import threading

from tqdm import tqdm
from scripts.watch_testing_half import snapshot


class PipelineProgress:
    def __init__(self, out, interval=2, stage=None):
        self.out, self.interval = Path(out), interval
        self.stage = stage
        self.stop = threading.Event()
        self.bar = None
        self.active = None

    def refresh(self):
        value = snapshot(self.out)
        stages = value['progress']
        if value['state'] in ('RULE_JUDGE_SCORING', 'COMPLETE') or stages['Report scoring']['done']:
            name, label = 'Report scoring', 'Report scoring'
        elif value['state'] == 'REPORT_GENERATION' or stages['Reports H0/FULL']['done']:
            name, label = 'Reports H0/FULL', 'Reports H0/FULL'
        else:
            name, label = 'Gate+Tracker', 'Pipeline'
        if self.stage:
            label = 'Resume checks'
        elif value['state'] == 'STARTUP_CHECKS':
            label = 'Startup checks'
        elif value['state'] == 'STOPPED':
            label += ' stopped'
        progress = stages[name]
        width = max(30, min(100, shutil.get_terminal_size((80, 24)).columns - 1))
        if self.bar is None:
            self.bar = tqdm(total=progress['total'], initial=progress['done'], desc=label,
                position=0, ncols=width, ascii=True, file=self.stream,
                bar_format='{desc}: {n_fmt}/{total_fmt} |{bar:12}| {percentage:3.0f}% [{elapsed}<{remaining}]')
        else:
            if name != self.active:
                self.bar.reset(total=progress['total'])
            self.bar.ncols = width
            self.bar.set_description_str(label, refresh=False)
            self.bar.update(progress['done']-self.bar.n)
            self.bar.refresh()
        self.active = name

    def _watch(self):
        while not self.stop.wait(self.interval):
            try:
                self.refresh()
            except (OSError, ValueError, KeyError):
                # A transient read cannot interrupt or mutate paid execution.
                continue

    def __enter__(self):
        self.stream = sys.stderr
        self.refresh()
        # Routine prints go to a file; the console contains only the single tqdm bar.
        self.log = self.out.with_name(self.out.name+'.console.log').open('a', encoding='utf-8', buffering=1)
        self.redirect = redirect_stdout(self.log)
        self.redirect.__enter__()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        self.redirect.__exit__(*args)
        self.log.close()
        try:
            self.refresh()
        finally:
            if self.bar is not None:
                self.bar.close()
