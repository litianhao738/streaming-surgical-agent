"""Retry only local atomic writes when Windows temporarily denies file access."""
from contextlib import contextmanager
from pathlib import Path
import time

from surgical_agent.artifacts import manifest


@contextmanager
def retry_local_writes():
    original = manifest.atomic_write_text
    def write(path, content):
        for attempt in range(21):
            try:
                return original(path, content)
            except PermissionError as exc:
                if attempt == 20:
                    raise PermissionError(f'Local write still blocked: {Path(path)}') from exc
                time.sleep(min(.05*(attempt+1), .25))
    manifest.atomic_write_text = write
    try:
        yield
    finally:
        manifest.atomic_write_text = original
