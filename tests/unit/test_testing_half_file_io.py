import pytest
from scripts.testing_half_file_io import retry_local_writes
from surgical_agent.artifacts import manifest


def test_transient_local_permission_error_retries_write_only(tmp_path, monkeypatch):
    original = manifest.atomic_write_text
    attempts = []
    def blocked(path, text):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError('sharing violation')
        return original(path, text)
    monkeypatch.setattr(manifest, 'atomic_write_text', blocked)
    monkeypatch.setattr('scripts.testing_half_file_io.time.sleep', lambda _: None)
    with retry_local_writes():
        manifest.atomic_write_json(tmp_path/'record.json', {'saved': True})
    assert len(attempts) == 3 and (tmp_path/'record.json').exists()
    assert manifest.atomic_write_text is blocked


def test_other_io_error_not_retried(monkeypatch):
    attempts = []
    def broken(*args):
        attempts.append(1)
        raise OSError('disk full')
    monkeypatch.setattr(manifest, 'atomic_write_text', broken)
    with retry_local_writes(), pytest.raises(OSError, match='disk full'):
        manifest.atomic_write_text('unused', 'data')
    assert len(attempts) == 1
