"""Offline checks of sync-root refusals; no real archives or gate runs."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gates'))
import fileaudit  # noqa: E402
import g51_heldout  # noqa: E402
import runnergate  # noqa: E402
from syncroot import sync_roots, under_sync_root  # noqa: E402

ROOTS = ('/home/kaimoe/lab/bnjettag', '/Users/kaiyamaguchi/lab/bnjettag')


@pytest.fixture(autouse=True)
def clean_sync_env(monkeypatch):
    monkeypatch.delenv('CBL_SYNC_ROOTS', raising=False)


@pytest.mark.parametrize('root', ROOTS)
@pytest.mark.parametrize('suffix', ['', '/campaigns/g51/work'])
def test_known_roots(root, suffix):
    assert under_sync_root(root + suffix) == Path(root).resolve()
    assert Path(root).resolve() in sync_roots()


@pytest.mark.parametrize('path', ['/tmp/cbl-g51-work', ROOTS[0] + '-scratch/work',
                                  ROOTS[1] + '-scratch/work'])
def test_outside_roots(path):
    assert under_sync_root(path) is None


def test_symlink_resolves_inside(tmp_path):
    alias = tmp_path / 'alias'
    alias.symlink_to(ROOTS[1], target_is_directory=True)
    assert under_sync_root(alias / 'campaigns/work') == Path(ROOTS[1]).resolve()


def test_environment_extends_roots(monkeypatch, tmp_path):
    extra = (tmp_path / 'one', tmp_path / 'two')
    monkeypatch.setenv('CBL_SYNC_ROOTS', f':{extra[0]}::{extra[1]}:')
    assert sync_roots() == tuple(Path(p).resolve() for p in (*ROOTS, *extra))
    for root in extra:
        assert under_sync_root(root / 'work') == root.resolve()
    assert under_sync_root(tmp_path / 'outside') is None


@pytest.mark.parametrize('root', ROOTS)
def test_g51_refuses_synced_work(root):
    with pytest.raises(SystemExit, match=r'FAIL  harness fault \(not a runner finding\).*sync root'):
        g51_heldout.check_work_paths(Path(root) / 'campaigns/g51/work')


@pytest.mark.parametrize('argument', ['hardlink_dir', 'temp_dir'])
def test_g51_refuses_other_synced_directories(argument, tmp_path):
    with pytest.raises(SystemExit, match='harness fault'):
        g51_heldout.check_work_paths(tmp_path, **{argument: Path(ROOTS[0]) / 'unsafe'})


def test_g51_checks_effective_temp_directory(monkeypatch, tmp_path):
    import tempfile

    monkeypatch.setattr(tempfile, 'tempdir', ROOTS[1])
    with pytest.raises(SystemExit, match='harness fault'):
        g51_heldout.check_work_paths(tmp_path)


def test_g51_accepts_unsynced_directories(tmp_path):
    assert g51_heldout.check_work_paths(tmp_path, hardlink_dir=tmp_path / 'links',
                                      temp_dir=tmp_path / 'tmp') is None


@pytest.mark.parametrize('unsafe', ['watch', 'cwd'])
def test_audit_refuses_before_inotify_or_child(monkeypatch, tmp_path, unsafe):
    def unexpected(*args, **kwargs):
        pytest.fail('audit started before rejecting unsafe paths')

    monkeypatch.setattr(fileaudit, 'Inotify', unexpected)
    monkeypatch.setattr(fileaudit.subprocess, 'Popen', unexpected)
    watched = Path(ROOTS[0]) / 'watched' if unsafe == 'watch' else tmp_path / 'watched'
    cwd = ROOTS[1] if unsafe == 'cwd' else tmp_path
    with pytest.raises(SystemExit, match='harness fault'):
        fileaudit.run_audited(['unused'], watches=iter([(watched, 'placeholder')]), cwd=cwd)


def test_placeholder_refuses_before_creation(monkeypatch, tmp_path):
    monkeypatch.setenv('CBL_SYNC_ROOTS', str(tmp_path))
    work = tmp_path / 'work'
    with pytest.raises(SystemExit, match='harness fault'):
        runnergate.placeholder_dir(work, tmp_path / 'unused.h5')
    assert not work.exists()
