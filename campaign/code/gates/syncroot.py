"""Reject audit paths exposed to known two-way file synchronizers."""

import os
from pathlib import Path


def sync_roots():
    """Return canonical campaign roots plus nonempty CBL_SYNC_ROOTS entries."""
    roots = ('/home/kaimoe/lab/bnjettag', '/Users/kaiyamaguchi/lab/bnjettag')
    extra = tuple(p for p in os.environ.get('CBL_SYNC_ROOTS', '').split(':') if p)
    return tuple(Path(os.path.realpath(p)) for p in roots + extra)


def under_sync_root(path):
    """Return the matching root, or None, after resolving symbolic links."""
    path = Path(os.path.realpath(path))
    return next((root for root in sync_roots() if path == root or root in path.parents), None)


def require_unsynced(*paths):
    """Fail closed before creating or watching files in a synchronized tree."""
    for path in paths:
        root = under_sync_root(path)
        if root is not None:
            raise SystemExit(
                f'FAIL  harness fault (not a runner finding): {path} resolves under sync root {root}; '
                'use work, hard-link, temporary and audit paths outside all sync roots '
                '(including CBL_SYNC_ROOTS).'
            )
