"""Locate external lab tools for B17 integration tests."""
import os
from pathlib import Path

import pytest


def lab_root(*needed_tools):
    override = os.environ.get('BNJETTAG_LAB')
    code = Path(__file__).resolve().parents[2]
    if override:
        lab = Path(override).expanduser()
    elif code.name == 'code' and code.parent.parent.name == 'campaigns':
        lab = code.parents[2]
    else:
        roots = (Path('/Users/kaiyamaguchi/lab/bnjettag'),
                 Path('/home/kaimoe/lab/bnjettag'))
        lab = next((root for root in roots if root.is_dir()), roots[0])
    for name in needed_tools:
        tool = lab / 'tools' / name
        if not tool.is_file():
            pytest.skip('missing lab tool: %s (set BNJETTAG_LAB)' % tool)
    return lab
