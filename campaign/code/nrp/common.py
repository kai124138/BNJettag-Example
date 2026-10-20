"""Shared, dependency-free contracts. No cluster access at import time."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re

CAMPAIGN = 'chang-bitlinear-20261016'
ROOT = '/data/cbl1016'
PRODUCTS = ('NVIDIA-GeForce-RTX-4090', 'NVIDIA-L40S', 'NVIDIA-L40',
            'NVIDIA-GeForce-RTX-3090', 'NVIDIA-A10')
IMAGE = 'buildpack-deps@sha256:17923645546d74acef88727bce62c23bbaed953fc7097c41a8b777bba3ca98de'
PYTHON = ROOT + '/runtime/venv-c8725569731a/bin/python'
STAGES = ('S1', 'S3', 'S4', 'smoke', 'bench', 'det')


class Refused(ValueError):
    pass


def need(ok, why):
    if not ok:
        raise Refused(why)


def encoded(obj):
    return (json.dumps(obj, sort_keys=True, indent=1, allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write_once(path, obj):
    path = Path(path)
    raw = encoded(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == raw, f'immutable output differs: {path}')
    else:
        with path.open('xb') as f:
            f.write(raw)
    return path


def atom(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_bytes(encoded(obj))
    tmp.replace(path)


def hexstr(value, n=64):
    need(isinstance(value, str) and re.fullmatch('[0-9a-f]{%d}' % n, value), f'expected {n}-hex identity')
    return value


def token(value):
    need(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,190}', value), 'unsafe name')
    return value


def pvc(value):
    need(isinstance(value, str) and value.startswith(ROOT + '/') and
         '..' not in value.split('/') and str(PurePosixPath(value)) == value, 'path outside campaign PVC root')
    return value


def relative(value):
    need(isinstance(value, str) and not value.startswith('/') and '..' not in value.split('/') and
         str(PurePosixPath(value)) == value, 'unsafe relative path')
    return value


def positive(value, name, integer=False):
    need(type(value) in ((int,) if integer else (int, float)) and math.isfinite(value) and value > 0,
         f'{name} must be positive' + (' integer' if integer else ''))
    return value


def load_tool(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def init_root(product, archive_kind):
    need(product in PRODUCTS and archive_kind in ('record', 'smoke'), 'invalid draw product/archive')
    return f'{ROOT}/initial/{product}/{archive_kind}'


def archive(kind):
    need(kind in ('record', 'smoke'), 'invalid archive kind')
    return ROOT + ('/data-smoke' if kind == 'smoke' else '/data') + '/150c-train.h5'
