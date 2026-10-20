"""M17: the config generator and the derived protocol (PREFLIGHT-PLAN section 2 row M17).

Standard library plus the backend-free lab modules `cbl.rng`, `cbl.select` and `cbl.wandb_log`
(NumPy and SciPy); no Keras, JAX, hgq or TensorFlow import anywhere in this package. Chang's
`chang/run_train.py` imports Keras at module top, so `gen.release` reads it by its syntax tree.

Modules:
  release   Chang's released training constants, read from chang/run_train.py (no import)
  arms      the STUDY section 5 arm catalogue: screen, PID and S4 templates, smoke, benchmark and
            the STUDY section 13 determinism record
  inputs    the decision-input record gen/inputs.json; inputs Kai or a later batch fixes are
            explicit and fail closed while unset
  configs   one JSON object per run (docs/interfaces.md section 5 and section 14), the configs/
            tree, its MANIFEST.json, and the launch check
  protocol  the protocol derived from the configs for the drift check (gate G59)

Command line (from the fork root):  python -I -B gen/cli.py generate | check | protocol
"""

STUDY_SHA256 = 'e4a6b89e0adbe3ab14b508a24335688fba48db3f801c4d1015812cc9dcd73d8e'  # snapshots/v0.15
STUDY_VERSION = 'v0.15'
PROTOCOL_REFERENCE_SHA256 = '8f48e716333a4ffb5ca9ae36012480bf6a3484aadc55545ecfdb00fd08d302c8'  # snapshots/v0.8
CAMPAIGN = 'chang-bitlinear-20261016'
