# env/

Environment records for the fork. A changed lock is a new set of files, a new commit and a rerun
of the gates that use it. Every reading here is an engineering record from the home PC, not a
quotable result.

## training/

The training stack for CH-8 and every arm (STUDY v0.8 §7.3, stack pin [P]): Keras 3.14.0, hgq2
0.1.9 and quantizers 1.2.2 under JAX 0.9.2 with the CUDA 12 plugin, Python 3.12.13, for the
platform `x86_64-manylinux_2_28`, plus wandb 0.31.0 for the §12 logging.

| file | sha256 | what it is |
| --- | --- | --- |
| `training/requirements.in` | `725c20c008fae095fc4e3fb7f400b1dab5c6c37ceed8968bbd84a21980e5e983` | the pins given to the resolver (B0's seven lines plus `wandb==0.31.0`) |
| `training/constraints.txt` | `bcde70dbaf55c6f601f854ca83ab4e805a742fc7a1275247a0f35d12ce40b314` | B0's 35-package freeze, byte-identical to `preflight/env/lock/freeze.txt`; holds every B0 package at its B0 version |
| `training/requirements.lock.txt` | `c8725569731a4a768f90059b44b06a5e11c844ed18690ba85340b9a0c2ecdb0c` | hashed lock from `uv pip compile --generate-hashes`, 66 packages |
| `training/freeze.txt` | `6dc635728a209e08655e0241dbb165f025b2820b024b7c60f08ce7c2c20d850b` | `uv pip freeze` of `~/lab/venvs/cbl1016-jax2` built from the lock (66 lines) |
| `training/hgq2-0.1.9-installed-files.sha256` | `b38e2cc5686e96c24140b3730ac30d0f96dc041f8a8ce42043d67e3209d9cec9` | sha256 of every installed `hgq/` and `qkeras/` file of hgq2 0.1.9, sorted by path (unchanged since B0) |

**Venv of record:** `~/lab/venvs/cbl1016-jax2` on `kaipc` (Python 3.12.13). Every PC gate from
batch B1 on runs in it, and the runtime image (G05, batch B16) is compared with it.
`~/lab/venvs/cbl1016-jax` is left untouched as the evidence of `preflight/env/`.

### Why the lock changed (batch B1, 2026-10-18)

STUDY §12 requires W&B logging in every run, and the B0 lock (copied from
`preflight/env/lock/`) had no `wandb`. The B1 brief therefore added `wandb` with every B0 package
held at its exact B0 version:

| | B0 | B1 |
| --- | --- | --- |
| `requirements.in` | `eff0733a6ce46c19bc3da20ba80265a7da1fd50b64fd7edf19145097e02548c4` | `725c20c008fae095fc4e3fb7f400b1dab5c6c37ceed8968bbd84a21980e5e983` |
| `requirements.lock.txt` | `fb17d2d86ede5a39c1d77175673f35f5b99370f874a8925c786621c36caab6f2` | `c8725569731a4a768f90059b44b06a5e11c844ed18690ba85340b9a0c2ecdb0c` |
| `freeze.txt` | `bcde70dbaf55c6f601f854ca83ab4e805a742fc7a1275247a0f35d12ce40b314` (35 lines) | `6dc635728a209e08655e0241dbb165f025b2820b024b7c60f08ce7c2c20d850b` (66 lines) |
| venv | `~/lab/venvs/cbl1016-jax` | `~/lab/venvs/cbl1016-jax2` |

What was checked (uv 0.11.31, `/home/kaimoe/.local/bin/uv`; full record with every intermediate
file in `preflight/pc-setup/b1-wandb-lock/`, `commands.txt` there):

- The first compile, with `wandb` unpinned and `-c constraints.txt`, picked `wandb==0.31.0`. All
  35 B0 pins are present unchanged, and each of the 35 keeps exactly its B0 hash set (439 hashes).
  31 packages were added: wandb and its dependencies (`added.c1.txt` there).
- The second compile, with `wandb==0.31.0` written into `requirements.in`, gave the same lock body
  as the first.
- A third compile of the same `.in` without `-c` gave the same pins and hashes; only the `# via`
  comments differ. On 2026-10-18 the lock was therefore reproducible from `requirements.in`
  alone; the constraints file keeps it so if a newer numpy, h5py or tqdm appears later.
- The venv freeze equals the lock's pin list byte for byte, and `diff constraints.txt freeze.txt`
  shows additions only (`freeze-b0-to-b1.diff` there).
- `uv pip freeze` of the old `cbl1016-jax` hashes to `bcde70db…b314` before and after.
- One added package is unusual: wandb 0.31.0 declares `cwevals==0.0.1b1` as a hard dependency
  (PyPI metadata of wandb 0.31.0, `requires_dist`; cwevals is "The official Python library for the
  CoreWeave Evaluations API", author "Core Weave Evaluations <support@wandb.ai>"). It brings
  `httpx`, `anyio`, `distro` and `sniffio`. Recorded, not acted on.
- In the new venv with `KERAS_BACKEND=jax`: `jax.devices()` is `[CudaDevice(id=0)]` with default
  backend `gpu`, `keras` 3.14.0 reports backend `jax`, `hgq` imports as 0.1.9 from the venv, and
  `import wandb` under `WANDB_MODE=disabled` reports 0.31.0.

### Commands (on `kaipc`, from `env/training/`)

```
export PATH=$HOME/.local/bin:$PATH
uv pip compile requirements.in -c constraints.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes -o requirements.lock.txt
uv venv ~/lab/venvs/cbl1016-jax2 --python 3.12.13
uv pip sync --python ~/lab/venvs/cbl1016-jax2/bin/python --require-hashes requirements.lock.txt
uv pip freeze --python ~/lab/venvs/cbl1016-jax2/bin/python | diff - freeze.txt     # must print nothing
```

### B0 record (superseded lock, kept for the history)

B0 copied the lock byte for byte from `campaigns/2026-10-16-chang-bitlinear/preflight/env/lock/`,
which `preflight/env/README.md` (sha256
`5f8551f91bd4e7ab730024209c2e71e68a0d67935c5a6d544c45dad8bd988121`, lines 21-23 and 245-250)
documents and `preflight/env/MANIFEST.sha256` lists. On 2026-10-17 B0 checked that
`uv pip freeze` of `~/lab/venvs/cbl1016-jax` is byte-identical to that `freeze.txt`
(`bcde70db…b314`, 35 lines; sorted hash `d256bc3238b6eddaabf88b090da8dba1abd08264592f10a05311a19657167507`).

## Not here yet

The export environments, one locked environment per da4ml version (STUDY §9, §13 export row), and
the JAX runtime image Dockerfile (§13 runtime row) are later batches (B14, B16).
