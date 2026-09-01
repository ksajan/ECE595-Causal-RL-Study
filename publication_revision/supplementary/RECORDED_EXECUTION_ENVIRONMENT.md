# Recorded Execution Environment

This file distinguishes the environment that produced the 30-seed GPU results
from the environment used to verify the archived JSON artifacts.

## Training environment

The environment was queried again on `x-indy-tasigpu3` on September 1, 2026:

- OS: Linux 6.17.0-20-generic, x86-64, glibc 2.39
- Python: 3.13.13
- uv: 0.11.13
- PyTorch: 2.13.0 (`torch.__version__` in artifacts: `2.13.0+cu130`)
- NumPy: 2.5.2
- SciPy: 1.18.1
- Gymnasium: 1.3.0
- CUDA runtime reported by PyTorch: 13.0
- cuDNN: 9.20.0
- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0

The complete `uv pip freeze` output is in
`recorded-gpu-requirements.txt`. Each main-study artifact also stores its Python,
library, CUDA, and GPU metadata.

## Verification environment

The hash-bound `pyproject.toml` and `uv.lock` are preserved exactly because the
result validators check their hashes. They predate the final GPU environment
and do not lock the versions above. On the packaging machine, `uv sync --frozen`
resolved Python 3.13.9, PyTorch 2.9.1, NumPy 2.3.5, SciPy 1.16.3, and Gymnasium
1.2.2. That environment passed all 50 tests, reproduced all five checked-in
summaries byte-for-byte, and regenerated every figure.

Therefore:

- use `uv sync --frozen` for artifact validation and figure regeneration;
- use the recorded freeze on a compatible CUDA 13 system for the closest full
  training rerun; and
- do not interpret byte-identical summary reproduction as byte-identical GPU
  retraining. PyTorch GPU kernels and hardware may still introduce numerical
  variation even with fixed seeds.

The unrelated `panda==0.3.1` dependency in the historical lock is unused by the
revision code. It is retained only because changing the hash-bound files would
break provenance.
