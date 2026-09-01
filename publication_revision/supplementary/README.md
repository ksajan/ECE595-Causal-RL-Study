# Reproducibility Instructions

This archive supports the revised paper, **Auditing Counterfactual Data
Augmentation in Reinforcement Learning**. It contains the exact source files,
frozen manifests, per-seed JSON artifacts, statistical summaries, plotting code,
and tests used for the reported CartPole audit.

The paper does **not** claim a successful learned-CTRL reproduction. The main
publication result is a simulator-only, paired 30-seed comparison of factual
noise reuse against a sibling-sharing-matched fresh-noise control. The attempted
learned BiCoGAN failed its model-quality gates and was not used downstream.

## Requirements

- Linux or another platform supported by PyTorch
- Python 3.13, selected by `.python-version`
- [`uv`](https://docs.astral.sh/uv/)
- An NVIDIA GPU is strongly recommended for full training reruns; artifact
  validation, statistics, tests, and plotting can run on CPU

The recorded 30-seed follow-up used Python 3.13.13, PyTorch 2.13.0+cu130,
NumPy 2.5.2, SciPy 1.18.1, Gymnasium 1.3.0, CUDA 13.0, and RTX 5090 GPUs.
Exact software metadata is stored in every `coupling_seed_*.json` file.

## Artifact-verification setup

Run all commands from this `supplementary` directory:

```bash
uv sync --frozen
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The supplied `pyproject.toml` and `uv.lock` are retained byte-for-byte because
their SHA-256 hashes are recorded in the result artifacts. They define the
verified artifact-analysis environment, not the exact GPU training environment.
See `RECORDED_EXECUTION_ENVIRONMENT.md` and
`recorded-gpu-requirements.txt` for the captured training environment.

## Quick verification

Run the complete test suite:

```bash
PYTHONPATH=. uv run --with pytest pytest -q
```

Recompute the four principal summaries from the raw per-seed artifacts:

```bash
uv run python -m scripts.revision.summarize_coupling_control \
  results/revision/coupling_control_final \
  --output /tmp/coupling_summary.json

uv run python -m scripts.revision.summarize_oracle_confirmatory \
  results/revision/oracle_confirmatory_final \
  --output /tmp/oracle_summary.json

uv run python -m scripts.revision.summarize_model_gate \
  results/revision/model_gate_final \
  --output /tmp/model_gate_summary.json

uv run python -m scripts.revision.summarize_protocol_sensitivity \
  results/revision/protocol_sensitivity_final \
  --output /tmp/protocol_summary.json
```

The summarizers reject missing or duplicate seeds, schema mismatches, manifest
changes, source-hash changes, configuration drift, and invalid arm sets. Compare
the regenerated files with the checked-in `summary.json` files:

```bash
cmp /tmp/coupling_summary.json results/revision/coupling_control_final/summary.json
cmp /tmp/oracle_summary.json results/revision/oracle_confirmatory_final/summary.json
cmp /tmp/model_gate_summary.json results/revision/model_gate_final/summary.json
cmp /tmp/protocol_summary.json results/revision/protocol_sensitivity_final/summary.json
```

Recompute the online learnability-control summary:

```bash
uv run python -m scripts.revision.summarize_cartpole_sanity \
  results/revision/online_sanity_10_final \
  --output /tmp/online_sanity_summary.json
```

## Regenerate figures

```bash
uv run python -m scripts.revision.plot_coupling_control \
  --summary results/revision/coupling_control_final/summary.json \
  --output-dir /tmp/coupling_figures

uv run python -m scripts.revision.plot_oracle_confirmatory \
  --oracle-summary results/revision/oracle_confirmatory_final/summary.json \
  --model-gate-summary results/revision/model_gate_final/summary.json \
  --output-dir /tmp/oracle_and_model_figures
```

Each plotting command writes PDF and PNG versions. The paper uses the five-arm
seed distribution and paired oracle-minus-fresh-shared delta figures. The model
gate figure is supplementary evidence.

## Full matched 30-seed rerun

The frozen design is in
`scripts/revision/coupling_control_manifest.json`. Run one seed with:

```bash
uv run python -m scripts.revision.cartpole_coupling_control \
  --seed 1030 \
  --output-dir results/rerun_coupling
```

Run all 30 seeds sequentially with:

```bash
for seed in $(seq 1030 1059); do
  uv run python -m scripts.revision.cartpole_coupling_control \
    --seed "$seed" \
    --output-dir results/rerun_coupling
done
```

Seeds may be distributed across machines because each process writes one unique
JSON artifact. Collect all 30 files into one directory before running the
summarizer. Do not change the manifest: the runner verifies its fixed SHA-256.
For the closest environment match, first create a Python 3.13.13 environment on
a compatible CUDA 13 host and install `recorded-gpu-requirements.txt`. The
historical `uv.lock` is intentionally not represented as an exact lock for the
reported GPU runs.

## Diagnostic reruns

Run one learned-model gate seed:

```bash
uv run python -m scripts.revision.cartpole_model_gate \
  --seed 960 \
  --output-dir results/rerun_model_gate
```

Run one protocol-sensitivity seed:

```bash
uv run python -m scripts.revision.cartpole_protocol_sensitivity \
  --seed 970 \
  --output-dir results/rerun_protocol
```

Run one online learnability-control seed:

```bash
uv run python -m scripts.revision.cartpole_sanity \
  --seed 610 \
  --train-episodes 800 \
  --eval-episodes 100 \
  --validation-episodes 20 \
  --validation-seed-base 700000 \
  --test-seed-base 800000 \
  --noise-semantics process \
  --output-dir results/rerun_online/seed_610
```

## Reported results

The authoritative numerical account is
`results/revision/PUBLICATION_RESULTS_REPORT.md`. The matched follow-up summary
is `results/revision/coupling_control_final/summary.json`.

Key paired result:

- Oracle CF minus fresh-shared synthetic, clean: -62.78 return points,
  95% paired bootstrap interval [-111.31, -14.56], 30 training seeds.
- Oracle CF minus fresh-shared synthetic, process noise: -2.05 points; its
  paired 90% bootstrap interval lies inside the study-defined [-5, 5] practical
  equivalence band.

The result is scoped to this simulator, dataset, stabilized learner, and 50:50
real/synthetic sampling rule. It does not establish that counterfactual methods
are generally harmful. LunarLander, MuJoCo, SAC, and D4RL pilots are deliberately
excluded because they did not provide valid, adequately powered CF comparisons.

Fresh-shared and fresh-independent use the same initial random-number stream but
consume it at different rates. Exactly one synthetic next state per seed is
therefore identical between these pools, about 0.003% of either pool. This
negligible overlap is disclosed in the paper and result report; distinct stream
offsets would be preferable in a future rerun.

## Directory map

- `scripts/revision/`: experiment, validation, statistics, and plotting code
- `tests/`: 50 focused implementation and artifact tests
- `results/revision/coupling_control_final/`: main 30-seed follow-up
- `results/revision/oracle_confirmatory_final/`: initial 30-seed oracle study
- `results/revision/model_gate_final/`: five learned-model development gates
- `results/revision/protocol_sensitivity_final/`: ten protocol-sensitivity seeds
- `results/revision/online_sanity_10_final/`: ten online-control seeds
- `REVISION_EXPERIMENT_PROTOCOL.md`: prospectively frozen local protocol record
- `RECORDED_EXECUTION_ENVIRONMENT.md`: training versus verification environment
- `recorded-gpu-requirements.txt`: captured GPU-host package freeze

Model checkpoints are not needed to verify the published statistics and are
omitted to keep the archive small. All reported values are recoverable from the
included seed-level JSON files.
