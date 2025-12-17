#!/usr/bin/env bash
set -euo pipefail

# Quick driver for CartPole-SD reproduction.
# Assumes deps are installed and run from the code/ directory.

DATASET=${1:-data/SD_dataset_clean.pt}
BICOGAN_DIR=${2:-results/cartpole/bicogan}
D3QN_REAL_DIR=${3:-results/cartpole/d3qn_real}
D3QN_CF_DIR=${4:-results/cartpole/d3qn_cf}
RAINBOW_DIR=${5:-results/cartpole/rainbow}
CF_FRAC=${6:-0.05}

echo "[1/6] dataset -> ${DATASET}"
python scripts/run_ctrl.py dataset --episodes 250 --horizon 200 --output "${DATASET}"

echo "[2/6] bicogan -> ${BICOGAN_DIR}"
python scripts/run_ctrl.py train-bicogan --dataset-path "${DATASET}" --output-dir "${BICOGAN_DIR}"

echo "[3/6] d3qn real -> ${D3QN_REAL_DIR}"
python scripts/run_ctrl.py train-d3qn --dataset-path "${DATASET}" --output-dir "${D3QN_REAL_DIR}"

echo "[4/6] d3qn cf (cf_sample_frac=${CF_FRAC}) -> ${D3QN_CF_DIR}"
python scripts/run_ctrl.py train-d3qn --dataset-path "${DATASET}" --use-cf --cf-sample-frac "${CF_FRAC}" --bicogan-dir "${BICOGAN_DIR}" --output-dir "${D3QN_CF_DIR}"

echo "[5/6] rainbow -> ${RAINBOW_DIR}"
python scripts/run_alt_algos.py rainbow --dataset-path "${DATASET}" --output-dir "${RAINBOW_DIR}"

echo "[6/6] plots -> results/plots"
python scripts/plot_results.py --rainbow-metrics "${RAINBOW_DIR}/metrics.json" --d3qn-real "${D3QN_REAL_DIR}/metrics.json" --d3qn-cf "${D3QN_CF_DIR}/metrics.json" --output-dir results/plots

echo "Done. Key outputs in results/plots/."
