# ECE595 Causal RL Deep Dive
Offline CartPole CTRL study with counterfactual-augmented D3QN (with CQL), plus Rainbow baseline. Includes data, scripts, plots, and demo videos.

## Quickstart
1) Install deps (Python 3.10+):
```bash
pip install torch gymnasium numpy matplotlib seaborn tqdm scikit-learn pandas
```
2) Build dataset:
```bash
python scripts/train.py dataset --episodes 250 --horizon 200 --output data/SD_dataset_clean.pt
```
3) Train:
```bash
# BiCoGAN (CF generator)
python scripts/train.py bicogan --dataset-path data/SD_dataset_clean.pt
# D3QN real-only
python scripts/train.py d3qn --dataset-path data/SD_dataset_clean.pt
# D3QN + CF (example)
python scripts/train.py d3qn --dataset-path data/SD_dataset_clean.pt --use-cf --cf-k 1 \
  --cf-sample-frac 0.10 --cf-action-noise-std 0.05 --cf-use-env-step \
  --bicogan-dir results/cartpole/bicogan/<ts>
```
4) Evaluate (clean vs CTRL noisy):
```bash
python scripts/eval.py --algo d3qn --run-dir results/cartpole/d3qn_cf/<run> --episodes 50
python scripts/eval.py --algo d3qn --run-dir results/cartpole/d3qn_cf/<run> --episodes 50 --use-ctrl-env
```
5) Plots (bar + losses):
```bash
python scripts/plot_results.py \
  --rainbow-metrics results/cartpole/rainbow_reboot/20251208-001041/metrics.json \
  --d3qn-real results/cartpole/d3qn_real/20251208-011400/20251208-011321/metrics.json \
  --d3qn-cf results/cartpole/d3qn_cf/20251208-031646/metrics.json \
  --output-dir results/plots_cf_final \
  --eval-override 'Rainbow=results/eval/rainbow/20251208-003300/rainbow.json' \
  --eval-override 'D3QN real=results/eval/d3qn/20251208-011553/d3qn.json' \
  --eval-override 'D3QN CF=results/eval/d3qn/20251208-031725/d3qn.json'
```
6) Record a demo:
```bash
python scripts/infer.py --algo d3qn \
  --model-path results/cartpole/d3qn_cf/20251208-031646/q_net.pt \
  --dataset-path data/SD_dataset_clean.pt --episodes 2 --record
```

## Key Results (50-episode evals)
| Model | Clean mean ± std | CTRL noisy mean ± std | Eval refs |
|---|---|---|---|
| D3QN real-only | 34.4 ± 22.3 | 16.7 ± 9.2 | `results/eval/d3qn/20251208-011544.json`, `...011553.json` |
| Rainbow (offline) | 10.9 ± 1.7 | n/a | `results/eval/rainbow/20251208-003300/rainbow.json` |
| D3QN CF (0.25) | 52.4 ± 25.6 | 18.6 ± 10.7 | `results/eval/d3qn/20251208-030337.json`, `...030347.json` |
| D3QN CF (0.10) | 51.3 ± 26.6 | 19.1 ± 12.6 | `results/eval/d3qn/20251208-030839.json`, `...030846.json` |
| D3QN CF (0.05) | 127.9 ± 66.1 | 21.1 ± 12.9 | `results/eval/d3qn/20251208-031715.json`, `...031725.json` |

## Plots
![Final comparison](results/plots_cf_final/comparison_bar.png)
![D3QN losses/evals](results/plots_cf_final/d3qn_overview.png)

## Demo Videos
- CartPole CF (0.05): `results/infer/d3qn/20251208-180811/demo-episode-0.mp4` (returns 94, 135)
- Earlier CartPole demos: `results/infer/d3qn/20251207-223936/demo-episode-0.mp4`, `results/infer/d3qn/20251207-223945/demo-episode-0.mp4`
- Rainbow: `results/infer/rainbow/20251207-224001/demo-episode-0.mp4`
- SAC: `results/infer/sac/20251207-223952/demo-episode-0.mp4`
- Lunar Lander (random): `results/infer/lunar/20251208-182039/demo-episode-0.mp4`, `demo-episode-1.mp4`

## Repo Layout
- `scripts/`: dataset, train, eval, infer, plotting.
- `CTRL/`, `ctrl_algorithms/`: env/dynamics, BiCoGAN, D3QN/Rainbow models and utilities.
- `data/`: SD dataset.
- `results/`: training artifacts, eval JSONs, plots, demos.
- `RUN_SUMMARY.md`, `ANALYSIS_NOTES.md`: detailed experiment log for the report.
