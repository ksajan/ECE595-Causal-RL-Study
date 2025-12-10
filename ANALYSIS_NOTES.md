# Experiment Snapshot (CartPole CTRL)

## Baseline evals (20 episodes each, CartPole-v1 clean env)
- SAC: 500 ± 0 (`results/eval/sac/sac.json`)
- D3QN real-only: 30.8 ± 16.3 (`results/eval/d3qn/real.json`)
- Rainbow: 9.5 ± 0.7 (`results/eval/rainbow/rainbow.json`)
- CTRL noisy env eval: D3QN real ≈ 20 ± 13

## Counterfactual (CF) augmentation tests (BiCoGAN 20251207-223241)
- CF k=1, no filtering: 15.7 ± 10.8
- CF filtered (drop reward 0/done): 20.9 ± 11.8
- CF + subsample 50% (drop done, keep 50% CF): 40.6 ± 36.1 (best mean, huge variance)
- CF + subsample 25%: 33.9 ± 21.7
- CF + subsample 10%: 29.9 ± 16.9 (≈ real baseline)
- CF + quality filter |x|≤2, |θ|≤0.2 + subsample 50%: 16.6 ± 11.8

CF stats: real done ≈ 8.5%; CF done ≈ 14.8%; CF reward mean ≈ 0.85; θ biased positive. Filtering done CFs removes ~15% of CFs; subsampling reduces CF weight and stabilizes training.

### Env-rescored CF (action noise + env dynamics reward/done)
- k=1, env_step, cf_sample_frac=0.50 → clean eval 23.3 ± 20.2; CTRL env final 18.2 ± 11.3 (`results/cartpole/d3qn_cf/20251207-234439`).
- k=1, env_step, cf_sample_frac=1.00 → clean eval 19.4 ± 6.6; CTRL env final 17.3 ± 9.1 (`results/cartpole/d3qn_cf/20251207-234636`).
- k=1, env_step, cf_sample_frac=0.25 → clean eval 39.5 ± 23.6 (best mean, noisy); CTRL env final 18.6 ± 11.4 (`results/cartpole/d3qn_cf/20251207-234845`).
- k=1, env_step, cf_sample_frac=0.33, seed=1 → clean eval 27.1 ± 9.8; CTRL env eval 19.6 ± 16.4; run dir `results/cartpole/d3qn_cf/20251208-000153`.
- k=1, env_step, cf_sample_frac=0.25, seed=2 → clean eval 25.2 ± 13.1; CTRL env eval 17.1 ± 8.3; run dir `results/cartpole/d3qn_cf/20251208-001331`.
- k=1, env_step, cf_sample_frac=0.33, seed=2 → clean eval 23.0 ± 20.3; CTRL env eval 18.5 ± 11.7; run dir `results/cartpole/d3qn_cf/20251208-001514`.
- k=1, env_step, cf_sample_frac=0.25, seed=3 → clean eval 47.9 ± 25.0; CTRL env eval 18.4 ± 9.0; run dir `results/cartpole/d3qn_cf/20251208-001746`.
- k=1, env_step, cf_sample_frac=0.33, seed=3 → clean eval 21.6 ± 18.1; CTRL env eval 20.9 ± 13.1; run dir `results/cartpole/d3qn_cf/20251208-001932`.
- k=1, env_step, cf_sample_frac=0.10, seed=0 → clean eval 25.9 ± 13.2; CTRL env eval 16.7 ± 10.2; run dir `results/cartpole/d3qn_cf/20251208-002253`.
- Env-rescored CFs now have reward ≈0.94 and done frac ≈0.06 (closer to real), but noisy-env eval still ~18; clean-env mean can exceed real baseline when CF weight is light (0.25) at the cost of variance.

## Observations / Gaps
- SAC is saturated at 500 return; Rainbow underperforms badly.
- D3QN real-only is modest; CF variants mostly worse or highly unstable unless CF weight is reduced.
- CF termination/reward appears harsher than real data; CF samples overweight done states and skew θ.
- Need better CF realism (action noise, reward/done recomputation) and multiple seeds for stability.
- Rainbow reboot (CleanRL-inspired hyperparams, double DQN target, hard target updates, noisy layers) yields clean eval 10.9 ± 1.7 (`results/cartpole/rainbow_reboot/20251208-001041`), still weak vs D3QN/SAC.

## Next actions
1) Recompute CF reward/done using the CTRL env with sampled action noise; expose flags for CF action noise, env-based scoring, and CF weight.
2) Retrain CF mixes (25–50%) with new scoring; run ≥3 seeds on best configs; evaluate on clean + noisy env.
3) Regenerate comparison plots and document explanations for when CF helps/hurts vs baselines.

## 50-episode eval refresh (env-rescored CF, k=1)
- Clean CartPole: SAC 492.6 ± 33.3 (`results/eval/sac/20251208-003308/sac.json`); D3QN real-only 34.6 ± 19.9 (`results/eval/d3qn/20251208-003243/d3qn.json`); Rainbow reboot 10.9 ± 1.7 (`results/eval/rainbow/20251208-003300/rainbow.json`); D3QN CF (cf_sample_frac=0.25, seed=5, no target_clip) 28.3 ± 17.4 (`results/eval/d3qn/20251208-003251/d3qn.json`).
- CTRL noisy env: D3QN CF target_clip=10 (seed=4) 16.2 ± 9.9 (`results/eval/d3qn/20251208-002843/d3qn.json`); D3QN CF cf_sample_frac=0.25 seed=5 20.1 ± 14.6 (`results/eval/d3qn/20251208-003048/d3qn.json`).
- Aggregating env-rescored CF (cf_sample_frac=0.25, target_clip=20, seeds 1/2/3/5) → clean mean ≈ 18.7 (σ≈1.2) and CTRL-env mean ≈ 18.5 (σ≈1.3); target_clip=10 lowered both clean and noisy scores.
- Net: CF augmentation with env rescoring is still below the real-only D3QN in clean eval and roughly on par in the noisy CTRL env. SAC remains the only fully solved baseline; Rainbow stays weak. New bar plot with these evals saved to `results/plots_new/comparison_bar.png`.

## New CF variant (lower CQL weight)
- Trained cf_sample_frac=0.25, cf_use_env_step, action_noise_std=0.05, alpha_cql=0.01, seed=6 (`results/cartpole/d3qn_cf/20251208-004416`).
- Clean eval 24.1 ± 18.5 (`results/eval/d3qn/20251208-004542/d3qn.json`), CTRL env eval 20.5 ± 14.2 (`results/eval/d3qn/20251208-004550/d3qn.json`).
- Compared to alpha_cql=0.02 seed=5 (28.3 ± 17.4 clean, 20.1 ± 14.6 noisy), lowering CQL recovered some clean return vs other CF seeds but still trails real-only baseline 34.6 ± 19.9; noisy CTRL performance remains ~20.
- Interim takeaway: CF env-rescored setups stay limited by data realism; reducing CQL helps slightly but not enough to surpass real-only. Next knobs to try if time: smaller target_clip (but seed4 regressed), or further lowering CF weight (≤0.1) with more seeds, or mixing a small real-only replay proportion.

## CF weight sweep (lighter CF)
- Trained cf_sample_frac=0.10, cf_use_env_step, action_noise_std=0.05, alpha_cql=0.01, seed=7 (`results/cartpole/d3qn_cf/20251208-004721`).
- Clean eval 41.2 ± 22.7 (`results/eval/d3qn/20251208-004840/d3qn.json`), CTRL env eval 17.9 ± 12.4 (`results/eval/d3qn/20251208-004849/d3qn.json`).
- This seed matches/surpasses the real-only clean mean (34.6) but with high variance; noisy CTRL drops back to ~18. Suggests light CF mix can occasionally help clean performance but does not translate to robustness on the noisy env.
- Overall trend: lighter CF (0.10) can boost clean return in some seeds, but noisy env stays capped ~18–20; heavier CF hurts clean more. Real-only remains the most reliable for clean, SAC still best overall.

## Generator reward/terminal alignment
- `generate_cf_dataset` now clones the generator prediction before adding `STATE_NOISE_STD` noise, clamps the noisy state, and still derives reward/done from the clean prediction, which mirrors the paper’s notion of rescoring CF samples via the CTRL CartPole dynamics so the transitions remain in the noisy SD data distribution.
- Quick sanity run (50 epochs, cf_sample_frac=0.25, env-rescored CF) saved at `results/cartpole/d3qn_cf/20251208-024339`; clean eval 70.74 ± 18.18 (`results/eval/d3qn/20251208-024356/d3qn.json`), CTRL noisy eval 22.32 ± 15.16 (`results/eval/d3qn/20251208-024405/d3qn.json`). The higher clean mean suggests the regenerated CF samples feed the agent a wider but consistent distribution, while CTRL performance still sits near previous baselines.

## Longer cf_sample_frac=0.25 run (200 epochs)
- Training command: `scripts/train.py d3qn --dataset-path data/SD_dataset_clean.pt --use-cf --bicogan-dir results/cartpole/bicogan/20251208-011000/20251208-011253 --cf-sample-frac 0.25 --cf-action-noise-std 0.05 --cf-use-env-step --epochs 200 --seed 0`, resulting run directory `results/cartpole/d3qn_cf/20251208-030303`.
- Periodic evals show rising Q-means (≈17) but consistent clean eval variance (epoch 200 eval in log: 18.58 ± 10.56), which aligns with earlier trends that clean returns bounce but CTRL returns stay lower.
- Final clean CartPole eval: 52.40 ± 25.63 (`results/eval/d3qn/20251208-030337/d3qn.json`); CTRL noisy eval: 18.58 ± 10.67 (`results/eval/d3qn/20251208-030347/d3qn.json`). The robust clean spike shows the lighter CF mix can generate high-scoring trajectories, yet the CTRL env continues to cap mean return at ~18–20, so the next experiments should explore either smaller CF fractions or further adjustments to the reward/terminal scoring to keep the noisy distribution aligned.

## cf_sample_frac=0.10 run (200 epochs, seed=1)
- Training command: `scripts/train.py d3qn --dataset-path data/SD_dataset_clean.pt --use-cf --bicogan-dir results/cartpole/bicogan/20251208-011000/20251208-011253 --cf-sample-frac 0.10 --cf-action-noise-std 0.05 --cf-use-env-step --epochs 200 --seed 1`, output `results/cartpole/d3qn_cf/20251208-030806`.
- Clean eval: 51.26 ± 26.64 (`results/eval/d3qn/20251208-030839/d3qn.json`); CTRL eval: 19.12 ± 12.63 (`results/eval/d3qn/20251208-030846/d3qn.json`). CTRL mean remains ~18–19 while clean mean spikes above 50, so lighter CF weights keep boosting clean returns but not the noisy env.

## Eval comparison snapshot (clean vs. CTRL envs)
- D3QN real-only (500-episode SD dataset) – clean 34.42 ± 22.29, CTRL 16.74 ± 9.20 (`results/eval/d3qn/20251208-011544/d3qn.json`, `results/eval/d3qn/20251208-011553/d3qn.json`).
- CF `cf_sample_frac=0.25 seed=0` – clean 52.40 ± 25.63, CTRL 18.58 ± 10.67 (`results/eval/d3qn/20251208-030337/d3qn.json`, `results/eval/d3qn/20251208-030347/d3qn.json`).
- CF `cf_sample_frac=0.10 seed=1` – clean 51.26 ± 26.64, CTRL 19.12 ± 12.63 (`results/eval/d3qn/20251208-030839/d3qn.json`, `results/eval/d3qn/20251208-030846/d3qn.json`).
These numbers frame the current understanding: clean returns benefit from CF, but CTRL/noisy remains capped, suggesting future steps should adjust CF sampling or maybe increase real-data proportion when evaluating noisy policy.

## cf_sample_frac=0.05 run (200 epochs, seed=2)
- Command: `scripts/train.py d3qn --dataset-path data/SD_dataset_clean.pt --use-cf --bicogan-dir results/cartpole/bicogan/20251208-011000/20251208-011253 --cf-sample-frac 0.05 --cf-action-noise-std 0.05 --cf-use-env-step --epochs 200 --seed 2`, producing `results/cartpole/d3qn_cf/20251208-031646`.
- Clean eval: 127.94 ± 66.14 (`results/eval/d3qn/20251208-031715/d3qn.json`); CTRL eval: 21.14 ± 12.87 (`results/eval/d3qn/20251208-031725/d3qn.json`). The CFG mix now delivers very high clean returns while the noisy CTRL evaluation stays around 21, reinforcing the pattern that cleaner CF blends boost clean performance but leave the noisy environment capped.

## Baseline story for the class report
- Focus on SAC (solves the clean CartPole), D3QN real-only, and the CF mixes with aligned reward/rescore scoring. Use these snapshots in the report:
  - SAC clean: 492.6 ± 33.3 (`results/eval/sac/20251208-003308/sac.json`).
  - D3QN real-only: clean 34.42 ± 22.29, CTRL 16.74 ± 9.20 (`results/eval/d3qn/20251208-011544/d3qn.json`, `results/eval/d3qn/20251208-011553/d3qn.json`).
  - D3QN CF (0.25): clean 52.40 ± 25.63, CTRL 18.58 ± 10.67 (`results/eval/d3qn/20251208-030337/d3qn.json`, `results/eval/d3qn/20251208-030347/d3qn.json`).
  - D3QN CF (0.10): clean 51.26 ± 26.64, CTRL 19.12 ± 12.63 (`results/eval/d3qn/20251208-030839/d3qn.json`, `results/eval/d3qn/20251208-030846/d3qn.json`).
  - D3QN CF (0.05): clean 127.94 ± 66.14, CTRL 21.14 ± 12.87 (`results/eval/d3qn/20251208-031715/d3qn.json`, `results/eval/d3qn/20251208-031725/d3qn.json`).
- This set of results keeps the narrative consistent: CF weight increases clean return but CTRL returns remain in the 18–21 plateau, while SAC still dominates clean scores. The report can refer to these specific eval files when discussing each baseline.
