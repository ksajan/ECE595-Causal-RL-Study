# Continuous-Control Validation Results

All rows use ten paired learner seeds and fixed evaluation seed banks. Intervals are paired 95% bootstrap intervals. Holm values adjust each test across the planned task-by-variant family within a domain.

## Online MuJoCo SAC

| Task | Arm | n | Absolute return (mean +/- SD) | Paired delta vs real [95% CI] | +/-/ties | Holm p (t/W/R) |
|---|---|---:|---:|---:|---:|---:|
| HalfCheetah | Real | 10 | 10904.34 +/- 346.87 | -- | -- | -- |
| HalfCheetah | Duplicate control | 10 | 11176.52 +/- 331.35 | 272.18 [39.75, 501.18] | 8/2/0 | 0.46/0.52/0.36 |
| HalfCheetah | Oracle counterfactual | 10 | 10653.30 +/- 983.45 | -251.04 [-770.21, 218.64] | 3/7/0 | 1.00/1.00/1.00 |
| Hopper | Real | 10 | 2969.76 +/- 651.33 | -- | -- | -- |
| Hopper | Duplicate control | 10 | 3171.01 +/- 638.97 | 201.25 [-178.98, 625.40] | 6/4/0 | 1.00/1.00/1.00 |
| Hopper | Oracle counterfactual | 10 | 2910.04 +/- 812.90 | -59.71 [-644.16, 536.84] | 5/5/0 | 1.00/1.00/1.00 |
| Walker2d | Real | 10 | 4620.99 +/- 459.21 | -- | -- | -- |
| Walker2d | Duplicate control | 10 | 4222.67 +/- 479.05 | -398.32 [-741.68, -49.80] | 3/7/0 | 0.46/0.59/0.48 |
| Walker2d | Oracle counterfactual | 10 | 4597.47 +/- 400.59 | -23.52 [-416.41, 371.25] | 5/5/0 | 1.00/1.00/1.00 |
| Ant | Real | 10 | 3842.57 +/- 922.17 | -- | -- | -- |
| Ant | Duplicate control | 10 | 3777.98 +/- 793.14 | -64.59 [-692.12, 683.19] | 3/7/0 | 1.00/1.00/1.00 |
| Ant | Oracle counterfactual | 10 | 4280.06 +/- 894.50 | 437.49 [-267.37, 1113.85] | 7/3/0 | 1.00/1.00/1.00 |

Interval diagnostics: HalfCheetah Duplicate control: positive interval; HalfCheetah Oracle counterfactual: interval includes zero; Hopper Duplicate control: interval includes zero; Hopper Oracle counterfactual: interval includes zero; Walker2d Duplicate control: negative interval; Walker2d Oracle counterfactual: interval includes zero; Ant Duplicate control: interval includes zero; Ant Oracle counterfactual: interval includes zero.

## Offline D4RL CQL

| Task | Arm | n | Absolute normalized score (mean +/- SD) | Paired delta vs real [95% CI] | +/-/ties | Holm p (t/W/R) |
|---|---|---:|---:|---:|---:|---:|
| HalfCheetah-medium | Real | 10 | 47.01 +/- 0.18 | -- | -- | -- |
| HalfCheetah-medium | Simulator mean | 10 | 47.43 +/- 0.28 | 0.42 [0.22, 0.61] | 9/1/0 | 0.01/0.02/0.02 |
| HalfCheetah-medium | Fresh residual | 10 | 47.35 +/- 0.17 | 0.35 [0.25, 0.43] | 10/0/0 | 0.00/0.01/0.01 |
| HalfCheetah-medium | Factual residual | 10 | 47.40 +/- 0.33 | 0.39 [0.17, 0.59] | 8/2/0 | 0.02/0.06/0.04 |
| Hopper-medium | Real | 10 | 54.26 +/- 1.77 | -- | -- | -- |
| Hopper-medium | Simulator mean | 10 | 54.23 +/- 2.31 | -0.03 [-1.62, 1.98] | 5/5/0 | 0.98/0.64/0.98 |
| Hopper-medium | Fresh residual | 10 | 50.36 +/- 2.61 | -3.89 [-5.76, -2.28] | 0/10/0 | 0.01/0.01/0.01 |
| Hopper-medium | Factual residual | 10 | 55.34 +/- 3.25 | 1.09 [-0.51, 2.72] | 6/4/0 | 0.49/0.64/0.48 |

Interval diagnostics: HalfCheetah-medium Simulator mean: positive interval; HalfCheetah-medium Fresh residual: positive interval; HalfCheetah-medium Factual residual: positive interval; Hopper-medium Simulator mean: interval includes zero; Hopper-medium Fresh residual: negative interval; Hopper-medium Factual residual: interval includes zero.

## Matched augmentation controls

These direct contrasts isolate each intervention-based augmentation arm from its matched data-volume or noise-resampling control. They are operational contrasts and do not by themselves prove causal identification.

| Domain | Task | Contrast | n | Paired delta [95% CI] | +/-/ties | Holm p (t/W/R) |
|---|---|---|---:|---:|---:|---:|
| Online SAC | HalfCheetah | Oracle CF minus duplicate replay | 10 | -523.22 [-1079.61, -67.54] | 2/8/0 | 0.29/0.20/0.20 |
| Online SAC | Hopper | Oracle CF minus duplicate replay | 10 | -260.96 [-888.28, 436.24] | 3/7/0 | 0.52/0.64/0.52 |
| Online SAC | Walker2d | Oracle CF minus duplicate replay | 10 | 374.80 [37.24, 719.80] | 7/3/0 | 0.29/0.39/0.23 |
| Online SAC | Ant | Oracle CF minus duplicate replay | 10 | 502.07 [-287.10, 1246.84] | 6/4/0 | 0.52/0.64/0.52 |
| Offline CQL | HalfCheetah-medium | Factual-noise minus fresh-noise residual | 10 | 0.04 [-0.10, 0.19] | 6/4/0 | 0.60/0.62/0.60 |
| Offline CQL | Hopper-medium | Factual-noise minus fresh-noise residual | 10 | 4.98 [3.46, 6.66] | 10/0/0 | 0.00/0.00/0.00 |

These experiments isolate simulator-based one-step augmentation. They are not a learned-BiCoGAN CTRL reproduction, and a confidence interval that excludes zero is not by itself evidence of practical importance or generality.
