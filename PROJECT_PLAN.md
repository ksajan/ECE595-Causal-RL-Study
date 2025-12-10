# Project Execution Plan

## 1. Context & Objectives
- Validate and extend the Counterfactual RL (CTRL) implementation for noisy CartPole and Lunar Lander environments.
- Deliver reproducible experiments comparing real-only training, CTRL counterfactual augmentation, and additional baselines (Base-S, SAC, Rainbow DQN).
- Clean technical debt (paths, seeding, logging) while aligning with paper expectations and preparing publication-quality artifacts (paper + presentation).

## 2. Existing Assets & Code Structure
- `CTRL/ctrl_data.py`: SD dataset generation, noisy CartPole environment, counterfactual synthesis utilities.
- `CTRL/ctrl_models.py`: BiCoGAN components (Generator, Encoder, Discriminator) and D3QN network definitions.
- `CTRL/ctrl_trainer.py`: BiCoGAN training loop plus offline D3QN+CQL training pipeline.
- `CTRL/ctrl_utilities.py`: Policy evaluation, plotting helpers, diagnostics.
- Notebooks: `CTRL/counter_factual_rl_expts.ipynb` (CartPole experiments) and `CTRL/CTRL_Lunar_Lander.ipynb` (Lunar Lander workflow with random transition collection).
- Data artifacts (e.g., `SD_dataset_clean.pt`) referenced but not versioned; ensure they are regenerated or stored with metadata.

## 3. Environment & Tooling Setup
- Standardize Python 3.10+ environment; document `pip install torch gymnasium numpy matplotlib seaborn tqdm scikit-learn pandas mlflow` plus any additional deps discovered while linting.
- Remove hard-coded absolute paths (e.g., `sys.path.append("/Users/.../CARL")`); rely on repo-relative imports or package installs.
- Configure seeds for `numpy`, `torch`, `random`, and Gym; expose seed as CLI/notebook parameter.
- Ensure GPU/CPU device detection uses `torch.device("cuda" if torch.cuda.is_available() else "cpu")` and encapsulate moves inside helper functions.

## 4. Codebase Cleanup Tasks
- Deduplicate imports and enforce Ruff/PEP 8 order (stdlib → third-party → local).
- Add precise type hints and docstrings for public APIs, stating tensor shapes and units where possible.
- Factor reusable notebook logic into Python modules (data collection, training loops, evaluation) to support testing and scripting.
- Validate Ruff (`ruff check CTRL CausalRL Latest-Research Basics`) and formatting (`ruff format` or `black .`) pass after refactors.

## 5. Baseline CTRL (CartPole) Reproduction
- Regenerate the SD dataset via `make_SD_dataset`, logging seed, number of episodes, horizon, and dataset hash/size.
- Train BiCoGAN: run pre-train and adversarial phases; save checkpoints, loss curves, and gamma schedule plots.
- Evaluate counterfactual quality using `test_counterfactual_quality`; record reconstruction metrics for documentation.
- Train offline D3QN+CQL with and without counterfactual augmentation (`build_training_buffer` toggles); log total losses, TD/CQL components, Q statistics, and evaluation returns via `evaluate_policy`.
- Summarize results in tables (mean ± std over fixed evaluation seeds) and archive plots under `results/cartpole/`.

## 6. Base-S Counterfactual Baseline
- Revisit the paper to define Base-S architecture, training objectives, and data requirements.
- Implement Base-S modules mirroring BiCoGAN interfaces so evaluation hooks stay interchangeable.
- Generate Base-S counterfactual transitions; enforce the same clipping, termination, and reward calculations as CTRL for fair comparison.
- Train offline agents on Base-S data (alone and mixed with real data) and benchmark against CTRL results.
- Document qualitative differences (e.g., transition diversity, reward distributions) using diagnostic plots.

## 7. Alternative RL Algorithms
- **Soft Actor-Critic (SAC)**: Implement continuous-action SAC leveraging the noisy action representation or adapt environment with hybrid action mapping; tune entropy coefficient and learning rates.
- **Rainbow DQN**: Apply to the discrete action abstraction (11 bins); enable prioritized replay, multi-step returns, distributional outputs as applicable.
- Integrate both algorithms with shared replay buffers and normalization pipelines; ensure evaluation uses the same `evaluate_policy` hooks or equivalent metrics.
- Conduct hyperparameter sweeps (documented grids) and compare sample efficiency, stability, and final returns versus D3QN.

## 8. Lunar Lander Experiment Track
- Modularize `CTRL_Lunar_Lander.ipynb`: move `CollectRandomTransitions`, dataset definitions, and BiCoGAN equivalents into Python modules for reuse.
- Replace notebook-specific hacks (absolute paths, inline class definitions) with imports from the new modules.
- Reproduce baseline D3QN results on Lunar Lander (real-only vs CF-augmented) including logging/evaluation parity with CartPole.
- Extend experiments to Base-S and alternative algorithms (SAC/Rainbow); capture environment-specific nuances (wind, turbulence settings) and record configuration metadata.
- Curate notebook markdown cells to narrate experimental setup, hyperparameters, and results; keep notebooks executable end-to-end.

## 9. Experiment Orchestration & Analytics
- Create CLI scripts (e.g., `scripts/run_ctrl.py`, `scripts/run_lander.py`) that accept flags for dataset creation, model training, counterfactual generation, and algorithm selection.
- Adopt a centralized logging strategy (CSV/JSON + optional MLflow runs) capturing hyperparameters, seeds, checkpoints, metrics, and artifacts.
- Store plots and metrics under a versioned `results/` hierarchy with README pointers describing contents.
- Perform ablations: number of counterfactuals per sample (`cf_k`), BiCoGAN pre-training epochs, noise levels, and seed robustness; report findings with statistical significance tests where feasible.
- Benchmark against published SOTA metrics for CartPole/Lunar Lander counterfactual RL or comparable benchmarks; highlight strengths and gaps.

## 10. Documentation & Deliverables
- Update `README.md` and subsystem READMEs with setup instructions, command examples, and experiment matrix summaries.
- Maintain the new `PROJECT_PLAN.md` and keep it synchronized with progress; add checklist items if scope evolves.
- Draft the paper using existing proposal LaTeX: include methodology diagrams, training/evaluation details, quantitative tables, ablation studies, and discussion of limitations.
- Prepare presentation slides summarizing motivations, system architecture, experimental results, and future work; include backups detailing hyperparameters and reproducibility steps.
- Assemble a reproducibility bundle: environment file, command scripts, expected outputs, troubleshooting tips.

## 11. Open Questions & Items Requiring Exploration
- Confirm exact Base-S implementation details from the original paper or supplementary materials.
- Determine best-practice hyperparameters for SAC/Rainbow in noisy counterfactual settings; may require literature survey.
- Decide on long-term logging platform (MLflow vs Weights & Biases vs custom) fitting team workflows.
- Evaluate compute constraints (GPU availability, training duration) and adjust schedules accordingly.
- Investigate additional environments (Pratyush exploring) to generalize findings; keep slot in plan for integrating that work.

## 12. Coordination & Timeline
- Aim to finalize core experiments (CartPole + Base-S + alternative agents) within the current week.
- Dedicate the following week to Lunar Lander replication, ablations, and logging/automation improvements.
- Reserve the subsequent week for paper drafting, figure creation, and slide preparation; iterate with team feedback.
- Schedule regular syncs (daily standups or mid-week calls) for Platipus, has_qad, and Pratyush to monitor progress and unblock issues.
- Track progress against this plan; update statuses (to-do / in-progress / complete) and note blockers promptly.
