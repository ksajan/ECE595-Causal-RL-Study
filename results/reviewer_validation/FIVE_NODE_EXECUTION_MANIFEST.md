# Five-Node Execution Manifest

This manifest records the optional continuous-control validation launched on
2026-09-02. It is not part of the frozen `claramas-2026-revision-v2`
submission.

## Runtime

All result-producing SAC processes use:

- Python 3.13.13
- PyTorch 2.13.0+cu130
- NumPy 2.5.2
- Gymnasium 1.3.0
- Stable-Baselines3 2.9.0
- MuJoCo 3.12.0
- NVIDIA GeForce RTX 5090

Each run writes its own `dependencies.json`; those files are authoritative for
the runtime actually observed by the process.

## Source identity

- `scripts/workshop/run_mujoco_oracle_cf_sac.py`:
  `b8c19e6d755ecc879111faaa4f7ec5f7c1ebc81ccc15ba6b86a584eb66501479`
- `scripts/workshop/queue_mujoco_oracle_seeds.sh`:
  `2e20508e1bfefeb41b88d906c36534236998da666d61c550796999b496929939`
- `scripts/cluster/retire_queue_after_child.sh`:
  `ab8bb83dbfb41b10c657acb92af99b01ea9bb8b8c26ac151b8d67e2d23465339`
- `scripts/workshop/finalize_reviewer_validation.sh`:
  `4cb1b30dbd4e9e220d42d6d2f0a40eb1d305569e2f9885839dfd89e3edc7b15d`
- `scripts/workshop/monitor_reviewer_validation_cluster.sh`:
  `060debf2fadaed2b8dd7e07b69b4734085b193f6eec969d02b3ec5ff7e2a1a62`
- `scripts/workshop/summarize_continuous_control_results.py`:
  `f756f6e1cabdf768d4b0f002fd90faf45c3f63513023fbc7fc457f049c03ea64`
- `scripts/workshop/plot_continuous_control_results.py`:
  `3c94e7f71590f24631e338f7720274bd632dcb574c9410117bfb666b719bb2c8`
- `scripts/workshop/render_continuous_control_report.py`:
  `c40bf50eef9518f5c8b31b009cc84ccddf28368019c790344de8522866672612`

The training-source hashes were checked on nodes 3--5 before their assigned
launches. Nodes 4 and 5 also passed a 1,000-step oracle-CF smoke run, including
evaluation and JSON artifact creation, before any study seed was started.

## SAC tail allocation

The original node-1 and node-2 parent queues were stopped without stopping
their active seed-5 children. Handoff watchers launched seeds 6--7 after those
children exited. Once node 3 became idle, the replacement queues were stopped
again while their seed-6 children continued; retirement watchers prevent those
queues from launching seed 7. Node 3 owns seed 7, while seeds 8--9 remain
disjoint on nodes 4 and 5.

| Node | Environment and variants | Seeds |
|---|---|---:|
| 1 | HalfCheetah-v4: real, duplicate, oracle_cf; Ant-v4: real | 5--6 |
| 2 | Hopper-v4: real, duplicate, oracle_cf; Ant-v4: duplicate | 5--6 |
| 3 | Ant-v4 oracle_cf and Walker2d-v4 complete; all remaining listed SAC arms | 7 |
| 4 | HalfCheetah-v4: real, duplicate, oracle_cf; Ant-v4: real | 8--9 |
| 5 | Hopper-v4: real, duplicate, oracle_cf; Ant-v4: duplicate | 8--9 |

The Ant-v4 oracle-CF arm was already complete through seed 9 on node 3, so it
was not duplicated on nodes 4 or 5.

## D4RL allocation

- Node 1 finishes the current HalfCheetah-medium-v2 seed-7 synthetic arms and
  seed-8 real arm.
- Node 3 runs HalfCheetah-medium-v2 synthetic seeds 8--9 and real seed 9.
- Node 2 retains the Hopper-medium-v2 queues through seed 9.

## Collection and publication gate

`scripts/workshop/monitor_reviewer_validation_cluster.sh` polls nodes 1--5,
synchronizes completed JSON artifacts, and recomputes paired inference. The
publication generator remains gated on ten exact paired learner seeds for every
required task and arm. It reports both augmentation-versus-real effects and the
matched operational contrasts (`oracle_cf - duplicate` for SAC and
`factual_residual - fresh_residual` for CQL). Partial rows are diagnostic only.
On the first successful gate, `scripts/workshop/finalize_reviewer_validation.sh`
atomically creates a timestamped `final_10seed_*` snapshot containing the
summaries, reportable figures and tables, execution notes, finalization metadata,
checksums for the snapshot, and a checksum manifest covering every synchronized
raw JSON artifact. An idempotent `FINALIZED_PATH` marker prevents later monitor
cycles from overwriting that first complete result.
