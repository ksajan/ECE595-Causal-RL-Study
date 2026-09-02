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

The source hashes were checked on nodes 4 and 5 before launch. Both nodes also
passed a 1,000-step oracle-CF smoke run, including evaluation and JSON artifact
creation, before any study seed was started.

## SAC tail allocation

The original node-1 and node-2 parent queues were stopped without stopping
their active seed-5 children. Handoff watchers launch only seeds 6--7 after the
corresponding child exits. Seeds 8--9 are disjoint and run on the new nodes.

| Node | Environment and variants | Seeds |
|---|---|---:|
| 1 | HalfCheetah-v4: real, duplicate, oracle_cf; Ant-v4: real | 5 active, then 6--7 |
| 2 | Hopper-v4: real, duplicate, oracle_cf; Ant-v4: duplicate | 5 active, then 6--7 |
| 3 | Existing Ant-v4 oracle_cf and Walker2d-v4 queues | through 9 |
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
required task and arm. Partial rows are diagnostic only.
