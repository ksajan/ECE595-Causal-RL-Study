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
- Node 2 retains Hopper-medium-v2 seed 8 for all synthetic arms, seed 9 for the
  real arm, and seed 9 for the simulator-mean arm.
- After their SAC seed-9 processes exit, node 4 runs Hopper-medium-v2
  fresh-residual seed 9 and node 5 runs factual-residual seed 9. The original
  node-2 queue parents for these two arms were retired while their seed-8 child
  processes remained live, preventing duplicate seed-9 launches.

Before the handoff, the Hopper dataset and augmentation cache were copied to
nodes 4 and 5 and verified against the node-2 source:

- `data/d4rl_hdf5/hopper-medium-v2.hdf5`:
  `5bdf1bc4a713c82941de44633df669b36c89850b652a25985166796d25cf71a0`
- `data/d4rl_simulator_cf_v2/hopper-medium-v2/augmentation_seed_0_scale_0p1_n_all.hdf5`:
  `96077e9670e8a410ff87137e5ee25d9f223e3426b9b7a0e8f83953a549bd8cf5`

All three participating nodes reported identical hashes for the D4RL runner
(`9482f5ae3599307a6f46e1bd7054ab599a71954543ce8d33920df335b7a59a10`)
and queue wrapper
(`5cce506cecaefc3a343773cd29325340e475bcb9f8b6be9ef9866dda2113dc0b`).

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
