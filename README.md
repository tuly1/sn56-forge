# sn56-forge

Original training pipeline for Gradients (Bittensor SN56) text tournaments.

Built from scratch on `transformers` + `trl` + `peft`. Techniques are informed
by public tournament results; all implementation, structure, and code in this
repository is original work.

## Validator contract (the only parts that must match the spec)

- Dockerfile: `ops/docker/standalone-text-trainer.dockerfile` (built from repo root)
- Entrypoint receives: `--task-id --model --dataset --dataset-type --task-type
  --file-format --expected-repo-name --hours-to-complete`
- Task types: `InstructTextTask`, `ChatTask`, `DpoTask`, `GrpoTask` (`EnvTask`
  falls through to the floor)
- Env: `USE_KL=1` + `KL_COEF` on KL-regularised instruct tasks — the training
  objective includes the matching KL(model ‖ base) term because scoring does
- Data/model are pre-staged read-only: dataset at
  `/cache/datasets/{task_id}_train_data.json`, base model at `/cache/models/...`
- Output: final model written to `/app/checkpoints/{task_id}/{expected_repo_name}`
- Runtime: no internet, `/cache` read-only, wall-clock kill at `hours_to_complete`

## How it works

Every task loads the pre-staged data and base model from `/cache`, attaches a
LoRA adapter, trains paced against the wall clock, and mirrors the adapter into
the output path continuously so a kill always leaves a scoreable model.

- **Instruct / Chat** (`forge/tasks/instruct.py`) — supervised fine-tuning with
  loss on completion tokens only. On KL tasks it swaps in `forge/tuning/kl.py`,
  which adds `kl_coef · KL(model ‖ base)` using the LoRA-disabled model as the
  reference — the exact term the evaluator scores.
- **DPO** (`forge/tasks/dpo.py`) — TRL `DPOTrainer` at β=0.1 (matching the
  evaluator), reference = the LoRA-disabled base.
- **GRPO** (`forge/tasks/grpo.py`) — TRL `GRPOTrainer` with the validator's
  reward-function source compiled into callables (`forge/tasks/rewards.py`).

## Layout

- `forge/cli.py` — argument parsing; funnels every failure to the fallback
- `forge/clock.py` — wall-clock accounting and pacing math
- `forge/data/` — dataset loading, dataset-type parsing, prompt assembly, tokenisation
- `forge/model.py` — cache-only base-model resolution + LoRA
- `forge/tasks/` — one handler per task type, plus the fallback floor
- `forge/tuning/` — hyperparameter plans, the deadline callback, the KL trainer
- `ops/docker/` — the CUDA submission image and a CPU twin for smoke testing

## Design principles

1. **Pace to the clock.** Measure real throughput per step, then stop with
   enough reserve to write the final model before the kill.
2. **Train how you're scored.** Match the eval objective, including the KL term
   and the same DPO/GRPO trainers the evaluator uses.
3. **Never forfeit.** A non-zero exit uploads nothing (scored −1); any handler
   failure degrades to a valid LoRA adapter over the base, so we always submit
   something scoreable.

## Testing

- `pytest` — ML-free unit tests over parsing, prompt assembly, loading, reward
  materialisation, and pacing math.
- `ops/docker/standalone-text-trainer.cpu.dockerfile` — a CPU image that runs the
  exact entrypoint end-to-end with a tiny model, validating the container
  contract (args, cache layout, output path, clean exit) without a GPU.
