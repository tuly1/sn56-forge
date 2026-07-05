# sn56-forge

Original training pipeline for Gradients (Bittensor SN56) text tournaments.

Built from scratch. Techniques are informed by public tournament results; all
implementation, structure, and code in this repository is original work.

## Validator contract (the only parts that must match the spec)

- Dockerfile: `ops/docker/standalone-text-trainer.dockerfile`
- Entrypoint receives: `--task-id --model --dataset --dataset-type --task-type
  --file-format --expected-repo-name --hours-to-complete`
- Task types: `InstructTextTask`, `DpoTask`, `GrpoTask`, `ChatTask`, `EnvTask`
- Env: `USE_KL=1` + `KL_COEF` on KL-regularised instruct tasks — the training
  objective must include the matching KL(model ‖ base) term because scoring does
- Output: final model written to `/app/checkpoints/{task_id}/{expected_repo_name}`
- Runtime: no internet, `/cache` read-only, wall-clock kill at `hours_to_complete`

## Layout

- `forge/cli.py` — argument parsing and task dispatch
- `forge/data/` — dataset download, schema parsing, prompt assembly
- `forge/tasks/` — one module per task type
- `forge/tuning/` — budget pacing, LR probing, adapter configuration
- `ops/docker/` — Dockerfiles and entrypoint

## Design principles

1. **Pace to the clock.** Measure real throughput early, then size training to
   finish just under the wall-clock kill with a saved, exportable model.
2. **Train how you're scored.** Match the eval objective (incl. KL terms).
3. **Never crash on cleverness.** Every adaptive decision degrades to a safe
   default.
