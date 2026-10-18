# Base-model fingerprint tables (hashes only)

Everything under `forge/routing/data/`, `forge/routing/base_hashes/` and `forge/routing/gemma_reference_v1/` is
**metadata about public base models**. None of it is a dataset, a checkpoint, an adapter or a tensor.

## What the files contain

| Path | Content |
|------|---------|
| `data/base_fingerprints.json` (5.5 MB, plain JSON) | 102 public models from the tournament pool. Per model: the pinned Hugging Face revision, parameter count, five config keys (architecture, sizes), and per tensor `[dtype, shape, sha256(first 16 KiB of stored bytes)[:32]]`. 1-D tensors also carry `sha256(all bytes)[:32]`. No model-name hashes. |
| `base_hashes/<key>.json` (5 Qwen bases) | Per tensor: full `sha256` of the stored bytes, shape, dtype. |
| `base_hashes/<key>.config.json` | A copy of the public `config.json`, used only to shortlist a table. |
| `gemma_reference_v1/unsloth__gemma-2-2b.json` + `.config.json` | The same per-tensor hash table and config for `unsloth/gemma-2-2b`. |
| `gemma_reference_v1/BUILD-RECEIPT.json`, `ROOT-*.json` | Build and replay receipts (file hashes, counts, verdicts). |

**No tensor values.** `base_fingerprints.json` holds strings, integers and nulls only (93,091 strings, 61,560 integers,
8 nulls, 0 floats). The hash tables hold strings and integers only. The only floats in these directories are scalar
settings inside the copied `config.json` files (for example `rms_norm_eps`) and two numbers in a replay receipt.
A SHA-256 digest cannot be inverted, so no weight can be recovered from these files.

## What they are used for

Routing and damage detection only. At task time the trainer hashes the **validator-delivered** base files on local disk
and compares the digests with these tables:

- `forge/routing/damage.py` (`load_table`, `identify_source`, `fingerprint_assess`): clean / damaged / unknown. The
  source is identified from the delivered norm-tensor hashes alone; the task's model argument is not read.
- `forge/routing/base_identity.py`: is every tensor byte-identical to a known public checkpoint.
- `forge/routing/gemma_identity_gate.py`: the same question for `unsloth/gemma-2-2b`.

The verdict selects **hyperparameters** (learning rate, precision, batch). It never changes which data is trained on,
never supplies weights, and is never a gradient signal. Nothing here is read by the model, the optimizer or the
tokenizer. No network call is made at task time: the tables are read from this directory and the delivered model
from `/cache/models/`. Training starts from the validator-delivered weights in every case.

## How to regenerate

All inputs are public Hugging Face repositories at the pinned revisions recorded in each table.

- `data/base_fingerprints.json`: `research/20260922-prod-patch/fingerprints/build_fingerprints.py` in the research
  repository (run as `PYTHONPATH=<this tree> python build_fingerprints.py`). It reads the safetensors header and the
  first 16 KiB of each tensor by HTTP range requests and calls `forge.routing.damage.fingerprint_files`, the same
  function the trainer runs on the delivered files, so shipped and task-time hashes come from one code path. The
  model list is `pool.json` beside the builder. The 2026-09-27 addendum (`addendum.models`) was built the same way.
  The 2026-10-18 edit (drop every `name_hash` field and the `gradients-io-tournaments/tournament-tourn_*` entries,
  keep everything else byte for byte) is `research/20261018-f2f3-decision/edit_table.py` in the research repository.
- `base_hashes/<key>.json`: `forge.routing.base_identity.build_table(<local public checkpoint dir>)`, keeping the
  `tensors` map; `<key>.config.json` is that checkpoint's `config.json`.
- `gemma_reference_v1/unsloth__gemma-2-2b.json`: the same `build_table` on
  `unsloth/gemma-2-2b@25319945f7fd83b8b903e12081777b7eef2ba993`; `BUILD-RECEIPT.json` records the source and hashes.

A rebuilt table is checked by `tests/test_damage_fingerprint.py`, `tests/test_base_identity_route.py` and
`tests/test_identity_report_e2e.py`.

## Scope of the model list

Public base models seen in the validator's text-task pool, up to 35 B (the validator does not augment larger ones).
Two entries come from the organiser's `gradients-io-tournaments` organisation and are kept because they are
organiser seed bases, not miner submissions: `continuous-sft-seed-quasar-king` (an InstructTextTask base in two
recorded tournaments) and `swe-base-qwen3-8b-continuous` (the validator's `ENV_TARGET_TOURN_MODEL`). Past
tournament-winner repositories (`gradients-io-tournaments/tournament-tourn_*`) were removed on 2026-10-18; a task on
such a base reads "unknown" or a reference-free verdict, and the verdict has no route consumer at their size (8.2-8.6 B).
