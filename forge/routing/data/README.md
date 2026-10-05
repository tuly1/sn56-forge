# Base-model fingerprint tables (hashes only)

Everything under `forge/routing/data/`, `forge/routing/base_hashes/` and `forge/routing/gemma_reference_v1/` is
**metadata about public base models**. None of it is a dataset, a checkpoint, an adapter or a tensor.

## What the files contain

| Path | Content |
|------|---------|
| `data/base_fingerprints.json` (5.9 MB, plain JSON) | 109 public models from the tournament pool. Per model: the pinned Hugging Face revision, `name_hash = sha256(model_id)[:16]`, parameter count, five config keys (architecture, sizes), and per tensor `[dtype, shape, sha256(first 16 KiB of stored bytes)[:32]]`. 1-D tensors also carry `sha256(all bytes)[:32]`. |
| `base_hashes/<key>.json` (5 Qwen bases) | Per tensor: full `sha256` of the stored bytes, shape, dtype. |
| `base_hashes/<key>.config.json` | A copy of the public `config.json`, used only to shortlist a table. |
| `gemma_reference_v1/unsloth__gemma-2-2b.json` + `.config.json` | The same per-tensor hash table and config for `unsloth/gemma-2-2b`. |
| `gemma_reference_v1/BUILD-RECEIPT.json`, `ROOT-*.json` | Build and replay receipts (file hashes, counts, verdicts). |

**No tensor values.** `base_fingerprints.json` holds strings, integers and nulls only (100,093 strings, 66,400 integers,
8 nulls, 0 floats). The hash tables hold strings and integers only. The only floats in these directories are scalar
settings inside the copied `config.json` files (for example `rms_norm_eps`) and two numbers in a replay receipt.
A SHA-256 digest cannot be inverted, so no weight can be recovered from these files.

## What they are used for

Routing and damage detection only. At task time the trainer hashes the **validator-delivered** base files on local disk
and compares the digests with these tables:

- `forge/routing/damage.py` (`load_table`, `identify_source`, `classify_fingerprint`): clean / damaged / unknown.
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
- `base_hashes/<key>.json`: `forge.routing.base_identity.build_table(<local public checkpoint dir>)`, keeping the
  `tensors` map; `<key>.config.json` is that checkpoint's `config.json`.
- `gemma_reference_v1/unsloth__gemma-2-2b.json`: the same `build_table` on
  `unsloth/gemma-2-2b@25319945f7fd83b8b903e12081777b7eef2ba993`; `BUILD-RECEIPT.json` records the source and hashes.

A rebuilt table is checked by `tests/test_damage_fingerprint.py`, `tests/test_base_identity_route.py` and
`tests/test_identity_report_e2e.py`.

## Entries from `gradients-io-tournaments/*` (owner decision pending)

9 of the 109 fingerprinted models are public repositories of the tournament organiser's
`gradients-io-tournaments` organisation: two environment-tournament base models and seven past tournament winner
repositories. They entered the table because they were in the public model pool. All are 8.2 B to 8.6 B parameters,
above the 4.25 B scope of the single-GPU text routes this table was built for. Removing them would make the detector
answer "unknown" for those bases (the caller then keeps its default route); the effect on larger-model routes has not
been measured. They are kept in this release; whether to remove them is an owner decision.

| Model id | Parameters | Pool role | Revision | name_hash |
|----------|------------|-----------|----------|-----------|
| `gradients-io-tournaments/continuous-sft-seed-quasar-king` | 8.60 B |  | `9eeefb5fd55b` | `4c655ac30fb7db82` |
| `gradients-io-tournaments/swe-base-qwen3-8b-continuous` | 8.19 B | env_target | `43eb28257388` | `033e2fe81f69afd1` |
| `gradients-io-tournaments/tournament-tourn_03a3ba3f5bb25c4a_20260817-4eb03520-0a95-40b4-be2b-84b04d4fc052-5HWPK9f6` | 8.19 B |  | `f2e38aea8c29` | `2e546eb964704ee0` |
| `gradients-io-tournaments/tournament-tourn_20f928c9169811d7_20260810-91d2a1a9-ccf6-49ce-a95b-c1c16c7b6dd3-5GU4Xkd3` | 8.19 B |  | `d0872c3f8c8b` | `746f99fd1145644e` |
| `gradients-io-tournaments/tournament-tourn_79c20b9b66ecda90_20260803-96ac3651-628a-4765-8e70-320118e725f6-5GU4Xkd3` | 8.19 B |  | `1198a09b9b81` | `2520ff1aa7afd166` |
| `gradients-io-tournaments/tournament-tourn_a75995710d264c0f_20260720-b3a70172-bc4d-4084-836c-ad455e2393e1-5Ca32LwM` | 8.60 B |  | `2804b2b67ef2` | `614575286137c09b` |
| `gradients-io-tournaments/tournament-tourn_c03a612f287687e0_20260713-372ad338-c737-4bea-a34f-e87b7ff2b5ac-5GU4Xkd3` | 8.19 B |  | `804012cc3308` | `8a909ceec27fba91` |
| `gradients-io-tournaments/tournament-tourn_cae8bd77d98adb77_20260727-13177b90-0065-4e79-9ace-29116f569f2f-5GU4Xkd3` | 8.19 B |  | `f63b62e24538` | `e51d7e0bccb22c68` |
| `gradients-io-tournaments/tournament-tourn_cae8bd77d98adb77_20260727-aaed45a4-f044-4017-a3cc-f8f1f706bc6a-5Ca32LwM` | 8.60 B |  | `a295ec3c7eb3` | `0d07e71f3deca398` |

2 further `gradients-io-tournaments/*` repositories are listed under `not_fingerprinted` (name hash, revision
and the reason only, no tensor hashes): `tournament-tourn_2ba0f79237bffcf7_20260907-3d…`, `tournament-tourn_add1dc83b8fd58b0_20260831-2a…`.
