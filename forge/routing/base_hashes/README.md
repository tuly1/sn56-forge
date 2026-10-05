# Public-base hash tables (hashes only)

Per-tensor SHA-256 digests, shapes and dtypes of five public Qwen checkpoints, plus a copy of each public
`config.json`. No tensor values. Used only by `forge/routing/base_identity.py` to decide whether the
validator-delivered base is byte-identical to a known public checkpoint; the answer selects hyperparameters.

Contents, use and regeneration are documented in [`../data/README.md`](../data/README.md).
