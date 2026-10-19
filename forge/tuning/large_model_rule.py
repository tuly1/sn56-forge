"""Large-model batch rule (release candidate v9, 2026-09-30; default ON; ported from c1f7501 + c919e4c).

For in-scope full-FT default tasks with params_b >= LARGE_MIN_PARAMS_B (3.0 B), after the timing probe, grow the effective
batch (more gradient accumulation, same micro-batch, so the memory fit holds) toward the public champion's tokens per step,
and scale the LR with the batch by (B_new / B_release) ** LR_ALPHA (0.5), bounded at LR_FACTOR_MAX (x2). No LR probe.
Below 3.0 B nothing here runs (sft_v2 checks the size before calling), so behaviour is 854bf7a's by construction.

(alpha, cap) = (0.5, 2.0) is the phase-ba frozen-rule choice (research/20260923-replay-bench/phase-ba,
reduce_phase_ba.py; results/DEV-VERDICT.json sha af0337fcdfe4: CHOSEN_ON_DEV chosen=C, eligible C and E), in offline
replays of past public tasks. Arm C on 7024722 (one seed = DISCOVERY): 522e0294 Qwen2.5-3B DEV +0.873 % vs 7024722
(inside the +1 % E1 bound; c1f7501's alpha 0.75 was DEV +2.32 %, a blow-up); 880b38e9 NuExtract-v1.5 DEV -4.323 %.
phase-aw (c1f7501 on 7024722): bc347f11 Qwen2.5-3B-Instruct ruqa logged sft_v2_large_model_batch_kept (no growth),
no measurable change (CI spans 0) = the release. Full replay results: research/20260923-replay-bench/phase-ba, phase-aw.
  Below 3 B growth hurt (LFM2.5-2.6B 90d361cb +0.0067 DEV; LFM2.5-1.2B cc4550ab +0.028..+0.056 DEV), hence the size gate.
  The 3.0 B threshold is fitted to those cells (anywhere in (2.70, 3.09] separates them), not derived.
"""

from __future__ import annotations

import math
import os
from typing import Any

SWITCH_ENV = "FORGE_V2_LARGE_MODEL_RULE"
DEFAULT = "1"
LARGE_MIN_PARAMS_B = 3.0
CHAMPION_BLOCKS = 48  # instruct_config 2_4_b bucket (every model >= 2 B)
BLOCK_TOKENS = 2048
REF_GPU_GB = 80.0
DEFAULT_FILL = 0.85  # unpacked rows: estimated packed fill of the champion's next-fit packer
S_GROW = 200  # steps per epoch that must remain (winners' step counts on the trio's lost R1 tasks: 200-268)
MIN_UPDATES = 250  # planned optimizer updates that must remain
MIN_EPOCHS_FIT = 1.0  # growth only when at least one epoch fits the window at the measured step time
LR_ALPHA = 0.5  # phase-ba arm C (CHOSEN_ON_DEV); c1f7501 used 0.75, which failed the blow-up guard on 522e0294 (+2.3 %)
LR_FACTOR_MAX = 2.0  # phase-ba arm C cap
# RESEARCH knobs (c919e4c, phase-ba). Unset (production: the validator sets no FORGE_ env) = LR_ALPHA 0.5, LR_FACTOR_MAX 2.0.
# An out-of-range or unparsable value falls back to these defaults.
ALPHA_ENV = "FORGE_V2_LARGE_MODEL_LR_ALPHA"  # exponent in [0, 1]; 0 = the release LR held (batch still grows)
CAP_ENV = "FORGE_V2_LARGE_MODEL_LR_CAP"  # LR factor bound in [1, 4]


def enabled() -> bool:
    return os.environ.get(SWITCH_ENV, DEFAULT).strip() == "1"


def applies(params_b: float) -> bool:
    try:
        return enabled() and float(params_b) >= LARGE_MIN_PARAMS_B
    except (TypeError, ValueError):
        return False


def champion_len(p50: float, p99: float) -> int:
    """The champion planner's block length: 1.1 x max(p99, 2 x p50), rounded up to 64."""
    return int(math.ceil(1.1 * max(float(p99), 2.0 * float(p50)) / 64.0) * 64)


def token_cap(champ_len: int, fill: float, gpu_gb: float | None) -> float:
    g = REF_GPU_GB if not gpu_gb or gpu_gb <= 0 else float(gpu_gb)
    return (CHAMPION_BLOCKS * min(max(64, int(champ_len)), BLOCK_TOKENS) * max(0.05, min(1.0, float(fill)))
            * min(1.0, g / REF_GPU_GB))


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    return v if lo <= v <= hi else default


def lr_alpha() -> float:
    return _env_float(ALPHA_ENV, LR_ALPHA, 0.0, 1.0)


def lr_cap() -> float:
    return _env_float(CAP_ENV, LR_FACTOR_MAX, 1.0, 4.0)


def lr_factor(new_tokens: float, old_tokens: float) -> float:
    return min(lr_cap(), max(1.0, (float(new_tokens) / max(1e-9, float(old_tokens))) ** lr_alpha()))


def plan_growth(*, train_tokens: float, tokens_per_unit: float, eff_units: int, micro: int, t_step_s: float,
                window_s: float, max_epochs: float, champ_len: int, fill: float,
                gpu_gb: float | None) -> tuple[int, dict[str, Any]] | None:
    """(new effective units, info) when the batch should grow, else None. Units are rows (unpacked) or packed blocks;
    the batch only grows, in whole micro-batches."""
    tpu = max(1.0, float(tokens_per_unit))
    micro = max(1, int(micro))
    eff_units = max(1, int(eff_units))
    n_units = max(1, int(round(float(train_tokens) / tpu)))
    spe = max(1, n_units // eff_units)
    if not t_step_s or t_step_s <= 0:
        return None
    e_fit = 0.85 * max(0.0, float(window_s)) / (float(t_step_s) * spe)
    info: dict[str, Any] = {"epochs_fit": round(e_fit, 3), "old_units": eff_units}
    if e_fit < MIN_EPOCHS_FIT:
        return None
    e_plan = min(float(max_epochs), 1.25 * e_fit)  # the release planner: one cosine 25% past the window, capped
    old_tokens = eff_units * tpu
    cap = token_cap(champ_len, fill, gpu_gb)
    target = min(cap, float(train_tokens) / S_GROW, e_plan * float(train_tokens) / MIN_UPDATES)
    new_units = (int(target // tpu) // micro) * micro
    if new_units <= eff_units:
        return None
    new_tokens = new_units * tpu
    info.update(new_units=new_units, epochs_planned=round(e_plan, 3), old_tokens=round(old_tokens, 1),
                new_tokens=round(new_tokens, 1), cap_tokens=round(cap, 1), lr_factor=round(lr_factor(new_tokens, old_tokens), 4),
                lr_alpha=lr_alpha(), lr_cap=lr_cap(),
                binding=min((("token_cap", cap), ("steps_per_epoch", float(train_tokens) / S_GROW),
                             ("updates", e_plan * float(train_tokens) / MIN_UPDATES)), key=lambda kv: kv[1])[0])
    return new_units, info
