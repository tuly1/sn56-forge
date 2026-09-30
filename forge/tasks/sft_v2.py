"""Full-weight supervised fine-tuning for small instruct tasks (v2 recipe).

Why a second SFT path: half of all tournament base models are deliberately
perturbed by the validator before miners see them (noise, pruning, scaling,
re-initialised layers), and a low-rank adapter cannot undo that. Full-weight
training can. This handler is used for non-KL InstructTextTask on one GPU when
the model is small enough; everything else stays on the validated LoRA path.

Pipeline: floor artifact -> tokenize at the evaluator cap -> exact dedup ->
adaptive max length + prompt-side truncation -> stratified dev split ->
learning-rate prior + short sweep -> time-aware epochs -> train with dev
evaluation every 1/8 epoch (validator statistic), best-checkpoint persistence,
early stop -> greedy soup of the best snapshots -> low-LR dev pass -> save.
"""

from __future__ import annotations

import gc
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from forge import telemetry
from forge.clock import Deadline
from forge.data import hygiene, packing, prompts, tokenize
from forge.data.schema import TaskSpec
from forge.data.split import (
    adaptive_max_len,
    dev_size_for,
    exact_dedup,
    grouped_stratified_split,
    random_subsample,
    smart_truncate,
    stratified_split,
)
from forge.tasks.common import (
    ARTIFACT_COMPLETE_BEST,
    ARTIFACT_FLOOR,
    ARTIFACT_PARTIAL_TRAINED_BEST,
    _free_cuda,
    _is_oom,
    compatible_dataclass_kwargs,
    save_adapter,
    workdir,
    write_artifact_truth,
)
from forge.tuning import large_model_rule, lr_probe, selection

EVAL_CAP = 4096
EFF_BATCH_TARGET = 64
# Field route (2026-09-15, after the Sept 14 Round-1 loss): every Round-1 winner on
# every task uploaded full weights trained at the public champion's geometry —
# unpacked rows at effective batch 140 (<1B) / 100 (1–2B) / 48 (2–4B), the analytic
# LR law without a sweep, one cosine sized 25% past the window (min-LR 0.25) so the
# wall-clock stop lands near the LR floor, early stop, greedy soup, dev pass. Our
# adapters (r32) finished 7th/15th/7th behind every competent full-weight entry.
FIELD_FULL_MAX_PARAMS_B = 3.5


def _field_full() -> bool:
    """Field route on by default since 2026-09-15 (GossipCop replay: full weights 0.9868 vs the
    adapter 1.1983 on identical rows, −17.7%, the Sept 14 gap to the winners). FORGE_V2_FIELD=0
    restores the adapter-first routing."""
    return os.environ.get("FORGE_V2_FIELD", "1") != "0"


def field_eff_batch(params_b: float) -> int:
    """Champion size buckets (rows per optimizer step, one GPU)."""
    try:
        forced = int(os.environ.get("FORGE_V2_FIELD_EFF_BATCH", "0") or 0)
    except ValueError:
        forced = 0
    if forced > 0:
        return forced
    if params_b < 1.0:
        return 140
    if params_b < 2.0:
        return 100
    return 48


FIELD_FULL_MODEL_TYPES = frozenset({"llama", "lfm2"})  # families where the field route beat the adapter on a replayed task
# lfm2 (LFM2.5-2.6B) joined 2026-09-16: MathFusionQA replay 58.9k rows / 37.6M tokens, full weights at the raw
# analytic LR (2.8e-5) with bf16 + paged 8-bit AdamW 0.33848 vs the pinned one-epoch adapter 0.34438 (paired win 0.693,
# lb99 0.633) and vs the v2 adapter 0.34255 (win 0.715). The earlier +22% loss was the floored LR (7.5e-5) in fp32.


def field_full_types() -> frozenset[str]:
    """Model families allowed on the field route (FORGE_V2_FIELD_TYPES overrides, comma-separated).
    Validated 2026-09-15: llama (GossipCop replay −17.7% vs the adapter). Excluded after a
    measured loss: gemma2 (gemma2-cas: full weights 0.0760 vs adapter 0.0333). Unknown families
    keep the adapter route — the downside of full weights on a sensitive family is far larger
    than the upside."""
    forced = os.environ.get("FORGE_V2_FIELD_TYPES", "").strip()
    if forced:
        return frozenset(x.strip().lower() for x in forced.split(",") if x.strip())
    return FIELD_FULL_MODEL_TYPES


FIELD_FULL_MIN_ROWS = 15000  # below this the adapter won (CodeFeedback 8k: adapter 0.822 vs full 0.843–0.893)


# family-specific lower gates (2026-09-17): Llama-3.2-1B on GossipCop subsampled to 10k rows / 6.0M tokens — the exact
# Sept 14 dataset below the generic gates — is validated in research/20260916-package (field vs the production v2 adapter);
# CodeFeedback 8k rows / 3.9M tokens still says adapter, so the llama gates sit between the two.
FIELD_FULL_MIN_ROWS_TYPES: dict[str, int] = {"llama": 10000}
FIELD_FULL_MIN_TOKENS_TYPES: dict[str, int] = {"llama": 5_000_000}


def field_full_min_rows(model_type: str | None = None) -> int:
    default = FIELD_FULL_MIN_ROWS_TYPES.get(str(model_type or "").lower(), FIELD_FULL_MIN_ROWS)
    try:
        return int(os.environ.get("FORGE_V2_FIELD_MIN_ROWS", str(default)))
    except ValueError:
        return default


FIELD_FULL_MIN_TOKENS = 8_000_000  # GossipCop 11.8M tokens won; Alpaca-20k 1.2M and CodeFeedback 3.9M lost (2026-09-16)


def field_full_min_tokens(model_type: str | None = None) -> int:
    default = FIELD_FULL_MIN_TOKENS_TYPES.get(str(model_type or "").lower(), FIELD_FULL_MIN_TOKENS)
    try:
        return int(float(os.environ.get("FORGE_V2_FIELD_MIN_TOKENS", str(default))))
    except ValueError:
        return default


# validated vocabulary per family: Llama-3.x 128256 (TinyLlama 32000 / SmolLM2 49152 share model_type "llama" but are
# unvalidated); LFM2.5 128000. Family-specific on purpose: a llama with 128000 or an lfm2 with 128256 is not the validated model.
FIELD_FULL_VOCABS: dict[str, frozenset[int]] = {"llama": frozenset({128256}), "lfm2": frozenset({128000}),
                                                 "qwen2": frozenset({151936})}
# Families admitted to the field route ONLY when the delivered base is byte-identical to a known public checkpoint
# (forge.routing.base_identity). Qwen2.5 (2026-09-16/17, Qwen2.5-0.5B): on clean bases full weights scored −0.33%
# (Alpaca-20k), −1.65% (ru-AAQG-29k) and +0.16% (long-tail-29k) against the production adapter, but on every
# validator-damaged base (scaling / noise / re-init) the adapter won by 1.3–2.9%. Off unless FORGE_V2_FIELD_IDENTICAL_BASE=1.
FIELD_IDENTICAL_BASE_TYPES = frozenset({"qwen2"})
# Identical-base admission, per exact public checkpoint (an "identical" verdict against any other shipped table is
# telemetry only and keeps the current route - Pro review 2026-09-17 12:46). Each entry carries its own size cap and
# data-volume floor, set by where the field route was measured to win:
#  * Qwen2.5-0.5B-Instruct: family gates (>=15k rows, >=8M tokens). Exact Sept 14 task, matched seeds.
#  * Qwen2-1.5B-Instruct (2026-09-19): field wins on grammar 21M tokens (-1.3%, two seeds) and clinical notes 58M tokens
#    (-2.5% at full speed; -1.4% with the GPU clocked to validator speed, per-row ci99 excludes 0, rule fixed beforehand).
#    Below that the adapter wins (Qwen2.5-1.5B/3B-Instruct on ru-AAQG 9.7M tokens: +0.8...+1.5%) -> floor 20M tokens.
#    3B has no positive evidence and stays on the adapter. Damaged bases keep the adapter: the 1.5B damaged-base pairs
#    ran fp32 x0.5, not the bf16 x1.0 that ships, and at 0.5B bf16 full weights lost 3% on a scaled base. (2026-09-23:
#    superseded for detected validator-damaged bases by DAMAGED_BASE_FAMILIES below; this identical-base rule is unchanged.)
#  * Qwen2.5-Math-1.5B-Instruct (2026-09-19): WritingPrompts (Aug 17 pool task) 65k rows / 59M tokens, 2.75 h: field
#    bf16 x1.0 packed 3.19983 vs the live v2 adapter 3.56264 (-10.2%, 99.9% of rows; one seed, full speed). The field
#    arm lands inside the live board (5th 3.157 / 6th 3.204) while the adapter sits above the whole visible top 7.
#    Same rule as Qwen2-1.5B. Qwen2.5-1.5B-Instruct (non-Math) has no measured win at any volume and stays out.
FIELD_IDENTICAL_BASE_MAX_PARAMS_B = 1.0  # default cap for an admitted candidate without its own rule
FIELD_IDENTICAL_BASE_RULES: dict[str, dict[str, float | int | None]] = {
    "Qwen__Qwen2.5-0.5B-Instruct": {"max_params_b": 1.0, "min_tokens": None},
    "Qwen__Qwen2-1.5B-Instruct": {"max_params_b": 2.0, "min_tokens": 20_000_000},
    "Qwen__Qwen2.5-Math-1.5B-Instruct": {"max_params_b": 2.0, "min_tokens": 20_000_000},
}
FIELD_IDENTICAL_BASE_CANDIDATES = frozenset(FIELD_IDENTICAL_BASE_RULES)


# Release default ON (2026-09-17): the validator runs this repo's image with its own environment, so the switch must
# live in code. FORGE_V2_FIELD_IDENTICAL_BASE=0 disables it (research replays of the pre-D behaviour).
FIELD_IDENTICAL_BASE_DEFAULT = "1"


def _field_identical_base() -> bool:
    return os.environ.get("FORGE_V2_FIELD_IDENTICAL_BASE", FIELD_IDENTICAL_BASE_DEFAULT).strip() == "1"


# Row-length scope of the identical-base admission. "long" (default): only tasks whose median row is above the v2 row gate
# (>= 256 tokens), where production otherwise runs the throughput-starved v2 full strategy (UltraChat-20k: field −5.2%,
# 98% of rows). Short-row tasks keep the legacy adapter (Sept-14-shaped fixture: field +1.3% worse). "all" admits both.
# 2026-09-17 owner decision: every tournament winner on Qwen submits full weights, and offline adapter wins have not
# transferred live (Sept 14: 7/9 with an adapter on a short-row task). Short rows join on identical bases with the
# best full-weight recipe measured under Sept-14-like step cost (field + packing: parity with the adapter, ahead of
# the unpacked field route and the champion tree). FORGE_V2_FIELD_IDENTICAL_ROWS=long restores the long-row scope.
FIELD_IDENTICAL_BASE_ROWS_DEFAULT = "all"


def _field_identical_rows_policy() -> str:
    v = os.environ.get("FORGE_V2_FIELD_IDENTICAL_ROWS", FIELD_IDENTICAL_BASE_ROWS_DEFAULT).strip().lower()
    return v if v in ("long", "all") else FIELD_IDENTICAL_BASE_ROWS_DEFAULT


def identical_base_applies(is_long: bool) -> bool:
    """Whether the identical-base admission is even considered for a task of this row length."""
    return _field_identical_rows_policy() == "all" or bool(is_long)


def _field_identical_base_max_b() -> float:
    try:
        return float(os.environ.get("FORGE_V2_FIELD_IDENTICAL_MAX_PARAMS_B", str(FIELD_IDENTICAL_BASE_MAX_PARAMS_B)))
    except ValueError:
        return FIELD_IDENTICAL_BASE_MAX_PARAMS_B

# minimum size per family where the validated configuration applies: lfm2 was validated only on LFM2.5-2.6B, which takes
# the bf16 + paged 8-bit AdamW path (>= _field_bf16_min_b()); smaller lfm2 models would run an unvalidated fp32 path
FIELD_FULL_MIN_PARAMS_B: dict[str, float] = {"lfm2": 2.0}


# Damaged-base full route (2026-09-22). The validator damages ~half of the text bases (scaling / pruning / noise /
# re-init; norms and embeddings untouched). There adapter entries trailed the winner by a median ~19.8% against ~4.1%
# for full entries (small, confounded sample; research/20260922-codex-review); 20/21 winners shipped full weights.
# When forge.routing.damage says "damaged", an enabled family up to the FIELD size cap takes the FIELD full route with
# exactly the recipe its 2026-09-22/23 candidate arm ran on source c0dc300 (plans/*candidate*.json route_env):
#   FORGE_V2_FIELD=1, FORGE_V2_FIELD_TYPES=<family>, FORGE_V2_FIELD_BF16=1, FORGE_V2_FIELD_LR_MULT=1,
#   FORGE_V2_FIELD_BF16_MIN_B=0.3 where the arm set it (llama, qwen3, lfm2-1.2B; qwen2 got 0.3 from
#   FIELD_BF16_MIN_B_TYPES; gemma2 and lfm2-2.6B ran the 2.0 default), FORGE_V2_STRATEGY=full for lfm2 (redundant: the
#   field route forces strategy full). FIELD_TYPES=<family> bypassed the family, vocabulary and minimum-size gates; here
#   the family gate is the table below, the vocabulary gate is the tested vocabulary per family (None = bypassed, lfm2),
#   and the lfm2 2.0 B minimum does not apply. Packing is unchanged: forge.data.packing admits native Llama (128256) and
#   Qwen2 (151936) only, as executed (qwen2 / llama packed sdpa at 1472; gemma2 / lfm2 / qwen3 unpacked). Row / token
#   floors (15k / 8M; llama 10k / 5M) and the FIELD size cap apply unchanged. Clean and unknown bases keep today's route.
# Evidence (closed grammar 70k rows / ~21M tokens, arms at task hours, deployed default control vs FIELD bf16 full, every
# paired 99% interval wholly below zero; SCIENCE-LEDGER 2026-09-23 00:43-01:55 UTC; DISCOVERY_SIGNAL, one fixture):
#   qwen2   Qwen2.5-1.5B pruned (1.54 B) s17 -1.90%; Qwen2.5-3B-Instruct scaled (3.09 B) s17 -1.25%, s7 -1.47%
#   qwen3   Qwen3-1.7B-Base re-init (1.72 B) s17 -2.70% (needed FIELD_BF16_MIN_B=0.3)
#   llama   Llama-3.2-1B-Instruct re-init (1.24 B) s17 -1.54% (control: the old v2 adapter, not today's fp32 FIELD;
#           family OFF, see below)
#   gemma2  gemma-2-2b re-init (2.61 B) s17 -3.27% (0.75 h)
#   lfm2    LFM2.5-1.2B-Instruct pruned (1.17 B) s7 -0.64%, s17 -1.66%; LFM2.5-2.6B scaled (2.70 B) s17 -1.31%
# Sizes tested on damaged bases: 1.17-3.09 B (clean Qwen2-0.5B, 0.49 B, two seeds). Nothing above the 3.5 B FIELD cap.
DAMAGED_BASE_FAMILIES: dict[str, dict[str, Any]] = {
    "qwen2": {"enabled": True, "max_params_b": FIELD_FULL_MAX_PARAMS_B, "bf16_min_b": 0.3, "lr_mult": 1.0,
              "vocabs": frozenset({151936}), "tested_params_b": (1.54, 3.09)},
    "qwen3": {"enabled": True, "max_params_b": FIELD_FULL_MAX_PARAMS_B, "bf16_min_b": 0.3, "lr_mult": 1.0,
              "vocabs": frozenset({151936}), "tested_params_b": (1.72, 1.72)},
    # llama OFF (executor decision 2026-09-23): production already trains Llama-3.2-1B on the FIELD route in fp32 master
    # + AdamW; the llama pair tested the old v2 adapter vs bf16 FIELD, not production fp32 FIELD vs bf16, so switching a
    # damaged llama to bf16 / paged 8-bit is untested. Off = damaged llama bases keep exactly the c0dc300 route.
    "llama": {"enabled": False, "max_params_b": FIELD_FULL_MAX_PARAMS_B, "bf16_min_b": 0.3, "lr_mult": 1.0,
              "vocabs": frozenset({128256}), "tested_params_b": (1.24, 1.24)},
    "gemma2": {"enabled": True, "max_params_b": FIELD_FULL_MAX_PARAMS_B, "bf16_min_b": 2.0, "lr_mult": 1.0,
               "vocabs": frozenset({256000}), "tested_params_b": (2.61, 2.61)},
    # lfm2 <= 1.5 B (2026-09-24, replay-bench Phase H, LFM2.5-1.2B-Instruct pruned, cc4550ab, H100, 2 seeds): fp32 master
    # + LR x2 beat the bf16 x1 route by -3.3% / -3.5% (paired 99% CIs wholly below 0); bf16 rounding froze small updates.
    # bf16_min_b 1.5 keeps the geometry's fp32 master (+ fused AdamW) at 1.17 B; LFM2.5-2.6B (2.70 B) keeps bf16, LR x1.
    "lfm2": {"enabled": True, "max_params_b": FIELD_FULL_MAX_PARAMS_B, "bf16_min_b": 1.5, "lr_mult": 1.0,
             "small_max_b": 1.5, "small_lr_mult": 2.0, "vocabs": None, "tested_params_b": (1.17, 2.70)},
}
# Master switch. The validator runs this image with its own environment, so the production value lives here;
# FORGE_V2_DAMAGED_BASE_ROUTE=0/1 overrides it for research and CPU replays.
DAMAGED_BASE_ROUTE_DEFAULT = "1"
# Detectors consulted (forge.routing.damage.DETECTORS). "identity" needs a shipped public table (base_hashes/);
# "zero_fraction" (pruning only, reference-free) is opt-in via FORGE_V2_DAMAGE_DETECTORS until the executor admits it.
# "fingerprint" (2026-09-22) is the damage-detector README rule against the shipped offline table
# routing/data/base_fingerprints.json (norm-hash source identification + first-16 KiB prefix comparison, local
# pruning / bf16-scaling fallback); offline validation with the strict structure rule (46ce97e+): 28/29 damaged (all
# seven pair bases above) + 1 "unknown" (open_llama-3b-v2: the re-save drops 26 rotary inv_freq buffers), 54/54 clean
# (research/20260922-prod-patch/FINGERPRINT-TABLE.md). FORGE_V2_DAMAGE_DETECTORS=identity restores the 681901f set.
DAMAGE_DETECTORS_DEFAULT: tuple[str, ...] = ("identity", "fingerprint")

# Clean small-Qwen FIELD route: ON (production candidate 2026-09-24, Codex SIGN_OFF_FOR_SMOKE on 6da9d02). First evidence:
# clean unsloth/Qwen2-0.5B (0.49 B), FIELD bf16 packed full vs the deployed legacy LoRA, grammar 70k: s7 -1.49%, s17 -1.34%
# (research/20260922-{qwen05b,clean-qwen2-05b-s17}-exec); replay evidence below. A qwen2 base the detectors read
# "clean" up to CLEAN_SMALL_QWEN_MAX_PARAMS_B takes the field route with the recipe that arm ran (= the qwen2 row of
# DAMAGED_BASE_FAMILIES: bf16 from 0.3 B, LR x1.0). FORGE_V2_CLEAN_SMALL_QWEN=0/1 overrides it for research.
CLEAN_SMALL_QWEN_FIELD_ROUTE_DEFAULT = "1"
CLEAN_SMALL_QWEN_MAX_PARAMS_B = 1.0
# Data floors for the clean small-Qwen admission only (the generic FIELD floors 15k rows / 8M tokens and the damaged/identical
# qwen2 routes are unchanged). Replay evidence 2026-09-23/24 (research/20260923-replay-bench, SCIENCE-LEDGER): below the
# generic floors today's route trained clean Qwen2-0.5B slower (full FP32 unpacked, or the legacy LoRA handler); the
# FIELD bf16 route won on trdpo (9.3k rows / 2.8M tok: -2.3%, -2.6% over 2 seeds) and on the Dolly-TR analogue
# (14k rows / 3.7M tok: -5.8%). FORGE_V2_FIELD_MIN_ROWS / FORGE_V2_FIELD_MIN_TOKENS still override for research.
# 8000 / 2.5M (Codex review of 6da9d02): the narrower initial floor, still below both measured tasks; smaller data is extrapolation.
CLEAN_SMALL_QWEN_MIN_ROWS = 8000
CLEAN_SMALL_QWEN_MIN_TOKENS = 2_500_000


def _damaged_base_route() -> bool:
    return os.environ.get("FORGE_V2_DAMAGED_BASE_ROUTE", DAMAGED_BASE_ROUTE_DEFAULT).strip() == "1"


def damaged_base_types() -> frozenset[str]:
    """Families switched on for the damaged-base route. FORGE_V2_DAMAGED_BASE_TYPES (comma-separated; "none" = empty)
    overrides the table; a family missing from DAMAGED_BASE_FAMILIES is never admitted."""
    forced = os.environ.get("FORGE_V2_DAMAGED_BASE_TYPES", "").strip().lower()
    if forced:
        return frozenset(x.strip() for x in forced.split(",") if x.strip() in DAMAGED_BASE_FAMILIES)
    return frozenset(k for k, v in DAMAGED_BASE_FAMILIES.items() if v.get("enabled"))


def damaged_base_family_enabled(model_type: str | None) -> bool:
    """Whether the damaged-base route is on for this family."""
    return _field_full() and _damaged_base_route() and str(model_type or "").lower() in damaged_base_types()


# Window-aware eval governor: OFF by default, ON for the clean small-Qwen FIELD route (the only route with replay
# evidence: research/20260923-replay-bench phase-f/phase-g). FORGE_V2_EVAL_SLACK=0/1 overrides both.
EVAL_SLACK_DEFAULT = "0"


def _eval_slack(route_default: bool = False) -> bool:
    forced = os.environ.get("FORGE_V2_EVAL_SLACK", "").strip()
    if forced:
        return forced == "1"
    return route_default or EVAL_SLACK_DEFAULT == "1"


def governed_eval_every(eval_every: int, per_eval_s: float, t_step: float, total_steps: int,
                        window_s: float | None) -> tuple[int, dict | None]:
    """Dev-eval interval after the first eval. Flat rule: keep evaluation under ~6% of training time. Window-aware
    rule (window_s given, i.e. FORGE_V2_EVAL_SLACK=1): when the planned phase leaves wall-clock slack, spend up to half
    of it on denser evals; short full-weight runs (Qwen2-0.5B trdpo: 4 epochs in ~5 of 45 min) otherwise select from
    ~3 dev points. Never denser than the planned interval, never sparser than the flat rule. Returns (interval, info)."""
    widened = int(math.ceil(per_eval_s / (0.06 * max(t_step, 1e-3))))
    info = None
    if window_s:
        slack = window_s - total_steps * t_step
        if slack > 0:
            slack_every = max(eval_every, int(math.ceil(total_steps * per_eval_s / (0.5 * slack))))
            if slack_every < widened:
                info = {"flat_rule": widened, "slack_s": round(slack, 1), "per_eval_s": round(per_eval_s, 1)}
                widened = slack_every
    return widened, info


def seed_b_window(deadline: Any, finish_reserve: float, min_window: float) -> tuple[float, bool]:
    """Window for the seed-B plan, re-read after the seed-A greedy soup and adapter re-init (which spend wall clock:
    Codex review of 6912643). Returns (window_s, fits)."""
    window = deadline.remaining_hard() - finish_reserve
    return window, window >= min_window


class SeedARestoreError(RuntimeError):
    """Seed B no longer fits after the seed-A soup and the seed-A weights could not be restored onto the re-initialised
    adapter. Training must not continue: the live model is neither seed A nor a planned seed B."""


def restore_seed_a_or_fail(model: Any, seed_a: dict[str, Any], pool: Any, state: dict[str, Any]) -> None:
    """Put seed A back as the incumbent (as the stage-b failure path does) or fail closed with SeedARestoreError.
    Pool and selection state change only after a successful load."""
    try:
        selection.load_state(model, seed_a["state"])
    except Exception as exc:
        _event_and_print("sft_v2_two_seed_restore_failed", error=f"{type(exc).__name__}: {exc}")
        raise SeedARestoreError("seed-A restore failed after seed-B no-fit") from exc
    pool.items = [{"loss": seed_a["loss"], "step": seed_a["step"], "state": seed_a["state"]}]
    state["best"], state["best_step"] = seed_a["loss"], seed_a["step"]


def _clean_small_qwen() -> bool:
    return os.environ.get("FORGE_V2_CLEAN_SMALL_QWEN", CLEAN_SMALL_QWEN_FIELD_ROUTE_DEFAULT).strip() == "1"


def clean_small_qwen_enabled(model_type: str | None) -> bool:
    return _field_full() and _clean_small_qwen() and str(model_type or "").lower() == "qwen2"


# Damaged MiniCPM5-1B FIELD route (RELEASE CANDIDATE 2026-09-27, branch release-r1-noqwen-20260927; NOT released).
# Evidence: replay-bench Phase Y (research/20260923-replay-bench/phase-y/CLOSEOUT-20260927.md): one validator-damaged
# openbmb/MiniCPM5-1B (G.O.D dd8e898 layer_reinit multi_layer 0.0993; 1.0806 B), OpenOrca 26k fixture, external DEV-512,
# seed 7, 1.0 h per arm on 1 x H100. F1 = 52d319d + FORGE_V2_FIELD_TYPES=llama, FORGE_V2_FIELD_BF16=1,
# FORGE_V2_FIELD_BF16_MIN_B=0.3 (FIELD full BF16 at the formula LR, clamped to 1e-4): 1.624800 vs the live v2 LoRA
# 1.647194, -1.36%, paired 99% CI [-0.04499, -0.00132] (PASS_SEED7). F2 (LR x0.5) REJECT, +7.18%. Seed 17 and a second
# fixture / damage type are pending.
# Scope = what was tested: model_type "llama" with vocabulary 130560 (MiniCPM5; Llama-3.x is 128256) and
# DAMAGED_MINICPM5_MIN_PARAMS_B < params_b <= DAMAGED_MINICPM5_MAX_PARAMS_B, read "damaged" by the damage detectors.
# Damage detection runs for exactly that llama shape; every other llama base (Llama-3.x 128256, TinyLlama 32000,
# SmolLM2 49152, other sizes) keeps its 52d319d route with no detection pass, and a clean / unknown MiniCPM5 keeps the
# adapter route (FIELD_FULL_VOCABS still refuses 130560). DAMAGED_BASE_FAMILIES["llama"] stays disabled.
# Recipe = F1's executed env, through the damaged-base recipe path (field_recipe(damaged_base=True) reads the llama row's
# recipe fields: bf16_min_b 0.3, lr_mult 1.0, no small band): bf16 weights + paged 8-bit AdamW, LR = clamp(analytic,
# 2e-5, 1e-4) x 1.0 (1e-4 on this base), FIELD eff batch 100 (1-2 B), unpacked (packing qualifies Llama 128256 only),
# llama floors 10k rows / 5M tokens. The export is the plain LlamaForCausalLM save (no auto_map, no .py files).
# Switches: FORGE_V2_DAMAGED_MINICPM5=0 restores 52d319d exactly (no detection, no admission); the damaged-base master
# switch FORGE_V2_DAMAGED_BASE_ROUTE=0 and FORGE_V2_FIELD=0 also turn it off.
DAMAGED_MINICPM5_ROUTE_DEFAULT = "1"
DAMAGED_MINICPM5_MODEL_TYPE = "llama"
DAMAGED_MINICPM5_VOCAB = 130560
DAMAGED_MINICPM5_MIN_PARAMS_B = 0.9   # exclusive; tested 1.0806 B
DAMAGED_MINICPM5_MAX_PARAMS_B = 1.3   # inclusive


def _damaged_minicpm5() -> bool:
    return os.environ.get("FORGE_V2_DAMAGED_MINICPM5", DAMAGED_MINICPM5_ROUTE_DEFAULT).strip() == "1"


def damaged_minicpm5_shape(model_type: str | None, params_b: float | None, vocab_size: int | None) -> bool:
    """Switches on and the base has the tested MiniCPM5-1B shape (llama, vocabulary 130560, size band). An unknown
    vocabulary or size never matches."""
    if not (_field_full() and _damaged_base_route() and _damaged_minicpm5()):
        return False
    if str(model_type or "").lower() != DAMAGED_MINICPM5_MODEL_TYPE or vocab_size is None or params_b is None:
        return False
    try:
        vocab, size = int(vocab_size), float(params_b)
    except (TypeError, ValueError):
        return False
    return vocab == DAMAGED_MINICPM5_VOCAB and DAMAGED_MINICPM5_MIN_PARAMS_B < size <= DAMAGED_MINICPM5_MAX_PARAMS_B


def damaged_minicpm5_admits(model_type: str | None, base_damage: str | None, params_b: float | None,
                            vocab_size: int | None) -> bool:
    """Damaged MiniCPM5-1B admission to the damaged-base FIELD route (row / token floors applied by field_full_route)."""
    return base_damage == "damaged" and damaged_minicpm5_shape(model_type, params_b, vocab_size)


# Clean MiniCPM5-1B LR (release-candidate-v8, 2026-09-30; a directive-8 CONDITIONAL rule). On the full-FT default route
# the "minicpm5" recipe trains at LR = clamp(analytic, 2e-5, 1e-4) x CLEAN_MINICPM5_LR_MULT (x0.5) instead of x1.0 ONLY
# when the damage detectors read the base "clean" AND the one public reference that verdict names is exactly
# openbmb/MiniCPM5-1B (fingerprint table revision 87179e5c; forge.routing.damage.reference_id form). Every other verdict
# ("damaged", "unknown", no detection pass, a clean verdict naming another or no reference) keeps the release x1.0: a
# false "clean" costs ~+5 % (the damaged pairs), a false "not clean" only forgoes the ~2-3 % clean gain.
# Evidence: replay-bench phase-bc (research/20260923-replay-bench/phase-bc/CARD.md closeout, results/DEV-VERDICTS.json,
# image 7024722, arm lrh = FORGE_V2_FIELD_LR_MULT=0.5) with phase-ay (evidence/SUPPLEMENT-DEV-VERDICTS.json): clean
# dsum s7 -3.02 %, dsum s11 -1.94 %, wealth -1.97 % DEV, all PASS; damaged dsum +6.14 %, wealth +3.76 % DEV, both REJECT
# (so the rule is conditional, not general). Detector on MiniCPM5 (7024722 code, CPU): public openbmb/MiniCPM5-1B ->
# clean (prefix_identity, fingerprint name+norms); 5/5 validator-style augmented copies (scaling multi-layer 0.7,
# pruning all-layer 0.3, noise all-layer 0.05, layer re-init multi-layer 0.05, noise single-layer 0.02) and the phase-y
# custody damaged base -> damaged (prefix_mismatch) (research/20260927-new-pool-compat/minicpm5-1b/results/
# damage-probe.json; phase-ay/base/DAMAGE-DETECTOR-CHECK.json). There is no exact per-tensor identity table for
# MiniCPM5 (identity detector "unknown"), so "clean" rests on the first-16 KiB prefix of all 219 tensors plus full
# hashes of the 49 1-D tensors. FORGE_V2_CLEAN_MINICPM5_LR=0 restores 7024722; an explicit FORGE_V2_FIELD_LR_MULT wins.
# Applied in run_field_recipe (after the family recipe), so it never touches another recipe's multiplier.
CLEAN_MINICPM5_LR_DEFAULT = "1"
CLEAN_MINICPM5_LR_MULT = 0.5
CLEAN_MINICPM5_REFERENCES = frozenset({"openbmb/minicpm5-1b"})


def clean_minicpm5_lr_rule(recipe_key: str | None, base_damage: str | None,
                           base_clean_ref: str | None) -> tuple[float | None, str]:
    """(LR multiplier default, reason) of the clean MiniCPM5-1B rule; None = the release recipe's own multiplier."""
    if recipe_key != "minicpm5":
        return None, "not_minicpm5_recipe"
    if os.environ.get("FORGE_V2_CLEAN_MINICPM5_LR", CLEAN_MINICPM5_LR_DEFAULT).strip() != "1":
        return None, "switch_off"
    if base_damage != "clean":
        return None, f"verdict_{base_damage or 'not_assessed'}"
    from forge.routing.damage import reference_id

    if not base_clean_ref or reference_id(base_clean_ref) not in CLEAN_MINICPM5_REFERENCES:
        return None, "clean_reference_not_minicpm5"
    return CLEAN_MINICPM5_LR_MULT, "clean_minicpm5"


def _explicit_field_lr_mult() -> bool:
    """FORGE_V2_FIELD_LR_MULT is set to a number (it then wins over every default, as in _field_lr_mult)."""
    raw = os.environ.get("FORGE_V2_FIELD_LR_MULT", "").strip()
    if not raw:
        return False
    try:
        float(raw)
    except ValueError:
        return False
    return True


def damage_detection_enabled(model_type: str | None, *, vocab_size: int | None = None,
                             params_b: float | None = None) -> bool:
    """Whether the dispatcher should run damage detection for this family at all (for llama: only the damaged
    MiniCPM5-1B shape, which needs ``vocab_size`` and ``params_b``)."""
    return (damaged_base_family_enabled(model_type) or clean_small_qwen_enabled(model_type)
            or damaged_minicpm5_shape(model_type, params_b, vocab_size))


def damage_detectors() -> tuple[str, ...]:
    forced = os.environ.get("FORGE_V2_DAMAGE_DETECTORS", "").strip()
    if forced:
        return tuple(x.strip() for x in forced.split(",") if x.strip())
    return DAMAGE_DETECTORS_DEFAULT


def _damaged_family(model_type: str | None) -> dict[str, Any]:
    return DAMAGED_BASE_FAMILIES.get(str(model_type or "").lower()) or {}


def damaged_base_admits(model_type: str | None, base_damage: str | None, params_b: float,
                        vocab_size: int | None = None) -> bool:
    """Damage verdict + enabled family + tested vocabulary + size cap (never above FIELD_FULL_MAX_PARAMS_B). The
    remaining field gates (GPUs, KL, rows, tokens) are applied by field_full_route."""
    _mt = str(model_type or "").lower()
    if damaged_minicpm5_admits(_mt, base_damage, params_b, vocab_size):
        return True  # the scoped damaged MiniCPM5-1B admission (the llama family row itself stays disabled)
    if base_damage != "damaged" or not damaged_base_family_enabled(_mt):
        return False
    fam = _damaged_family(_mt)
    vocabs = fam.get("vocabs")
    if vocabs is not None and vocab_size is not None and int(vocab_size) not in vocabs:
        return False  # a family member with an untested vocabulary (e.g. TinyLlama 32000 as "llama") keeps its route
    try:
        cap = float(fam.get("max_params_b") or FIELD_FULL_MAX_PARAMS_B)
    except (TypeError, ValueError):
        cap = FIELD_FULL_MAX_PARAMS_B
    return 0 < params_b <= min(cap, FIELD_FULL_MAX_PARAMS_B)


def clean_small_qwen_min_rows() -> int:
    try:
        return int(os.environ.get("FORGE_V2_FIELD_MIN_ROWS", str(CLEAN_SMALL_QWEN_MIN_ROWS)))
    except ValueError:
        return CLEAN_SMALL_QWEN_MIN_ROWS


def clean_small_qwen_min_tokens() -> int:
    try:
        return int(float(os.environ.get("FORGE_V2_FIELD_MIN_TOKENS", str(CLEAN_SMALL_QWEN_MIN_TOKENS))))
    except ValueError:
        return CLEAN_SMALL_QWEN_MIN_TOKENS


def clean_small_qwen_admits(model_type: str | None, base_damage: str | None, params_b: float,
                            vocab_size: int | None = None) -> bool:
    """Clean small-Qwen field admission (default ON since 2026-09-24; see CLEAN_SMALL_QWEN_FIELD_ROUTE_DEFAULT)."""
    if base_damage != "clean" or not clean_small_qwen_enabled(model_type):
        return False
    if vocab_size is not None and int(vocab_size) not in DAMAGED_BASE_FAMILIES["qwen2"]["vocabs"]:
        return False
    return 0 < params_b <= min(CLEAN_SMALL_QWEN_MAX_PARAMS_B, FIELD_FULL_MAX_PARAMS_B)


# Clean LFM2.5-1.2B-Instruct FIELD route (CONTINGENT RELEASE CANDIDATE 2026-09-28, branch release-clean-lfm12-field-20260928).
# Ships only if replay-bench Phase AJ (research/20260923-replay-bench/phase-aj/CARD.md: clean public LFM2.5-1.2B-Instruct
# @0f604ada, Turkish cc4550ab reduced TRAIN 80,704 rows / 16.79M tokens, row median 179, seed 7, 1.0 h per arm on 1 x H100)
# meets the owner rule: arm F (11c41a7 + FORGE_V2_FIELD_TYPES=lfm2 + FORGE_V2_FIELD_LR_MULT=2) clearly beats arm C
# (11c41a7 as deployed: the legacy production LoRA on short rows, the v2 LoRA on long rows), or C is worse than untrained.
# The route is exactly what F executes, now without env, on the SHORT-ROW branch only (RESP-065, 2026-09-28): a task whose
# sampled row median is below the v2 row gate (long_row_task False, median < LORA_MIN_MEDIAN_TOKENS = 256), where
# 11c41a7 runs the legacy production LoRA and where Phase AJ measured. Long-row tasks keep the v2 LoRA (no scored
# evidence against that incumbent). There: FIELD full weights (the route forces strategy full), field_recipe(damaged_base=False) = bf16 False at 1.17 B (< the 2.0 B
# bf16 minimum) -> the geometry's fp32 master + fused AdamW, LR = clamp(analytic, 2e-5, 1e-4) x 2, FIELD eff batch 100,
# unpacked (packing admits Llama / Qwen2 only), and the generic FIELD floors 15k rows / 8M tokens (F's floors: the
# FIELD_TYPES env bypassed only the family, vocabulary and 2.0 B minimum-size gates). F on the same base = the recipe
# production already ships for the damaged small lfm2 band (DAMAGED_BASE_FAMILIES lfm2: fp32 master, LR x2).
# Scope = what was tested: model_type lfm2, a "clean" damage verdict whose named public reference is exactly
# LiquidAI/LFM2.5-1.2B-Instruct (fingerprint prefix identity; the task's model name is NOT trusted), 0 < params_b <= 1.5.
# LFM2-1.2B, LFM2.5-1.2B-Base, LFM2(.5)-350M / 700M, unknown or damaged lfm2 bases and LFM2.5-2.6B keep their 11c41a7
# routes (a damaged 1.2B already takes the damaged-base small band). FORGE_V2_CLEAN_SMALL_LFM2=0 restores 11c41a7 exactly.
CLEAN_SMALL_LFM2_ROUTE_DEFAULT = "1"
CLEAN_SMALL_LFM2_MODEL_TYPE = "lfm2"
CLEAN_SMALL_LFM2_REFERENCES = frozenset({"liquidai/lfm2.5-1.2b-instruct"})  # forge.routing.damage.reference_id form
CLEAN_SMALL_LFM2_MAX_PARAMS_B = 1.5  # inclusive; tested 1.1703 B
CLEAN_SMALL_LFM2_LR_MULT = 2.0


def _clean_small_lfm2() -> bool:
    return os.environ.get("FORGE_V2_CLEAN_SMALL_LFM2", CLEAN_SMALL_LFM2_ROUTE_DEFAULT).strip() == "1"


def clean_base_reference(report: dict[str, Any] | None) -> str | None:
    """The one public reference (forge.routing.damage.reference_id form) that a "clean" damage.assess verdict names; None
    for any other verdict, a reference-free verdict or disagreeing references."""
    if not isinstance(report, dict) or report.get("verdict") != "clean":
        return None
    from forge.routing.damage import reference_id

    refs = {reference_id(r.get("reference") or r.get("source")) for r in (report.get("detectors") or {}).values()
            if isinstance(r, dict) and r.get("verdict") == "clean"} - {None}
    return next(iter(refs)) if len(refs) == 1 else None


def clean_small_lfm2_admits(model_type: str | None, base_damage: str | None, params_b: float | None,
                            base_clean_ref: str | None = None, *, long_rows: bool | None = None) -> bool:
    """Clean LFM2.5-1.2B-Instruct FIELD admission on the short-row branch only (``long_rows`` must be exactly False, as
    long_row_task reports it; None / True keep 11c41a7). Row / token floors and the size cap are applied by field_full_route."""
    if not (_field_full() and _clean_small_lfm2()) or long_rows is not False:
        return False
    if str(model_type or "").lower() != CLEAN_SMALL_LFM2_MODEL_TYPE or base_damage != "clean" or not base_clean_ref:
        return False
    from forge.routing.damage import reference_id

    if reference_id(base_clean_ref) not in CLEAN_SMALL_LFM2_REFERENCES:
        return False
    try:
        size = float(params_b)
    except (TypeError, ValueError):
        return False
    return 0 < size <= min(CLEAN_SMALL_LFM2_MAX_PARAMS_B, FIELD_FULL_MAX_PARAMS_B)


def field_full_route(*, params_b: float, n_gpus: int, is_kl: bool, model_type: str | None = None,
                     n_rows: int | None = None, total_tokens: int | None = None, vocab_size: int | None = None,
                     base_identity: str | None = None, base_candidate: str | None = None,
                     base_damage: str | None = None, base_clean_ref: str | None = None,
                     long_rows: bool | None = None) -> bool:
    """Full weights at the field geometry for single-GPU, non-KL instruct tasks up to
    FIELD_FULL_MAX_PARAMS_B (FORGE_V2_FIELD_MAX_PARAMS_B overrides) on an allow-listed
    model family (field_full_types) with at least field_full_min_rows() rows: full weights
    memorise a small corpus within a few steps (CodeFeedback 8k: best dev at step 10 at both
    1e-4 and 4.8e-5) while they keep improving for two epochs on 19.5k GossipCop rows.
    On by default; FORGE_V2_FIELD=0 disables it. ``base_identity`` (forge.routing.base_identity status) admits
    FIELD_IDENTICAL_BASE_TYPES only when it is "identical" and FORGE_V2_FIELD_IDENTICAL_BASE=1. ``base_damage``
    (forge.routing.damage verdict) admits an enabled DAMAGED_BASE_FAMILIES family on a "damaged" base (tested vocabulary,
    no family minimum size) and, only when switched on, a small qwen2 base read "clean" (clean_small_qwen_admits);
    the row and token floors and the size cap still apply. ``base_clean_ref`` (clean_base_reference of the damage report)
    admits the clean LFM2.5-1.2B-Instruct (clean_small_lfm2_admits; short-row tasks only, ``long_rows`` False) past the lfm2 vocabulary and 2.0 B minimum gates, as
    FORGE_V2_FIELD_TYPES=lfm2 did in Phase AJ arm F; the family, floors and cap still apply."""
    if not _field_full() or is_kl or n_gpus != 1 or params_b <= 0:
        return False
    _mt = str(model_type).lower() if model_type is not None else None
    damaged_admit = damaged_base_admits(_mt, base_damage, params_b, vocab_size)
    csq_admit = not damaged_admit and clean_small_qwen_admits(_mt, base_damage, params_b, vocab_size)
    damaged = damaged_admit or csq_admit
    csl_admit = not damaged and clean_small_lfm2_admits(_mt, base_damage, params_b, base_clean_ref, long_rows=long_rows)
    if damaged:
        pass  # the family allow-list, FIELD_FULL_VOCABS and minimum-size gates below do not apply; rows / tokens / cap do
    elif _mt is not None and _mt not in field_full_types():
        # a family outside the allow-list is admitted only on a byte-identical public base (see FIELD_IDENTICAL_BASE_TYPES)
        if not (_mt in FIELD_IDENTICAL_BASE_TYPES and base_identity == "identical" and _field_identical_base()):
            return False
        _rule = FIELD_IDENTICAL_BASE_RULES.get(base_candidate or "", {})
        # FORGE_V2_FIELD_IDENTICAL_MAX_PARAMS_B (research) overrides every per-candidate cap
        _cap = (_field_identical_base_max_b() if os.environ.get("FORGE_V2_FIELD_IDENTICAL_MAX_PARAMS_B")
                else float(_rule.get("max_params_b") or FIELD_IDENTICAL_BASE_MAX_PARAMS_B))
        if params_b > _cap:
            return False
        _floor = _rule.get("min_tokens")
        if _floor and not os.environ.get("FORGE_V2_FIELD_MIN_TOKENS") and (total_tokens is None or total_tokens < int(_floor)):
            return False  # this checkpoint's win was measured on large corpora only; unknown volume keeps the route
    if vocab_size is not None and not damaged and not csl_admit and not os.environ.get("FORGE_V2_FIELD_TYPES"):
        # family-specific: the validated wins are Meta Llama-3.2 (128256) and LFM2.5 (128000); an unknown family
        # or a mismatched vocabulary keeps its route
        if _mt is None or int(vocab_size) not in FIELD_FULL_VOCABS.get(_mt, frozenset()):
            return False
    if (_mt is not None and not damaged and not csl_admit and not os.environ.get("FORGE_V2_FIELD_TYPES")
            and params_b < FIELD_FULL_MIN_PARAMS_B.get(_mt, 0.0)):
        return False  # below the size where the family's validated configuration applies
    min_rows = clean_small_qwen_min_rows() if csq_admit else field_full_min_rows(_mt)
    min_tokens = clean_small_qwen_min_tokens() if csq_admit else field_full_min_tokens(_mt)
    if n_rows is not None and n_rows < min_rows:
        return False
    if total_tokens is not None and total_tokens < min_tokens:
        # data volume, not row count, separates the wins from the losses: full weights
        # overfit small token budgets within an epoch at every tested LR
        return False
    try:
        cap = float(os.environ.get("FORGE_V2_FIELD_MAX_PARAMS_B", str(FIELD_FULL_MAX_PARAMS_B)))
    except ValueError:
        cap = FIELD_FULL_MAX_PARAMS_B
    return params_b <= cap


# ---- Full fine-tuning by default (release-fullft-default-20260929) ----
# Owner standing directives 1-4 (SN56-text/AGENTS.md, 2026-09-28): every single-GPU, non-KL instruct task up to 4.2 B
# parameters trains full weights. There is no family allow-list, row / token floor, row-length gate, vocabulary gate or
# minimum size; identity and damage-detector verdicts may choose hyperparameters, never adapter vs full weights. Adapters
# run only for a named, dated exception (CURRENT-CAMPAIGN "Adapter exceptions") or as the crash fallback of last resort
# (instruct.run: full FT -> cheaper full FT -> adapter). The admission above (field_full_route and its allow-lists and
# floors) is no longer consulted for in-scope tasks; it still documents which recipe each family measured.
# FORGE_V2_FULLFT_DEFAULT=0 restores cab930c's routing exactly (research replays only; the validator sets no env).
FULLFT_DEFAULT_ROUTE_DEFAULT = "1"
# "up to 4.2B parameters" (owner scope). Qwen3.5-4B loads text-only at 4.2058 B (research/20260925-release-lfm-q35drift/
# CPU-PRECHECK.json), so the numeric cap is 4.25 to keep the 4B class inside the owner's rounded 4.2 B. The next pool size
# above it is Qwen3.5-4B loaded with its vision tower (4.66 B) and the 7-9 B class; nothing in the pool sits in (4.21, 4.25].
FULLFT_DEFAULT_MAX_PARAMS_B = 4.25
# Named, dated quality exceptions (CURRENT-CAMPAIGN "Adapter exceptions"); matched in instruct.run by the loaded
# architecture, never by the request name:
#   falcon-rw-1b                    our adapter placed 1st on 08-31 (instruct.eligible_falcon_rw1b_legacy_route)
# REMOVED 2026-09-29: "lfm2.5-2.6b-short-rows" (tokenized median < 256 kept the pinned one-epoch adapter; owner decision
#   2026-09-28 "drop the exception if full FT wins"). replay-bench phase-an (research/20260923-replay-bench/phase-an/
#   REDUCE-20260929.md) on the task that earned it (29fc1a10, clean LFM2.5-2.6B, seed 7): full FT (family recipe = what
#   FORGE_V2_TAKES_LFM25=1 selected) vs the pinned adapter, DEV -0.0408 99% [-0.0593, -0.0248] PASS; historical -3.57%;
#   1st of 13 on the retained 09-07 board (adapter 3rd). LFM2.5-2.6B tasks in scope now train full weights at every row
#   length, as long rows already did; the pin still applies out of scope (cab930c).
QUALITY_EXCEPTIONS = ("falcon-rw-1b",)
LFM25_26B_SHORT_ROW_MEDIAN = 256
# The LFM2.5-2.6B pin (forge.tuning.lfm25_epoch_cap._matches_base_model) matches the architecture only, which LFM2-2.6B
# (vocabulary 65536) shares; the exception names LFM2.5-2.6B, so it also requires LFM2.5's vocabulary.
LFM25_26B_VOCAB = 128000
# Crash exceptions: model_type -> {"since": "YYYY-MM-DD", "test": "<failing crash-sweep item / receipt path>",
# optional "min_params_b" / "max_params_b" (inclusive band)}. A family listed here skips full weights and trains the
# fixed legacy adapter. Empty until the crash sweep (research/20260929-release-fullft-default/CRASH-SWEEP-PLAN.md) names a
# failing family; an entry expires when the crash is fixed. FORGE_V2_CRASH_EXCEPTIONS="type[,type]" adds entries for tests.
CRASH_EXCEPTIONS: dict[str, dict[str, Any]] = {}
# Unmeasured families (no row in DAMAGED_BASE_FAMILIES and not llama): FIELD geometry at the clamped analytic LR x1.0
# (every measured full-weight family win ran x1.0 except the fp32 LFM2 small band; x0.5 lost on MiniCPM5 (+7.2%) and on
# the exact Sept-14 Qwen task), fp32 master + fused AdamW below 2.0 B (the Llama production path; bf16 rounding froze
# small updates on LFM2.5-1.2B, Phase H) and bf16 weights + paged 8-bit AdamW from 2.0 B (the FIELD default threshold;
# fp32 master at 3-4 B is throughput- and memory-bound: Gemma-2-2B 0.49 epoch in 45 min, the 76 GB OOM).
GENERIC_FIELD_BF16_MIN_B = 2.0
GENERIC_FIELD_LR_MULT = 1.0
# lfm2 small band on the full-FT default route (release-candidate-v8, 2026-09-30). The "family" recipe of an lfm2 base
# with 0 < params_b <= DAMAGED_BASE_FAMILIES["lfm2"]["small_max_b"] (1.5 B) keeps its fp32 master + fused AdamW and
# bucket, but trains at LR = clamp(analytic, 2e-5, 1e-4) x FULLFT_LFM2_SMALL_LR_MULT instead of the band's
# small_lr_mult (x2, 7024722). Evidence: replay-bench phase-bb (research/20260923-replay-bench/phase-bb/results/
# REDUCED.md; image 7024722, arm B = FORGE_V2_FIELD_LR_MULT=1, which is exactly this code path), LFM2.5-1.2B-Instruct
# (1.1703 B): clean hko1aw s7 -2.80 % / s11 -2.73 % DEV PASS, komedcl -1.16 % PASS; damaged (pruned cc4550ab) turkish
# s7 -0.43 % / s11 -0.81 %, komed -0.44 % HOLD (every CI spans 0, none worse). trclean was NOT_RUN at build time.
# Band members never measured at either multiplier: LFM2-350M, LFM2-700M, LFM2-1.2B, LFM2.5-350M, LFM2.5-1.2B-Base.
# The cab930c research routes (FORGE_V2_FULLFT_DEFAULT=0: damaged small band, clean LFM2.5-1.2B-I route) keep x2.
# None = 7024722 (the band's small_lr_mult). An explicit FORGE_V2_FIELD_LR_MULT still overrides.
FULLFT_LFM2_SMALL_LR_MULT: float | None = 1.0


def _fullft_default() -> bool:
    return _field_full() and os.environ.get("FORGE_V2_FULLFT_DEFAULT", FULLFT_DEFAULT_ROUTE_DEFAULT).strip() == "1"


def fullft_default_max_params_b() -> float:
    try:
        return float(os.environ.get("FORGE_V2_FULLFT_MAX_PARAMS_B", str(FULLFT_DEFAULT_MAX_PARAMS_B)))
    except ValueError:
        return FULLFT_DEFAULT_MAX_PARAMS_B


def crash_exceptions() -> dict[str, dict[str, Any]]:
    table = dict(CRASH_EXCEPTIONS)
    for mt in os.environ.get("FORGE_V2_CRASH_EXCEPTIONS", "").split(","):
        if mt.strip():
            table.setdefault(mt.strip().lower(), {"since": "env", "test": "FORGE_V2_CRASH_EXCEPTIONS"})
    return table


def crash_exception(model_type: str | None, params_b: float | None) -> str | None:
    """The crash exception naming this family (and size band), else None."""
    entry = crash_exceptions().get(str(model_type or "").lower())
    if entry is None:
        return None
    try:
        size = float(params_b or 0.0)
    except (TypeError, ValueError):
        size = 0.0
    lo, hi = entry.get("min_params_b"), entry.get("max_params_b")
    if (lo is not None and size < float(lo)) or (hi is not None and size > float(hi)):
        return None
    return f"crash:{str(model_type).lower()}:{entry.get('since')}:{entry.get('test')}"


def fullft_scope_refusal(*, task_type: str | None, has_instruct: bool, is_kl: bool, n_gpus: int, params_b: float,
                         quasar: bool = False) -> str | None:
    """Why a task is outside the full-FT-default scope (None = in scope). Scope = owner directive scope: single-GPU,
    non-KL InstructTextTask up to FULLFT_DEFAULT_MAX_PARAMS_B. Out-of-scope tasks keep cab930c's routing unchanged."""
    if not _fullft_default():
        return "switch_off"
    if task_type != "InstructTextTask" or not has_instruct:
        return "not_instruct_task"  # ChatTask / DPO / GRPO keep their handlers
    if is_kl:
        return "kl"
    if n_gpus != 1:
        return "not_single_gpu"
    if quasar:
        return "quasar_remote_code"  # 8.6 B anyway; sft_v2.eligible refuses it
    if not (0 < params_b <= fullft_default_max_params_b()):
        return "size"
    return None


@dataclass(frozen=True)
class RouteDecision:
    """Full-FT-default dispatcher verdict for an in-scope task."""

    route: str  # "full" | "adapter_exception"
    reason: str  # "default" or the exception name
    recipe: str = ""  # default_recipe_key() for "full"


def default_recipe_key(model_type: str | None, params_b: float, vocab_size: int | None) -> str:
    """Which measured recipe a full-FT-default task trains with. Independent of the damage verdict (directive 3: a
    family's measured recipe ships clean and damaged):
      "minicpm5"  llama, vocabulary 130560, 0.9 < params_b <= 1.3: the damaged MiniCPM5-1B F1 recipe (bf16 + paged 8-bit,
                  LR clamp(analytic) x1.0 = 1e-4 on that base; Phase Y)
      "family"    a DAMAGED_BASE_FAMILIES row (qwen2, qwen3, gemma2, lfm2): that row's recipe (qwen2/qwen3 bf16 from 0.3 B
                  x1.0 = the Qwen FIELD recipe; gemma2 bf16 x1.0 + the champion port in its size band; lfm2 fp32 master
                  up to 1.5 B at LR x FULLFT_LFM2_SMALL_LR_MULT (x1; 7024722 x2), bf16 x1.0 above)
      "llama"     every other llama (Llama-3.2, TinyLlama, SmolLM/SmolLM2, open_llama): the production Llama FIELD path
                  (fp32 master + fused AdamW up to FP32_MASTER_MAX_B, x1.0)
      "generic"   everything else (phi3, mistral/Phi-3, phi, gpt_neox, falcon, bloom, gemma, granite, qwen3_5, ...):
                  GENERIC_FIELD_* above."""
    mt = str(model_type or "").lower()
    try:
        vocab = int(vocab_size) if vocab_size is not None else None
    except (TypeError, ValueError):
        vocab = None
    if (mt == DAMAGED_MINICPM5_MODEL_TYPE and vocab == DAMAGED_MINICPM5_VOCAB
            and DAMAGED_MINICPM5_MIN_PARAMS_B < float(params_b) <= DAMAGED_MINICPM5_MAX_PARAMS_B):
        return "minicpm5"
    if mt in DAMAGED_BASE_FAMILIES and DAMAGED_BASE_FAMILIES[mt].get("enabled"):
        return "family"
    if mt == "llama":
        return "llama"
    return "generic"


def fullft_default_route(*, model_type: str | None, params_b: float, vocab_size: int | None,
                         falcon_rw1b: bool = False, lfm25_26b: bool = False, median_tokens: int | None = None,
                         long_rows: bool | None = None) -> RouteDecision:
    """Route of an IN-SCOPE task (fullft_scope_refusal returned None). The only non-full outcomes are the named
    exceptions. ``falcon_rw1b`` / ``lfm25_26b``: the loaded architecture matched the exception's base
    (instruct.eligible_falcon_rw1b_legacy_route / the LFM2.5-2.6B pin match); ``median_tokens``: sampled tokenized row
    median (long_row_task). The damage verdict is deliberately not an input."""
    if falcon_rw1b:
        return RouteDecision("adapter_exception", "quality:falcon-rw-1b")
    # lfm25_26b / median_tokens / long_rows: kept for telemetry and callers; the LFM2.5-2.6B short-row quality exception
    # was removed on 2026-09-29 (phase-an: full FT won), so they no longer refuse full weights
    crash = crash_exception(model_type, params_b)
    if crash:
        return RouteDecision("adapter_exception", crash)
    return RouteDecision("full", "default", default_recipe_key(model_type, params_b, vocab_size))


def legacy_adapter_fixes_enabled(*, fallback: bool, crash_exception_route: bool, quality_exception_route: bool) -> bool:
    """Legacy adapter fixes (forge.tuning.legacy_adapter_fixes) run on the crash-fallback adapter and on crash
    exceptions of in-scope tasks. The quality exceptions keep the exact adapter that earned the exception on a real board
    unless FORGE_LEGACY_FIXES_ON_EXCEPTIONS=1 (their recipe is the evidence; changing it is unmeasured)."""
    if os.environ.get("FORGE_LEGACY_ADAPTER_FIXES", "1").strip() == "0":
        return False
    if quality_exception_route:
        return os.environ.get("FORGE_LEGACY_FIXES_ON_EXCEPTIONS", "0").strip() == "1"
    return bool(fallback or crash_exception_route)


class ZeroTrainedStepsError(RuntimeError):
    """sft_v2 finished without a single optimizer step. The output holds only the untrained floor, which must never be
    shipped while a cheaper rung can still train: the dispatcher catches this and moves to the next rung."""


FULLFT_EXTEND_MIN_WINDOW_S = 300.0


def fullft_epoch_cap_extension(*, recipe_key: str | None, phase: int, extended: bool, last_phase_complete: bool,
                               stopped_overfit: bool, best_step: int, step_offset: int, epochs_done: float,
                               max_epochs: float, window_s: float, min_window_s: float, t_step: float | None,
                               steps_per_epoch: int) -> float:
    """Extra epochs for ONE continuation phase of a full-FT default run whose epoch cap ended training early (window
    use, 2026-09-29), else 0.

    The FIELD route trains one cosine over min(epoch cap, what fits). When the cap binds, the rest of the paid window is
    idle: the live NuExtract task stopped at the 3-epoch cap after 2,306 of 4,500 s with the DEV loss improving at every
    evaluation (0.8060 -> 0.8023 -> 0.7984). Continuing is only allowed when the DEV loss says training still helps:
    phase 0 completed its plan (the cap bound, not the clock), no overfit stop, the best DEV loss is the phase's LAST
    evaluation, and at least FULLFT_EXTEND_MIN_WINDOW_S / 25 steps of window remain. The continuation is a fresh
    annealed cosine at the restart LR (x0.5) over at most one more cap of epochs, planned inside the window. The best-
    snapshot pool keeps the phase-0 end, the overfit early stop is active and the greedy soup is monotone, so on DEV the
    shipped weights are never worse than stopping at the cap. FORGE_V2_FULLFT_EXTEND=0 switches it off."""
    if os.environ.get("FORGE_V2_FULLFT_EXTEND", "1").strip() == "0":
        return 0.0
    if not recipe_key or extended or phase != 1 or not last_phase_complete or stopped_overfit or not t_step:
        return 0.0
    if best_step <= 0 or best_step != step_offset:
        return 0.0  # DEV was not improving at the end of the capped phase
    if epochs_done < max_epochs - 1e-3:
        return 0.0  # the cap did not bind (the clock or the planner ended the phase)
    if window_s < max(FULLFT_EXTEND_MIN_WINDOW_S, min_window_s):
        return 0.0
    spe = max(1, int(steps_per_epoch))
    extra = round(max(0.0, min(float(max_epochs), window_s * 0.85 / float(t_step) / spe)), 3)
    if math.ceil(spe * extra) < max(10, int(0.15 * spe)):
        return 0.0  # below the planner's minimum continuation phase
    return extra


class PersistFailedError(RuntimeError):
    """sft_v2 trained but no trained artifact could be written (release defect D5). Raised on the full-FT default route
    only, so the dispatcher's ladder retries instead of shipping the untrained floor with a clean exit."""


CHEAP_FULL_MIN_REMAINING_S = 900.0  # a cheaper full-FT retry needs a reload + tokenization + at least a few hundred steps


FIELD_BF16_MIN_B_TYPES: dict[str, float] = {"qwen2": 0.3}  # qwen2 validated at 0.494 B with bf16 + paged 8-bit AdamW


def _identical_base_family_defaults(model_type: str | None) -> bool:
    """True when ``model_type`` is an identical-base family and that admission is switched on."""
    return str(model_type or "").lower() in FIELD_IDENTICAL_BASE_TYPES and _field_identical_base()


def _field_bf16_min_b(model_type: str | None = None, *, damaged_base: bool = False, generic: bool = False) -> float:
    default = FIELD_BF16_MIN_B_TYPES.get(str(model_type or "").lower(), 2.0) if _identical_base_family_defaults(model_type) else 2.0
    if generic and not damaged_base:
        default = GENERIC_FIELD_BF16_MIN_B
    if damaged_base:
        default = float(_damaged_family(model_type).get("bf16_min_b", 2.0))  # the threshold that family's arm ran
    try:
        return float(os.environ.get("FORGE_V2_FIELD_BF16_MIN_B", str(default)))
    except ValueError:
        return default


FIELD_BF16_DEFAULT_TYPES = frozenset({"lfm2"})  # families whose field win was measured with bf16 weights + paged 8-bit AdamW
# qwen2 joins the bf16 / half-raw-LR defaults ONLY while the identical-base admission is switched on (every validated
# Qwen2.5-0.5B full-weight arm ran FORGE_V2_FIELD_BF16=1, BF16_MIN_B=0.3, LR_MULT=0.5); with the switch off nothing changes.


def _field_bf16_enabled(model_type: str | None, *, damaged_base: bool = False, generic: bool = False) -> bool:
    """bf16 weights + paged 8-bit AdamW on the field route (>= _field_bf16_min_b() B). FORGE_V2_FIELD_BF16=1/0
    forces it; unset, it is on only for FIELD_BF16_DEFAULT_TYPES (lfm2: the 2026-09-16 MathFusion win ran this way,
    1.7 epochs in 2 h; fp32 master at the same size managed 0.49–2.2 partial epochs). Llama-1B never reaches the
    size threshold, so its deployed fp32 path is unchanged."""
    forced = os.environ.get("FORGE_V2_FIELD_BF16", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    return (damaged_base or generic or str(model_type or "").lower() in FIELD_BF16_DEFAULT_TYPES
            or _identical_base_family_defaults(model_type))


# qwen2 (Qwen2.5-0.5B): 0.5 until 2026-09-18, chosen on a chat-style stand-in fixture (x1.0 lost 2.2% there, one seed each).
# On the EXACT Sept 14 live task (validator train + test rows, official evaluator, matched seeds 7/11) the raw law wins:
# x0.5 1.29943/1.30013 (mean 1.29978) vs x1.0 1.28701/1.29211 (mean 1.28956), -0.0102 nats, same order on both seeds,
# against a rule fixed before the second seed (>= 0.005). Same bf16 path and 16-minute schedule, so no window risk.
# Limits: one task, first selected on those rows. research/20260918-exact-replays, SCIENCE-LEDGER 2026-09-18 15:25 UTC.
FIELD_LR_MULT_TYPES: dict[str, float] = {"qwen2": 1.0}


def _field_lr_mult(model_type: str | None = None, *, damaged_base: bool = False, params_b: float | None = None,
                   clean_small_lfm2: bool = False, recipe_key: str | None = None) -> float:
    """Multiplier on the clamped raw analytic LR of the field route: FORGE_V2_FIELD_LR_MULT, else the family default
    (a damaged family may set a size-scoped small_lr_mult for params_b <= small_max_b; the clean LFM2.5-1.2B-Instruct
    route uses CLEAN_SMALL_LFM2_LR_MULT). ``recipe_key``: the full-FT default recipe (run_field_recipe; None on the
    cab930c routes and for callers that pass no key)."""
    default = FIELD_LR_MULT_TYPES.get(str(model_type or "").lower(), 1.0) if _identical_base_family_defaults(model_type) else 1.0
    if clean_small_lfm2 and not damaged_base:
        default = CLEAN_SMALL_LFM2_LR_MULT
    if damaged_base:
        fam = _damaged_family(model_type)
        default = float(fam.get("lr_mult", 1.0))
        if params_b is not None and fam.get("small_max_b") is not None and 0 < params_b <= float(fam["small_max_b"]):
            default = float(fam.get("small_lr_mult", default))
            if (recipe_key == "family" and str(model_type or "").lower() == "lfm2"
                    and FULLFT_LFM2_SMALL_LR_MULT is not None):
                default = float(FULLFT_LFM2_SMALL_LR_MULT)  # full-FT default lfm2 small band (v8, phase-bb)
    try:
        return float(os.environ.get("FORGE_V2_FIELD_LR_MULT", str(default)) or default)
    except ValueError:
        return default


def field_recipe(model_type: str | None, params_b: float, *, damaged_base: bool = False,
                 clean_small_lfm2: bool = False, generic: bool = False, recipe_key: str | None = None) -> dict[str, Any]:
    """Field-route recipe knobs sft_v2.run applies (env overrides included): bf16 weights + paged 8-bit AdamW (instead of
    the fp32 master when the geometry would choose it), its size threshold, the LR multiplier and the rows-per-step
    bucket. ``damaged_base`` selects the family's DAMAGED_BASE_FAMILIES recipe; ``clean_small_lfm2`` (the clean
    LFM2.5-1.2B-Instruct route) changes only the LR multiplier default (x2, = Phase AJ arm F's FORGE_V2_FIELD_LR_MULT);
    ``generic`` (full-FT default, unmeasured family) = bf16 from GENERIC_FIELD_BF16_MIN_B, LR x GENERIC_FIELD_LR_MULT."""
    generic = bool(generic and not damaged_base)
    min_b = _field_bf16_min_b(model_type, damaged_base=damaged_base, generic=generic)
    # a damaged family's small band (0 < params_b <= small_max_b) is fp32 master + small_lr_mult, boundary included,
    # matching _field_lr_mult's <= (without this, exactly small_max_b == bf16_min_b got bf16 + the small LR). It is a
    # DEFAULT only: an explicit FORGE_V2_FIELD_BF16 / FORGE_V2_FIELD_BF16_MIN_B research control keeps precedence.
    small_max = _damaged_family(model_type).get("small_max_b") if damaged_base else None
    explicit_precision = any(os.environ.get(k, "").strip() for k in ("FORGE_V2_FIELD_BF16", "FORGE_V2_FIELD_BF16_MIN_B"))
    small = small_max is not None and 0 < params_b <= float(small_max) and not explicit_precision
    return {"bf16": bool(_field_bf16_enabled(model_type, damaged_base=damaged_base, generic=generic)
                         and params_b >= min_b and not small),
            "bf16_min_b": min_b, "lr_mult": _field_lr_mult(model_type, damaged_base=damaged_base, params_b=params_b,
                                                         clean_small_lfm2=clean_small_lfm2, recipe_key=recipe_key),
            "eff_batch": field_eff_batch(params_b), "strategy": "full"}


# D7 SWITCH (release defect D7, 2026-09-29). The "llama" full-FT default recipe keeps fp32 master + fused AdamW up to
# FP32_MASTER_MAX_B (3.3 B, so Llama-3.2-3B, open_llama_3b and WizardVicuna-3b train fp32 master: ~51 GB static). fp32
# master at 3 B is UNMEASURED; the decision comes from the phase-am am-llama32-3bi step time (fp32 master) against a
# bf16 + paged 8-bit arm. None = OFF = the b676020 behaviour. A float X = llama recipe tasks with params_b >= X train
# bf16 weights + paged 8-bit AdamW (the generic recipe uses 2.0). FORGE_V2_LLAMA_BF16_MIN_B overrides for research.
LLAMA_RECIPE_BF16_MIN_B: float | None = None


def llama_recipe_bf16_min_b() -> float | None:
    raw = os.environ.get("FORGE_V2_LLAMA_BF16_MIN_B", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            return LLAMA_RECIPE_BF16_MIN_B
    return LLAMA_RECIPE_BF16_MIN_B


def llama_recipe_precision(recipe: dict[str, Any], params_b: float) -> dict[str, Any]:
    """Apply the D7 switch to a "llama" recipe dict (unchanged when the switch is off)."""
    min_b = llama_recipe_bf16_min_b()
    if min_b is None or recipe.get("bf16") or float(params_b) < float(min_b):
        return recipe
    return {**recipe, "bf16": True, "bf16_min_b": float(min_b)}


def large_model_rule_eligible(*, field: bool, recipe_key: str | None, cheap_full: bool, g2c: bool, strategy: str | None,
                              params_b: float) -> bool:
    """Static part of the >= 3.0 B large-model batch rule gate in run (release-candidate-v9): the full-FT default FIELD
    route (a recipe_key), rung 1 (not cheap_full), not the damaged Gemma2 champion port, full strategy, and
    large_model_rule.applies(params_b) (params_b >= 3.0 and FORGE_V2_LARGE_MODEL_RULE unset or "1"). run additionally
    needs a measured step time, a fixed FIELD LR (_lr_override > 0) and no FIELD LR calibration experiment; then
    large_model_rule.plan_growth decides from the window and data whether the batch grows (else
    sft_v2_large_model_batch_kept). Used by the route snapshot (tests/test_route_snapshot_v9.py)."""
    return bool(field and recipe_key and not cheap_full and not g2c and strategy == "full"
                and large_model_rule.applies(params_b))


def run_field_recipe(model_type: str | None, params_b: float, *, field: bool, recipe_key: str | None = None,
                     damaged_base_route: bool = False, clean_small_qwen_route: bool = False,
                     clean_small_lfm2_route: bool = False, base_damage: str | None = None,
                     base_clean_ref: str | None = None) -> dict[str, Any]:
    """The recipe dict sft_v2.run applies for its routing kwargs (precision, LR multiplier, bucket). Extracted verbatim
    from run (release-candidate-v8, no behaviour change) so the route snapshot test (tests/test_route_snapshot_v8.py) can
    compare what every dispatched cell would train with against 7024722 without a GPU. ``base_damage`` /
    ``base_clean_ref`` (the dispatcher's damage verdict and the reference a "clean" verdict names) select
    hyperparameters only (clean_minicpm5_lr_rule)."""
    family_recipe = bool(field and recipe_key in ("family", "minicpm5"))  # full-FT default: the family's measured row
    generic_recipe = bool(field and recipe_key == "generic")
    damaged_recipe = bool(field and (damaged_base_route or clean_small_qwen_route or family_recipe))
    csl_field = bool(field and clean_small_lfm2_route and not damaged_recipe)
    recipe = field_recipe(model_type, params_b, damaged_base=damaged_recipe, clean_small_lfm2=csl_field,
                          generic=generic_recipe, recipe_key=recipe_key if field else None)
    if field and recipe_key == "llama":
        recipe = llama_recipe_precision(recipe, params_b)
    cm5_mult = clean_minicpm5_lr_rule(recipe_key if field else None, base_damage, base_clean_ref)[0]
    if cm5_mult is not None and not _explicit_field_lr_mult():
        recipe = {**recipe, "lr_mult": float(cm5_mult)}  # clean MiniCPM5-1B (v8, phase-bc; detector-conditional)
    return recipe


def field_bf16_micro_accum(*, unit_len: int, vocab: int, eff_default: int) -> tuple[int, int]:
    """(micro batch, grad accumulation) of the field bf16 + paged 8-bit AdamW geometry for a training unit of
    ``unit_len`` tokens (the pack length when packed) at ``eff_default`` units per optimizer step."""
    tok_budget = 49152 if vocab < 200_000 else 36864
    micro = max(1, min(64, tok_budget // max(1, unit_len), eff_default))
    return micro, max(1, math.ceil(eff_default / micro))


# Damaged Gemma2 champion recipe (RESEARCH CANDIDATE 2026-09-26, branch research-gemma-champion-port-20260926; NOT released).
# Evidence: replay-bench Phase U2 ran the public champion tree 251011a on the damaged gemma-2-2b ctxsplit replay (task
# 59469903, layer_reinit, 0.75 h, native TRAIN 26,980 rows, 1 x H100 PCIe, one run, seed not controllable): historical
# all-row 0.0474835 vs our damaged route 0.053679 / 0.053774 (Phase I aab85c9 s7 / s17; paired 99% CIs wholly below 0)
# and the winner 0.0451093. Executed recipe (phase-u2/results-extra/pu2-r-gemma2-ctxsplit-champ/EXECUTED-RECIPE.md, from
# the retained W&B config and trainer state): bf16 full FT + paged 8-bit AdamW, 32 FA-packed blocks of 3072 tokens per
# optimizer step (the 2_4_b bucket 48 x 2048 / pack length), searched peak LR 2.9757061485479407e-05, warmup 11, cosine to
# 0.25 x planned over 1.3 epochs (the planner floor) and stopped by the clock at 0.83 epoch, NEFTune alpha 1, wd 0, clip 1.
# This route ports the pieces our stack does not already share: row-isolated packing (sdpa + position-id masks here, not
# FA varlen), the block geometry, that fixed peak LR (not the search) and the 1.3-epoch planner floor. Everything else is
# the incumbent damaged gemma2 route (dev split, evals and best-snapshot selection, soup, dev pass, finish reserve).
# Scope = what was tested: the damaged-base FIELD route (DAMAGED_BASE_FAMILIES "gemma2" admitted it), model_type gemma2,
# vocabulary 256000, GEMMA2_CHAMPION_MIN_PARAMS_B < params_b <= GEMMA2_CHAMPION_MAX_PARAMS_B (the gemma-2-2b class,
# 2.614 B). Clean / unknown gemma2 (the adapter route), other families and other sizes are unchanged. If packing does not
# qualify on the image the whole recipe is skipped (incumbent route, event sft_v2_gemma2_champion_skipped).
# The validator runs the image with its own environment, so the production value is this constant;
# FORGE_V2_GEMMA2_CHAMPION_RECIPE=0/1 overrides it for research replays (0 restores the incumbent route exactly).
GEMMA2_CHAMPION_RECIPE_DEFAULT = "1"
GEMMA2_CHAMPION_MIN_PARAMS_B = 2.0  # exclusive
GEMMA2_CHAMPION_MAX_PARAMS_B = 3.0  # inclusive
GEMMA2_CHAMPION_VOCAB = 256000
GEMMA2_CHAMPION_PEAK_LR = 2.9757061485479407e-05  # the champion's search result on 59469903 (W&B learning_rate)
GEMMA2_CHAMPION_MIN_EPOCHS = 1.3  # train_instruct.py planner: optimal_epochs = max(1.3, min(3, 1.25 * achievable / spe))
GEMMA2_CHAMPION_BUCKET_BLOCKS = 48  # instruct_config 2_4_b batch_size
GEMMA2_CHAMPION_BUCKET_LEN = 2048  # blocks shrink by BUCKET_LEN / pack_len above it (instruct_config default_max)
GEMMA2_CHAMPION_MICRO_TOKENS = 36864  # micro-batch token budget (= field_bf16_micro_accum at vocab >= 200k)


def _gemma2_champion_recipe() -> bool:
    return os.environ.get("FORGE_V2_GEMMA2_CHAMPION_RECIPE", GEMMA2_CHAMPION_RECIPE_DEFAULT).strip() == "1"


# Damaged Gemma2 pack-speed candidate (research-gemma-pack-speed-20260927, research/20260927-gemma-pack-speed/DESIGN.md):
# phase-w measured the ported route at 11.4 s per 32-block step on H100 PCIe against the public champion's 6.0 s (same
# physical GPU). On the champion route only, packed training batches take varlen attention (forge.data.packing: PyTorch's
# FA2 kernel per row from the reset position ids, no 4-D mask, row isolation and sdpa semantics unchanged, softcap still
# off like the evaluator) and the step runs as ONE micro-batch of all its blocks (32 x 1 instead of 8 x 4), as the
# champion did; the memory probe still shrinks it. The varlen kernel is verified against a per-row reference on the GPU
# before use; if that fails, the route stays exactly on c292732's sdpa 8 x 4. FORGE_V2_GEMMA2_PACK_SPEED=0 restores c292732
# exactly; FORGE_V2_GEMMA2_VARLEN=0/1 and FORGE_V2_GEMMA2_ONE_MICRO=0/1 split the two parts for component timing arms.
GEMMA2_PACK_SPEED_DEFAULT = "1"
GEMMA2_ONE_MICRO_MAX_TOKENS = 131072  # one micro-batch up to 41 blocks of 3136 (the champion ran 32 x 3072 = 98,304)


def _gemma2_pack_speed() -> bool:
    return os.environ.get("FORGE_V2_GEMMA2_PACK_SPEED", GEMMA2_PACK_SPEED_DEFAULT).strip() == "1"


def _gemma2_pack_speed_part(name: str) -> bool | None:
    """Explicit research setting of one part (FORGE_V2_GEMMA2_VARLEN / FORGE_V2_GEMMA2_ONE_MICRO), else None."""
    raw = os.environ.get(name, "").strip()
    return None if raw not in ("0", "1") else raw == "1"


def gemma2_one_micro(blocks_per_step: int, pack_len: int) -> tuple[int, int]:
    """(micro, accumulation) running ``blocks_per_step`` blocks as one micro-batch within GEMMA2_ONE_MICRO_MAX_TOKENS."""
    micro = max(1, min(int(blocks_per_step), GEMMA2_ONE_MICRO_MAX_TOKENS // max(1, int(pack_len))))
    return micro, max(1, math.ceil(int(blocks_per_step) / micro))


def gemma2_champion_admits(model_type: str | None, params_b: float, vocab_size: int | None, *,
                           damaged_base_route: bool) -> bool:
    """Damaged Gemma2 champion recipe admission (see GEMMA2_CHAMPION_RECIPE_DEFAULT): a run the dispatcher already put on
    the damaged-base FIELD route, gemma2, vocabulary 256000, the tested size class."""
    if not (damaged_base_route and _gemma2_champion_recipe()):
        return False
    if str(model_type or "").lower() != "gemma2" or vocab_size is None or int(vocab_size) != GEMMA2_CHAMPION_VOCAB:
        return False
    return GEMMA2_CHAMPION_MIN_PARAMS_B < params_b <= GEMMA2_CHAMPION_MAX_PARAMS_B


def gemma2_champion_blocks(pack_len: int) -> int:
    """Packed blocks per optimizer step: the champion's 2_4_b bucket (48) scaled by 2048 / pack length when the pack
    length exceeds 2048 (instruct_config.get_training_json): 3072 -> 32, as executed on 59469903."""
    if pack_len <= GEMMA2_CHAMPION_BUCKET_LEN:
        return GEMMA2_CHAMPION_BUCKET_BLOCKS
    return max(1, int(GEMMA2_CHAMPION_BUCKET_BLOCKS * GEMMA2_CHAMPION_BUCKET_LEN / pack_len))


def gemma2_champion_micro_accum(pack_len: int, blocks: int) -> tuple[int, int]:
    """(micro, accumulation) for ``blocks`` packed blocks per step within the micro token budget: the product closest to
    ``blocks`` (ties: at least ``blocks``, then fewer accumulation steps). 32 x 3072 -> 8 x 4."""
    cap = max(1, min(64, GEMMA2_CHAMPION_MICRO_TOKENS // max(1, pack_len), blocks))
    first = max(1, math.ceil(blocks / cap))
    best: tuple[tuple[int, int, int], int, int] | None = None
    for accum in range(first, first + 3):
        for micro in {blocks // accum, math.ceil(blocks / accum)}:
            if 1 <= micro <= cap:
                key = (abs(micro * accum - blocks), 0 if micro * accum >= blocks else 1, accum)
                if best is None or key < best[0]:
                    best = (key, micro, accum)
    assert best is not None
    return best[1], best[2]


def _field_lr_calibration() -> bool:
    """CPU-reviewed research integration; never enabled by default."""
    return os.environ.get("FORGE_V2_FIELD_LR_CALIBRATION", "0") == "1"


def _field_sweep() -> bool:
    """Run the LR search on the field route (centred on the raw analytic law). The champion
    stack always searches; our fixed law lost on every 2–4B family at the floored value."""
    return os.environ.get("FORGE_V2_FIELD_SWEEP", "0") == "1"


def v2_takes_lfm25() -> bool:
    forced = os.environ.get("FORGE_V2_TAKES_LFM25", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    return V2_TAKES_LFM25
MAX_PARAMS_B = 5.0  # hard ceiling of the handler (memory / throughput)
# Validated regime (2026-09-10 harness, official evaluator, paired rule): full-weight
# v2 beat production LoRA on SmolLM2-360M (-2.6%, win rate 0.76) and lost on every
# 2-3B model tried. Larger models stay on the production adapter path unless the
# operator widens the gate with FORGE_V2_MAX_PARAMS_B.
DEFAULT_MAX_PARAMS_B = 0.6
# Adapter (LoRA) strategy inside v2 for larger models, validated 2026-09-10 on
# Gemma-2-2B clean (-3.3%), pruned (-6.4%) and noise-damaged (-2.6%) bases against
# production at effective batch 16; LFM/Qwen confirmations pending at commit time.
DEFAULT_LORA_MAX_PARAMS_B = 5.0
# LFM2.5-2.6B has a production one-epoch-cap route (2nd live on Sept 7); v2 takes it only
# once the batch-16 adapter arm beats that route on the harness.
V2_TAKES_LFM25 = False


MULTI_GPU_LORA_MAX_PARAMS_B = 40.0  # device_map-sharded adapter route (Round-2 Qwen3-32B on 4xH100)
MULTI_GPU_LORA_MIN_PARAMS_B = 12.0  # only the 4xH100 class is validated; 4–12B on 2 GPUs keeps the production path


def _multi_gpu_lora() -> bool:
    """Adapter route for large models sharded across several GPUs (naive model
    parallel via device_map). On by default since 2026-09-12: on Qwen3-32B / 4xH100
    the v2 handler (no fused kernels, micro 8, one annealed cycle at 80% of the
    window, periodic + final evals, soup, dev pass) scored 0.464868 against the
    production path's 0.507592 (paired win 0.676). FORGE_V2_MULTI_GPU_LORA=0 disables."""
    return os.environ.get("FORGE_V2_MULTI_GPU_LORA", "1") != "0"


def _gates() -> tuple[float, float]:
    try:
        full_b = float(os.environ.get("FORGE_V2_MAX_PARAMS_B", str(DEFAULT_MAX_PARAMS_B)))
    except ValueError:
        full_b = DEFAULT_MAX_PARAMS_B
    lora_cap = MULTI_GPU_LORA_MAX_PARAMS_B if _multi_gpu_lora() else MAX_PARAMS_B
    try:
        lora_b = float(os.environ.get("FORGE_V2_LORA_MAX_PARAMS_B", str(lora_cap)))
    except ValueError:
        lora_b = lora_cap
    return min(full_b, MAX_PARAMS_B), min(lora_b, lora_cap)


# The adapter path beats production only where production is throughput-bound: long
# rows (gemma2-cas, median 762 tokens: 3/3 wins). On short-row tasks (median ~60–100
# tokens: alpaca, indic-dolly) production's continuous schedule is as good or better.
LORA_MIN_MEDIAN_TOKENS = 256


def long_row_task(rows: list, spec: TaskSpec, tokenizer: Any, *, sample: int = 512, threshold: int | None = None) -> tuple[bool, int]:
    """(median tokenized length of a row sample >= threshold, median)."""
    import random

    thr = threshold if threshold is not None else int(os.environ.get("FORGE_V2_LORA_MIN_MEDIAN", str(LORA_MIN_MEDIAN_TOKENS)))
    if spec.instruct is None or not rows:
        return False, 0
    picked = rows if len(rows) <= sample else random.Random(3).sample(list(rows), sample)
    try:
        if spec.instruct.output is None:
            docs = prompts.build_completion_documents(picked, spec.instruct)
            toks = tokenize.tokenize_completion(docs, tokenizer, EVAL_CAP)
        else:
            examples = prompts.build_instruct_examples(picked, spec.instruct)
            toks = tokenize.tokenize_instruct(examples, tokenizer, EVAL_CAP)
    except Exception:
        return False, 0
    lengths = sorted(len(ex["input_ids"]) for ex in toks)
    if not lengths:
        return False, 0
    med = lengths[len(lengths) // 2]
    return med >= thr, med


# Model families that take full weights up to MAX_PARAMS_B regardless of the size gate
# (harness-validated per family; empty until a family beats production).
FULL_FT_MODEL_TYPES: frozenset[str] = frozenset()


def _full_ft_types() -> frozenset[str]:
    extra = {t.strip().lower() for t in os.environ.get("FORGE_V2_FULL_TYPES", "").split(",") if t.strip()}
    return FULL_FT_MODEL_TYPES | frozenset(extra)


LEARNABILITY_RATIO = 3.0      # untrained loss / loss after a 60-step adapter probe
LEARNABILITY_MIN_EPOCHS = 1.0  # projected full-weight epochs that must fit the window


def learnability_probe(model: Any, tokenizer: Any, rows: list, spec: TaskSpec, deadline: Deadline, params_b: float,
                       *, steps: int = 60, n_dev: int = 64) -> dict[str, Any]:
    """Decide full weights vs adapter for a 0.6–5B task from the task itself.

    A short adapter probe (steps x batch 16 at 2e-4) measures how far the loss
    falls from the untrained model on held-out rows and how fast the adapter
    trains. Structured tasks (QA generation, segmentation, formatting) fall by
    3x or more within 60 steps and the field wins them with full weights;
    open-ended instruct data falls by ~1.5x and adapters generalise better. Full
    weights are only chosen when they can also afford >= 1 projected epoch at
    about half the adapter's throughput. The temporary adapter is unloaded, so
    the base weights are untouched."""
    import torch
    from forge.model import attach_lora

    out: dict[str, Any] = {"decision": "lora"}
    t0 = time.monotonic()
    assert spec.instruct is not None
    if spec.instruct.output is None:
        docs = prompts.build_completion_documents(rows, spec.instruct)
        tokenized = tokenize.tokenize_completion(docs, tokenizer, EVAL_CAP)
    else:
        examples = prompts.build_instruct_examples(rows, spec.instruct)
        tokenized = tokenize.tokenize_instruct(examples, tokenizer, EVAL_CAP)
    tokenized = exact_dedup(tokenized)
    if len(tokenized) < 200:
        out["reason"] = "too_few_rows"
        return out
    lengths = [len(ex["input_ids"]) for ex in tokenized]
    max_len = adaptive_max_len(lengths, ceiling=EVAL_CAP)
    train_ex, dev_ex = stratified_split(tokenized, dev_size_for(len(tokenized), hours=deadline.remaining_hard() / 3600.0), seed=7)
    train_ex = [smart_truncate(ex, max_len) for ex in train_ex]
    dev_probe = [smart_truncate(ex, max_len) for ex in dev_ex[:n_dev]]
    collator = tokenize.PadCollator(tokenizer.pad_token_id)
    micro = max(1, min(16, 24576 // max(1, max_len)))
    accum = max(1, math.ceil(16 / micro))
    import random as _random

    rng = _random.Random(5)
    picks = list(range(len(train_ex)))
    rng.shuffle(picks)
    need = steps * accum
    batches = [collator([train_ex[i] for i in picks[j : j + micro]]) for j in range(0, min(len(picks), need * micro), micro)][:need]
    model.config.use_cache = False
    peft_model = attach_lora(model, r=32, alpha=64, dropout=0.05)
    gc_on = params_b >= 0.8 and hasattr(peft_model, "gradient_checkpointing_enable")
    if gc_on:
        peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cuda = torch.cuda.is_available()
    try:
        untrained, _ = selection.per_example_dev_loss(peft_model, dev_probe, collator, batch_size=1, autocast_bf16=cuda)
        timing: dict[str, Any] = {"secs": 0.0, "steps": 0}
        lr_probe.run_trial(
            peft_model, batches, lr=2.0e-4, opt_steps=steps, grad_accum=accum,
            optimizer_factory=lambda ps, lr: torch.optim.AdamW(ps, lr=lr, weight_decay=0.0),
            autocast_bf16=cuda, warmup_steps=6, timing=timing,
        )
        probe_loss, _ = selection.per_example_dev_loss(peft_model, dev_probe, collator, batch_size=1, autocast_bf16=cuda)
    finally:
        if gc_on and hasattr(peft_model, "gradient_checkpointing_disable"):
            peft_model.gradient_checkpointing_disable()
        base = peft_model.unload() if hasattr(peft_model, "unload") else peft_model
        for prm in base.parameters():
            prm.requires_grad_(False)
        _free_cuda()
    ratio = (untrained / probe_loss) if (probe_loss and math.isfinite(probe_loss) and probe_loss > 0) else 0.0
    secs = float(timing.get("secs") or 0.0)
    nsteps = int(timing.get("steps") or 0)
    adapter_sps = (16.0 * nsteps / secs) if (secs > 0 and nsteps > 0) else 0.0
    window = max(0.0, deadline.remaining_hard() - 600.0)
    projected_full_epochs = (window * 0.85 * adapter_sps / 2.0 / max(1, len(train_ex))) if adapter_sps else 0.0
    decision = "full" if (ratio >= LEARNABILITY_RATIO and projected_full_epochs >= LEARNABILITY_MIN_EPOCHS and params_b <= MAX_PARAMS_B) else "lora"
    out.update(decision=decision, untrained=round(untrained, 4), probe_loss=round(probe_loss, 4), ratio=round(ratio, 3),
               adapter_samples_per_s=round(adapter_sps, 2), projected_full_epochs=round(projected_full_epochs, 3),
               median_len=sorted(lengths)[len(lengths) // 2], probe_s=round(time.monotonic() - t0, 1))
    _event_and_print("sft_v2_learnability", **out)
    return out


def strategy_for(params_b: float, model_type: str | None = None) -> str:
    """'full' inside the full-weight gate (or for a validated family), 'lora'
    inside the adapter gate, else ''. FORGE_V2_STRATEGY=full|lora forces one."""
    forced = os.environ.get("FORGE_V2_STRATEGY", "auto").strip().lower()
    if forced in ("full", "lora"):
        return forced
    full_b, lora_b = _gates()
    if model_type and str(model_type).lower() in _full_ft_types() and 0 < params_b <= MAX_PARAMS_B:
        return "full"
    if 0 < params_b <= full_b:
        return "full"
    if 0 < params_b <= lora_b:
        return "lora"
    return ""
FP32_MASTER_MAX_B = 3.3
EXPORT_RESERVE_S = 150.0

# Large-data full-weight route (2026-09-11, Qwen2.5-3B-Instruct / ru-AAQG at the Aug 31
# task's scale: 52k rows, 1.5 h): full FT at the champion's per-row LR, 1e-5 for 64
# rows per update, one annealed epoch + soup + dev pass, scored 0.523538 where the
# same recipe on 20k rows / 1 h ties the adapter. Off until the matched adapter
# comparison (r59-lora) says the route wins by the confirmation margin.
BIGDATA_FULL_MIN_ROWS = 40000
BIGDATA_FULL_MIN_HOURS = 1.2
BIGDATA_FULL_MODEL_TYPES: frozenset[str] = frozenset({"qwen2", "qwen3"})
BIGDATA_FULL_LR = 1.0e-5


def bigdata_full_route(*, n_rows: int, hours: float, model_type: str | None, params_b: float) -> bool:
    """True when a long-row adapter task should train full weights instead:
    validated model family, fp32-master-sized model, >= BIGDATA_FULL_MIN_ROWS rows
    and >= BIGDATA_FULL_MIN_HOURS of budget. FORGE_V2_BIGDATA_FULL=1 enables it."""
    if os.environ.get("FORGE_V2_BIGDATA_FULL", "0") != "1":
        return False
    try:
        min_rows = int(os.environ.get("FORGE_V2_BIGDATA_MIN_ROWS", str(BIGDATA_FULL_MIN_ROWS)))
        min_hours = float(os.environ.get("FORGE_V2_BIGDATA_MIN_HOURS", str(BIGDATA_FULL_MIN_HOURS)))
    except ValueError:
        min_rows, min_hours = BIGDATA_FULL_MIN_ROWS, BIGDATA_FULL_MIN_HOURS
    types = BIGDATA_FULL_MODEL_TYPES | {t.strip().lower() for t in os.environ.get("FORGE_V2_BIGDATA_TYPES", "").split(",") if t.strip()}
    return (
        str(model_type or "").lower() in types
        and 0 < params_b <= FP32_MASTER_MAX_B
        and n_rows >= min_rows
        and hours >= min_hours
    )
MIN_TASK_SECONDS_FOR_V2 = 20 * 60
TIMING_PROBE_MIN_REMAINING_S = 600.0  # D4: fixed-LR full-FT default runs (incl. late rung 2) still measure step time


def _log(name: str, payload: dict[str, Any] | None = None, **kw: Any) -> None:
    _event_and_print(name, **(payload or {}), **kw)


_orig_event = telemetry.event


def _event_and_print(name: str, **kv: Any) -> None:
    """Record the telemetry event and mirror v2 events to stderr so container
    logs show the recipe's decisions (the JSON flight recorder is only visible
    after the artifact is copied out)."""
    _orig_event(name, **kv)
    if name.startswith(("sft_v2", "lr_probe", "soup", "liger")):
        try:
            import sys

            compact = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in kv.items()}
            print(f"[sft_v2] {name} {compact}", file=sys.stderr, flush=True)
        except Exception:
            pass


def eligible(spec: TaskSpec, *, is_kl: bool, params_b: float, n_gpus: int, model: Any) -> bool:
    if os.environ.get("FORGE_SFT_V2", "1") == "0":
        return False
    if spec.task_type != "InstructTextTask" or spec.instruct is None or is_kl:
        return False
    allow_cpu = os.environ.get("FORGE_SFT_V2_ALLOW_CPU") == "1"
    model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "")
    strategy = strategy_for(params_b, model_type)
    cap = MULTI_GPU_LORA_MAX_PARAMS_B if (_multi_gpu_lora() and strategy == "lora" and n_gpus > 1) else MAX_PARAMS_B
    if params_b <= 0 or params_b > cap or not strategy:
        telemetry.event("sft_v2_size_gated", params_b=round(params_b, 3), gates=_gates())
        return False
    multi_ok = _multi_gpu_lora() and n_gpus > 1 and strategy == "lora" and params_b >= MULTI_GPU_LORA_MIN_PARAMS_B
    if n_gpus != 1 and not (allow_cpu and n_gpus == 0) and not multi_ok:
        return False
    try:
        import torch

        if not torch.cuda.is_available() and not allow_cpu:
            return False
    except Exception:
        return False
    from forge.model import is_quasar_model

    if is_quasar_model(model):
        return False
    return True


@dataclass
class Geometry:
    micro_batch: int
    grad_accum: int
    max_len: int
    gradient_checkpointing: bool
    fp32_master: bool
    optim: str

    @property
    def eff_batch(self) -> int:
        return self.micro_batch * self.grad_accum


def _vocab(model: Any) -> int:
    return int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)


def estimate_train_gb(*, params_b: float, tokens: int, layers: int, hidden: int, vocab: int, fp32_master: bool,
                      gradient_checkpointing: bool, liger: bool) -> float:
    """Rough peak-memory estimate (GB) for one micro-batch of `tokens` tokens."""
    weights = params_b * (16.0 if fp32_master else 8.0)  # weights+grads+Adam (8-bit Adam ~ bf16 sizes)
    per_token_layer = 34.0 * hidden * 2.0  # bf16 activations kept for backward (no checkpointing)
    if gradient_checkpointing:
        per_token_layer = 4.0 * hidden * 2.0 + per_token_layer / max(1, layers) * 2
    acts = tokens * layers * per_token_layer / 1e9
    logits = 0.0 if liger else tokens * vocab * 6.0 / 1e9
    return weights + acts + logits + 4.0


def choose_geometry(*, params_b: float, max_len: int, vocab: int, per_gpu_gb: float, bnb_ok: bool,
                    layers: int = 32, hidden: int = 2048, liger: bool = False, eff_default: int = EFF_BATCH_TARGET) -> Geometry:
    """Geometry from measured behaviour (H100, Gemma-2-2B, 2026-09-10 benchmark):
    checkpointing with a large micro-batch (8 x 3136 tokens) ran at 5.7 samples/s
    while no-checkpointing at micro 2 crawled at 1.9 samples/s from memory
    pressure. So: models >= 0.8B train with checkpointing and a token budget per
    micro-batch of ~24k (fp32 master) / ~48k (bf16 + 8-bit Adam); tiny models skip
    checkpointing. The memory-shape probe below still shrinks anything that does
    not fit."""
    fp32_master = params_b <= FP32_MASTER_MAX_B
    if os.environ.get("FORGE_V2_FP32_MASTER", "").strip() in ("0", "1"):
        fp32_master = os.environ["FORGE_V2_FP32_MASTER"].strip() == "1"  # ablation knob: bf16 weights + 8-bit AdamW when 0
    gc_on = params_b >= 0.8
    if gc_on:
        tok_budget = 24576 if fp32_master else 49152
        if vocab >= 200_000 and not liger:
            tok_budget //= 3
        elif vocab >= 200_000:
            tok_budget = int(tok_budget * 0.75)
    else:
        tok_budget = 16384 if vocab < 100_000 else 8192
    if params_b > 4.0:
        tok_budget = int(tok_budget * 0.75)
    micro = max(1, min(64, tok_budget // max_len))
    try:
        eff_target = int(os.environ.get("FORGE_V2_EFF_BATCH", str(eff_default)))
    except ValueError:
        eff_target = eff_default
    if os.environ.get("FORGE_V2_GC") == "1":
        gc_on = True
    elif os.environ.get("FORGE_V2_GC") == "0":
        gc_on = False
    micro = max(1, min(micro, max(1, eff_target)))  # short rows: the micro-batch alone exceeded the target
    accum = max(1, math.ceil(max(1, eff_target) / micro))
    if fp32_master:
        optim = "adamw_torch_fused"
    else:
        optim = "paged_adamw_8bit" if bnb_ok else "adamw_torch"
    return Geometry(micro, accum, max_len, gc_on, fp32_master, optim)


def optimizer_state_bytes(model: Any, optim: str) -> int:
    """Bytes the optimizer will allocate lazily at its first step (not visible
    to a forward/backward probe): two fp32 moments per trainable parameter for
    torch AdamW, two 8-bit moments for bitsandbytes' 8-bit AdamW."""
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    per = 2 if "8bit" in optim else 8
    return int(n * per)


def memory_probe(model: Any, *, micro: int, max_len: int, vocab: int, autocast: bool, reserve_bytes: int = 0,
                 headroom: float = 0.07, packed_mode: str = "off") -> bool:
    """One synthetic forward/backward at the worst-case micro-batch shape. Fits
    only if the peak plus `reserve_bytes` (optimizer state) leaves `headroom`
    (7% by default; fp32-master field-route weights use 15%: LFM2.5-2.6B passed at
    micro 13 with 7% and OOM'd on the first real optimizer step, 2026-09-15)."""
    import torch

    device = next(model.parameters()).device
    ids = torch.randint(5, max(6, vocab - 1), (micro, max_len), device=device)
    batch = (packing.probe_batch(ids, packed_mode) if packed_mode != "off" else
             {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()})
    model.train()
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
        else:
            loss = model(**batch).loss
        loss.backward()
        if device.type == "cuda" and reserve_bytes > 0:
            peak = torch.cuda.max_memory_allocated(device)
            total = torch.cuda.get_device_properties(device).total_memory
            if peak + reserve_bytes > (1.0 - headroom) * total:
                _event_and_print("sft_v2_memory_probe_reserve", micro=micro, peak_gb=round(peak / 1e9, 1),
                                 reserve_gb=round(reserve_bytes / 1e9, 1), total_gb=round(total / 1e9, 1))
                return False
        return True
    except Exception as exc:  # noqa: BLE001
        if _is_oom(exc):
            return False
        _event_and_print("sft_v2_memory_probe_error", micro=micro, error=f"{type(exc).__name__}: {exc}"[:200])
        return False
        raise
    finally:
        model.zero_grad(set_to_none=True)
        del batch
        _free_cuda()


def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def _liger_ok(model: Any) -> bool:
    if os.environ.get("FORGE_LIGER", "1") == "0":
        return False
    try:
        import liger_kernel  # noqa: F401
    except Exception:
        return False
    mt = str(getattr(getattr(model, "config", None), "model_type", "") or "")
    return mt in {"llama", "gemma", "gemma2", "gemma3", "gemma3_text", "qwen2", "qwen3", "mistral", "mixtral", "phi3", "smollm3", "olmo2", "granite"}


class _StopAt:
    """Stops training when the remaining wall clock drops below the finish reserve."""

    def __init__(self, deadline: Deadline, finish_reserve_s: float) -> None:
        self.deadline = deadline
        self.finish_reserve_s = finish_reserve_s
        self.fired = False

    def should_stop(self) -> bool:
        return self.deadline.remaining_hard() <= self.finish_reserve_s


def run(
    spec: TaskSpec,
    deadline: Deadline,
    rows: list,
    loaded: Any,
    tokenizer: Any,
    *,
    baseline_summary: Any,
    params_b: float,
    per_gpu_gb: float,
    strategy_override: str | None = None,
    route_lr: float | None = None,
    field_route: bool = False,
    damaged_base_route: bool = False,
    clean_small_qwen_route: bool = False,
    clean_small_lfm2_route: bool = False,
    recipe_key: str | None = None,
    cheap_full: bool = False,
    base_damage: str | None = None,
    base_clean_ref: str | None = None,
) -> None:
    """``recipe_key`` (default_recipe_key, full-FT default route): which measured recipe the FIELD route trains with.
    ``cheap_full`` (fallback rung 2): bf16 weights + paged 8-bit AdamW at any size, half the micro-batch, checkpointing
    on, no Gemma2 champion geometry. ``base_damage`` / ``base_clean_ref``: the dispatcher's damage verdict and the
    reference a "clean" verdict names (hyperparameters only: clean_minicpm5_lr_rule). Raises ZeroTrainedStepsError when
    no optimizer step was taken."""
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from forge.tasks.fallback import emit_untrained_copy

    t_start = time.monotonic()
    model = loaded.model
    loaded.model = None
    _event_and_print("sft_v2_selected", params_b=round(params_b, 3), hours_remaining=round(deadline.remaining_hard() / 3600, 3))

    # ---- floor first: a valid untrained adapter at the output path ----
    emit_untrained_copy(spec)
    write_artifact_truth(spec.output_dir, ARTIFACT_FLOOR, optimizer_step=0, reason="pretraining_floor")

    # ---- data ----
    assert spec.instruct is not None
    if spec.instruct.output is None:
        docs = prompts.build_completion_documents(rows, spec.instruct)
        tokenized = tokenize.tokenize_completion(docs, tokenizer, EVAL_CAP)
    else:
        examples = prompts.build_instruct_examples(rows, spec.instruct)
        tokenized = tokenize.tokenize_instruct(examples, tokenizer, EVAL_CAP)
    n_raw = len(tokenized)
    # ---- PLACEHOLDER (release-candidate-v9; NOT IMPLEMENTED): phase-bg exposure arm slot (1 of 2) ----
    # Nothing here runs; the exact dedup below is 854bf7a's. The slot waits on replay-bench phase-bg
    # (research/20260923-replay-bench/phase-bg: LFM2.5-1.2B arms B keep exact duplicates, C 4-epoch cap + no DEV overfit
    # stop + no epoch-cap extension, D = B + C; research knobs on research-bg-exposure-20260930 @ 32984fb). If an arm is
    # chosen on DEV and passes the directive-8 suite check, port the knob here (keep exact duplicates) as its own commit,
    # scoped by an observable condition (not a model ID) and with its own route-snapshot declaration and disclosure.
    tokenized = exact_dedup(tokenized)
    if not tokenized:
        raise RuntimeError("no trainable examples after tokenization")
    strategy = strategy_override or strategy_for(params_b, str(getattr(getattr(model, "config", None), "model_type", "") or "")) or "full"
    _dm = getattr(model, "hf_device_map", None) or {}
    sharded = len({str(v) for v in _dm.values()}) > 1
    # the field geometry applies only when the dispatcher chose the field route (family, rows,
    # size gates); the legacy v2 full path (tiny models by size) keeps its previous behaviour
    field = field_route and strategy == "full" and not sharded
    model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "").lower()
    # This experiment is valid only for Gemma2, non-KL, non-FIELD, single-GPU execution.
    _small_batch_requested = os.environ.get("FORGE_V2_GEMMA_ADAPTER_SMALL_BATCH", "1") != "0"
    # The deployment integration sets this only after the reviewed identity
    # gate and default-geometry check below. Unknown or altered bases keep the
    # incumbent geometry instead of turning an opt-in flag into task failure.
    gemma_adapter_small_batch = False
    if _small_batch_requested and strategy == "lora" and model_type == "gemma2":
        _event_and_print("sft_v2_small_batch_route", strategy=strategy, model_type=model_type,
                         field=bool(field), sharded=bool(sharded), cuda_available=bool(torch.cuda.is_available()),
                         cuda_devices=int(torch.cuda.device_count()), use_kl=bool(getattr(spec, "use_kl", False)))
    # ---- field hygiene (FORGE_V2_FIELD_HYGIENE=1, field route only): near-duplicate groups (removed when the
    # validator reports a duplicated corpus), group-aware dev split, prompt-loss weight; see forge.data.hygiene ----
    field_hygiene = field and hygiene.enabled()
    hyg: dict[str, Any] = {"dedup_removed": 0, "groups": 0, "grouped_rows": 0, "dedup": False, "ratio": None, "plw": 0.0}
    dup_groups: list[list[int]] | None = None
    if field_hygiene:
        try:
            dup_groups = hygiene.near_duplicate_groups(hygiene.render_texts(tokenized, tokenizer))
            hyg["groups"] = sum(1 for g in dup_groups if len(g) > 1)
            hyg["grouped_rows"] = sum(len(g) for g in dup_groups if len(g) > 1)
            hyg["dedup"] = hygiene.dedup_triggered(baseline_summary, forced=hygiene.dedup_forced())
            if hyg["dedup"]:
                before = len(tokenized)
                tokenized, dup_groups = hygiene.drop_near_duplicates(tokenized, dup_groups)
                hyg["dedup_removed"] = before - len(tokenized)
            hyg["ratio"] = hygiene.prompt_completion_ratio(tokenized)
            hyg["plw"] = hygiene.prompt_loss_weight(hyg["ratio"])
        except Exception as exc:  # an opt-in experiment must never floor the task: keep the plain data path
            _event_and_print("sft_v2_hygiene_failed", stage="data", error=f"{type(exc).__name__}: {exc}"[:200])
            dup_groups = None
            hyg["plw"] = 0.0
    lengths = [len(ex["input_ids"]) for ex in tokenized]
    model_cap = int(getattr(model.config, "max_position_embeddings", EVAL_CAP) or EVAL_CAP)
    max_len = adaptive_max_len(lengths, ceiling=min(EVAL_CAP, max(256, model_cap)))
    hours_total = deadline.remaining_hard() / 3600.0
    dev_n = dev_size_for(len(tokenized), hours=hours_total)
    if dup_groups is not None:
        train_ex, dev_ex = grouped_stratified_split(tokenized, dev_n, dup_groups, seed=7)
    else:
        train_ex, dev_ex = stratified_split(tokenized, dev_n, seed=7)
    train_ex = [smart_truncate(ex, max_len) for ex in train_ex]
    dev_ex = [smart_truncate(ex, max_len) for ex in dev_ex]
    _event_and_print(
        "sft_v2_data", rows=len(rows), tokenized=n_raw, deduped=len(tokenized), train_n=len(train_ex), dev_n=len(dev_ex),
        max_len=max_len, p50=sorted(lengths)[len(lengths) // 2], p99=sorted(lengths)[min(len(lengths) - 1, int(0.99 * len(lengths)))],
    )

    # ---- model preparation (full weights) ----
    vocab = _vocab(model)
    cfg = getattr(model, "config", None)
    layers = int(getattr(cfg, "num_hidden_layers", 32) or 32)
    hidden = int(getattr(cfg, "hidden_size", 2048) or 2048)
    use_liger = _liger_ok(model)
    if sharded:
        # device_map model parallel (Round-2 32B on 4xH100): Triton fused kernels
        # raised on cross-device tensors during the probe (2026-09-12 dry-2v), so
        # the sharded route uses plain kernels and a capped micro-batch.
        use_liger = False
        _event_and_print("sft_v2_sharded", devices=len({str(v) for v in _dm.values()}), liger=False)
    if use_liger:
        try:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(model=model)
        except Exception as exc:  # fused kernels are an optimisation, never a requirement
            _event_and_print("liger_apply_failed", error=f"{type(exc).__name__}: {exc}")
            use_liger = False
    # damaged Gemma2 champion recipe (GEMMA2_CHAMPION_RECIPE_DEFAULT): row-isolated packing is admitted for this model
    # instance only (forge.data.packing.GEMMA2_ROUTE_ATTR); geometry, LR and planner floor are applied below
    _family_recipe = bool(field and recipe_key in ("family", "minicpm5"))  # full-FT default: the family's measured row
    _generic_recipe = bool(field and recipe_key == "generic")
    _g2c = bool(field and not cheap_full
                and gemma2_champion_admits(model_type, params_b, _vocab(model),
                                           damaged_base_route=damaged_base_route or _family_recipe))
    if _g2c:
        setattr(model, packing.GEMMA2_ROUTE_ATTR, True)
    if field:
        model.config.use_cache = False
    pack = packing.plan_field_packing(
        model, train_ex, max_len=max_len, pad_token_id=tokenizer.pad_token_id,
        model_cap=min(EVAL_CAP, model_cap), field_route=field,
        field_hygiene=field_hygiene, log=_event_and_print,
    )
    unit_len = pack.pack_len if pack else max_len
    # adapters train at production's proven effective batch 16 (LFM: 875 updates/epoch at
    # batch 16 beat 178 at batch 76); full weights follow the champion's batch 64, or
    # the champion's size bucket on the field route
    _eff_default = 16 if strategy == "lora" else (field_eff_batch(params_b) if field else EFF_BATCH_TARGET)
    if field:
        _event_and_print("sft_v2_field_route", params_b=round(params_b, 3), eff_batch=_eff_default)
    if pack:
        _eff_default = pack.eff_blocks(_eff_default)
    geo = choose_geometry(params_b=params_b, max_len=unit_len, vocab=vocab, per_gpu_gb=per_gpu_gb, bnb_ok=_bnb_available(),
                          layers=layers, hidden=hidden, liger=use_liger, eff_default=_eff_default)
    if _small_batch_requested and strategy == "lora" and model_type == "gemma2":
        try:
            from forge.routing.gemma_small_batch_admission import qualify as _qualify_small_batch

            _device = str(next(model.parameters()).device)
            _admission = _qualify_small_batch(
                model_dir=spec.cached_model_dir, model_type=model_type, strategy=strategy,
                field=field, is_kl=bool(getattr(spec, "use_kl", False)),
                cuda_available=bool(torch.cuda.is_available()), device=_device,
                n_gpus=int(torch.cuda.device_count()), sharded=sharded,
                micro_batch=geo.micro_batch, grad_accum=geo.grad_accum,
                effective_batch=geo.eff_batch,
            )
            if not isinstance(_admission, dict):
                raise TypeError("invalid small-batch admission result")
        except Exception as _gate_error:
            _admission = {"action": "incumbent", "status": "error",
                          "reason": f"admission_exception:{type(_gate_error).__name__}"}
        _event_and_print("sft_v2_small_batch_gate", action=_admission.get("action"),
                         status=_admission.get("status"), reason=_admission.get("reason"),
                         gate_reason=_admission.get("gate", {}).get("reason") if isinstance(_admission.get("gate"), dict) else None)
        gemma_adapter_small_batch = _admission.get("action") == "enable_gemma_small_batch"
    # damaged-base (and, when switched on, clean small-Qwen) field route: the family's executed recipe
    _damaged_recipe = bool(field and (damaged_base_route or clean_small_qwen_route or _family_recipe))
    _csq_field = bool(field and clean_small_qwen_route)  # clean small-Qwen FIELD run: window-aware eval governor on
    _csl_field = bool(field and clean_small_lfm2_route and not _damaged_recipe)  # clean LFM2.5-1.2B-Instruct: LR x2 only
    _recipe = run_field_recipe(str(getattr(getattr(model, "config", None), "model_type", "") or ""), params_b,
                               field=field, recipe_key=recipe_key, damaged_base_route=damaged_base_route,
                               clean_small_qwen_route=clean_small_qwen_route,
                               clean_small_lfm2_route=clean_small_lfm2_route, base_damage=base_damage,
                               base_clean_ref=base_clean_ref)
    if field:
        _event_and_print("sft_v2_field_recipe", damaged_base=bool(damaged_base_route),
                         clean_small_qwen=bool(clean_small_qwen_route), recipe_key=recipe_key, cheap_full=bool(cheap_full),
                         **_recipe)
        if (damaged_base_route and damaged_minicpm5_shape(model_type, params_b, vocab)) or recipe_key == "minicpm5":
            _event_and_print("sft_v2_damaged_minicpm5_route", params_b=round(params_b, 4), vocab=vocab)
        if _csl_field:
            _event_and_print("sft_v2_clean_small_lfm2_route", params_b=round(params_b, 4), vocab=vocab,
                             lr_mult=_recipe["lr_mult"])
    if field and geo.fp32_master and _recipe["bf16"] and os.environ.get("FORGE_V2_FP32_MASTER", "").strip() != "1":
        # 2–4B full weights in fp32 master are throughput-bound in a tournament window
        # (Gemma-2-2B 12.4 s/step → 0.49 epoch in 45 min); bf16 weights + 8-bit AdamW
        # learned identically to fp32 on the Llama ablation at 1.4x the speed
        micro, _accum = field_bf16_micro_accum(unit_len=unit_len, vocab=vocab, eff_default=_eff_default)
        geo = Geometry(micro, _accum, geo.max_len, geo.gradient_checkpointing, False,
                       "paged_adamw_8bit" if _bnb_available() else "adamw_torch")
        _event_and_print("sft_v2_field_bf16", params_b=round(params_b, 3), micro=geo.micro_batch, grad_accum=geo.grad_accum, optim=geo.optim)
    if cheap_full and strategy == "full":
        # fallback rung 2 (owner directive 4): the cheaper full fine-tune after the first attempt failed - bf16 weights +
        # paged 8-bit AdamW whatever the size or recipe, half the micro-batch (same effective batch), checkpointing on
        if geo.fp32_master or _recipe.get("bf16"):
            micro, _accum = field_bf16_micro_accum(unit_len=unit_len, vocab=vocab, eff_default=max(1, geo.eff_batch))
        else:
            micro = geo.micro_batch
        micro = max(1, micro // 2)
        geo = Geometry(micro, max(1, math.ceil(max(1, geo.eff_batch) / micro)), geo.max_len, True, False,
                       "paged_adamw_8bit" if _bnb_available() else "adamw_torch")
        _event_and_print("sft_v2_cheap_full", params_b=round(params_b, 3), micro=geo.micro_batch, grad_accum=geo.grad_accum,
                         optim=geo.optim)
    _g2c_geometry = None
    if _g2c and (pack is None or geo.fp32_master):
        # the recipe is packed bf16 + paged 8-bit AdamW; without either the incumbent route runs unchanged
        _event_and_print("sft_v2_gemma2_champion_skipped", reason="packing_unavailable" if pack is None else "fp32_master")
        _g2c = False
    if _g2c:
        # research geometry env keeps precedence (as on every FIELD route); otherwise the champion's blocks per step
        _g2c_env_geo = [k for k in (packing.ENV_EFF_BLOCKS, "FORGE_V2_EFF_BATCH", "FORGE_V2_FIELD_EFF_BATCH")
                        if os.environ.get(k, "").strip() not in ("", "0")]
        _g2c_blocks = gemma2_champion_blocks(pack.pack_len)
        if not _g2c_env_geo:
            micro, _accum = gemma2_champion_micro_accum(pack.pack_len, _g2c_blocks)
            geo = Geometry(micro, _accum, geo.max_len, geo.gradient_checkpointing, False, geo.optim)
            _eff_default = _g2c_blocks
        _g2c_geometry = {"pack_len": pack.pack_len, "blocks": pack.n_blocks,
                         "mean_rows_per_block": round(pack.mean_rows_per_block, 3), "target_blocks": _g2c_blocks,
                         "env_geometry": _g2c_env_geo, "planned_micro": geo.micro_batch, "planned_grad_accum": geo.grad_accum}
        if _gemma2_pack_speed():
            # pack-speed candidate (GEMMA2_PACK_SPEED_DEFAULT): varlen attention, then one micro-batch per step
            _g2v_want = _gemma2_pack_speed_part("FORGE_V2_GEMMA2_VARLEN")
            _g2v_backend = None
            if (_g2v_want is None or _g2v_want) and pack.mode == "sdpa":
                _g2v_backend = packing.enable_gemma2_varlen(model, pack_len=pack.pack_len, log=_event_and_print)
            if _g2v_backend is not None:
                pack.collator = packing.PackedPadCollator(tokenizer.pad_token_id, attention_mask=False, pad_as_row=True)
                pack.attn_implementation = f"{packing.VARLEN_ATTN}:{_g2v_backend}"
            _g2v_one = _gemma2_pack_speed_part("FORGE_V2_GEMMA2_ONE_MICRO")
            _g2v_one = (_g2v_backend is not None) if _g2v_one is None else _g2v_one
            if _g2v_one and not _g2c_env_geo:
                micro, _accum = gemma2_one_micro(geo.micro_batch * geo.grad_accum, pack.pack_len)
                geo = Geometry(micro, _accum, geo.max_len, geo.gradient_checkpointing, False, geo.optim)
            _event_and_print("sft_v2_gemma2_pack_speed", varlen_backend=_g2v_backend, one_micro=bool(_g2v_one and not _g2c_env_geo),
                             micro=geo.micro_batch, grad_accum=geo.grad_accum, blocks_per_step=geo.eff_batch,
                             planned_micro=_g2c_geometry["planned_micro"],
                             planned_grad_accum=_g2c_geometry["planned_grad_accum"])
    _small_batch_default_eff = None
    _small_batch_target_eff = None
    if strategy == "lora":
        from forge.model import attach_lora

        try:
            _r = int(os.environ.get("FORGE_V2_LORA_R", "32"))
            _alpha = int(os.environ.get("FORGE_V2_LORA_ALPHA", str(2 * _r)))
            _drop = float(os.environ.get("FORGE_V2_LORA_DROPOUT", "0.0"))
        except ValueError:
            _r, _alpha, _drop = 32, 64, 0.05
        _event_and_print("sft_v2_adapter", r=_r, alpha=_alpha, dropout=_drop)
        model = attach_lora(model, r=_r, alpha=_alpha, dropout=_drop)
        # Adapter training is light: no fp32 master copy, plain fused AdamW,
        # checkpointing only when the probe says the geometry does not fit.
        geo = Geometry(geo.micro_batch, geo.grad_accum, geo.max_len, False, False, "adamw_torch_fused")
        if sharded and geo.micro_batch > 8:
            # a 32B split over 4 cards keeps ~60 GB per card for activations:
            # micro 16 x 768 tokens without checkpointing does not fit, micro 8 does
            geo = Geometry(8, max(1, geo.eff_batch // 8), geo.max_len, False, False, "adamw_torch_fused")
        _small_batch_default_eff = geo.eff_batch
        if gemma_adapter_small_batch:
            if geo.grad_accum <= 1:
                raise RuntimeError("small-batch candidate requires default accumulation > 1")
            _small_batch_target_eff = geo.micro_batch * max(1, math.ceil(geo.grad_accum / 2))
            geo = Geometry(geo.micro_batch, max(1, math.ceil(geo.grad_accum / 2)), geo.max_len, geo.gradient_checkpointing, False, "adamw_torch_fused")
    else:
        if geo.fp32_master:
            model = model.float()
        for p in model.parameters():
            p.requires_grad_(True)
    model.config.use_cache = False
    autocast = geo.fp32_master and torch.cuda.is_available() and strategy == "full"
    if not torch.cuda.is_available():
        geo = Geometry(geo.micro_batch, geo.grad_accum, geo.max_len, False, geo.fp32_master, "adamw_torch")

    def _apply_gc(enabled: bool) -> None:
        if not hasattr(model, "gradient_checkpointing_enable"):
            return
        if enabled:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        elif hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    _apply_gc(geo.gradient_checkpointing)
    if torch.cuda.is_available():
        # Worst-case shape probe: shrink the micro-batch, then enable checkpointing, until it fits.
        ladder = [(geo.micro_batch, geo.gradient_checkpointing)]
        if not geo.gradient_checkpointing:
            ladder += [(geo.micro_batch, True)]
        ladder += [(max(1, geo.micro_batch // 2), True), (1, True)]
        seen = set()
        for micro_c, gc_c in ladder:
            if (micro_c, gc_c) in seen:
                continue
            seen.add((micro_c, gc_c))
            _apply_gc(gc_c)
            ok = memory_probe(model, micro=micro_c, max_len=geo.max_len, vocab=vocab, autocast=autocast,
                              reserve_bytes=optimizer_state_bytes(model, geo.optim),
                              headroom=0.15 if (field and geo.fp32_master and strategy == "full") else 0.07,
                              packed_mode=pack.mode if pack else "off")
            _event_and_print("sft_v2_memory_probe", micro=micro_c, gradient_checkpointing=gc_c, fits=ok)
            if gemma_adapter_small_batch and not ok:
                _event_and_print("sft_v2_small_batch_disabled", reason="memory_probe_failed", micro=micro_c, gradient_checkpointing=gc_c)
                gemma_adapter_small_batch = False
                _small_batch_target_eff = None
                geo = Geometry(geo.micro_batch, max(1, math.ceil(_small_batch_default_eff / geo.micro_batch)), geo.max_len, geo.gradient_checkpointing, geo.fp32_master, geo.optim)
            if ok:
                eff_target = geo.eff_batch
                geo = Geometry(micro_c, max(1, math.ceil(eff_target / micro_c)), geo.max_len, gc_c, geo.fp32_master, geo.optim)
                break
    telemetry.set_meta(
        handler="sft_v2", strategy=strategy, params_b=round(params_b, 3), seq_len=max_len, batch=geo.micro_batch,
        grad_accum=geo.grad_accum, eff_batch=geo.eff_batch, gradient_checkpointing=geo.gradient_checkpointing,
        fp32_master=geo.fp32_master, optim=geo.optim, liger=use_liger, vocab=vocab, train_n=len(train_ex), val_n=len(dev_ex),
    )

    if pack:
        telemetry.set_meta(packing=pack.telemetry(), train_blocks=pack.n_blocks,
                           rows_per_step=pack.rows_per_step(geo.eff_batch))

    def rows_per_step() -> int:
        return pack.rows_per_step(geo.eff_batch) if pack else geo.eff_batch

    collator = tokenize.PadCollator(tokenizer.pad_token_id)
    # training rows carry per-token loss weights while the hygiene prompt-loss weight is active; dev
    # evaluation, the timing probe and the dev pass keep the plain collator (plain CE, validator statistic)
    train_collator = hygiene.PromptWeightCollator(tokenizer.pad_token_id, hyg["plw"]) if (field_hygiene and hyg["plw"] > 0) else collator
    if pack:
        train_collator = pack.collator
    from forge.model import median_weight_rms

    # BaselineSummary.gradient_noise_scale (validator training statistic) feeds only the gated NEFTune
    # escalation (_neftune_alpha, field hygiene); the analytic LR law's noise term stays inert as before
    gns = None
    if strategy == "lora":
        # production trains the adapter at 1.5e-4 with effective batch 16; the prior
        # sits just above it and scales with sqrt(batch) (the Smol sweep chose
        # 4.4e-4 at batch 84, consistent with this law).
        # LR scan on gemma2-cas at batch 15 (2026-09-11), dropout 0.05: 1.9e-4 → 0.0364, 3e-4 → 0.0358,
        # 4.5e-4 → 0.0400. With dropout 0 (now the default) 1.9e-4 gives 0.0352 while 3e-4 gives 0.0363:
        # the two changes overshoot together, so the prior stays at 2e-4; the sweep explores ±0.3 decades
        prior_lr = 2.0e-4 * math.sqrt(max(1, geo.eff_batch) / 16.0)
        if sharded:
            # 12–40B models sharded over several GPUs: 1.5e-4 is the tested value
            # (Qwen3-32B dry-2v 0.464868); the 3B prior 2e-4 is untested there.
            prior_lr = min(prior_lr, 1.5e-4)
    else:
        prior_lr = lr_probe.analytic_lr(weight_rms=_layer_weight_rms(model) or median_weight_rms(model), params_b=params_b,
                                        eff_batch=rows_per_step(), gradient_noise_scale=gns)
        # the analytic law under-shoots the field by ~7x on Gemma-2-2B (2e-5 vs
        # the champion's 7.5e-5 bucket); anchor the prior to the bucket, the sweep decides
        if field:
            # the field route ships the champion's analytic law WITHOUT the size-cap floor:
            # the floored 1e-4 / 7.5e-5 was too hot everywhere it was tried against the raw
            # law (GossipCop 0.9868 vs 0.9503; CodeFeedback dev best at step 10; Gemma-2-2B
            # and Qwen2.5-3B regressions). Clamped to [2e-5, 1e-4]; FORGE_V2_FIELD_LR_MULT
            # scales it for experiments; no sweep (the short probe under-picks for full weights).
            _raw = lr_probe.analytic_lr(weight_rms=_layer_weight_rms(model) or median_weight_rms(model), params_b=params_b,
                                        eff_batch=rows_per_step(), gradient_noise_scale=gns)
            _fm = _recipe["lr_mult"]
            prior_lr = max(2.0e-5, min(1.0e-4, _raw)) * _fm
            _event_and_print("sft_v2_field_lr", prior_lr=prior_lr, raw=_raw, mult=_fm, eff_batch=geo.eff_batch, rows_per_step=rows_per_step())
        else:
            prior_lr = max(prior_lr, 0.5 * lr_probe.size_cap_lr(params_b))

    try:
        _lr_override = float(os.environ.get("FORGE_V2_LR", "") or 0.0)
    except ValueError:
        _lr_override = 0.0
    if _lr_override > 0:
        _event_and_print("sft_v2_lr_override", prior_lr=prior_lr, override=_lr_override)
        prior_lr = _lr_override
    elif field and not _field_sweep():
        _lr_override = prior_lr  # fixed LR: timing steps only, no sweep
    elif route_lr and route_lr > 0:
        # a routed recipe ships its tested LR: no sweep (the short-horizon probe
        # picks 2.6e-5–7.5e-5 for full weights, which lose 4–13% at 64 rows/step)
        _event_and_print("sft_v2_route_lr", prior_lr=prior_lr, route_lr=route_lr)
        prior_lr = route_lr
        _lr_override = route_lr
    if field and recipe_key == "minicpm5":
        # clean MiniCPM5-1B rule: the detector verdict it read and the LR this run applies (release-candidate-v8)
        _cm5_mult, _cm5_reason = clean_minicpm5_lr_rule(recipe_key, base_damage, base_clean_ref)
        _event_and_print("sft_v2_minicpm5_lr_rule", verdict=base_damage, clean_ref=base_clean_ref, reason=_cm5_reason,
                         rule_mult=_cm5_mult, lr_mult=_recipe["lr_mult"], lr=prior_lr,
                         env_lr_mult=os.environ.get("FORGE_V2_FIELD_LR_MULT") or None, cheap_full=bool(cheap_full))

    # A separate opt-in FIELD experiment owns its bounded probes. Suppress the
    # older generic sweep even when conflicting experiment flags are rejected.
    field_calibration = field and _field_lr_calibration()
    if field_calibration:
        _lr_override = prior_lr
    if _g2c:
        # damaged Gemma2 champion recipe: the executed peak LR, fixed (no clamp, multiplier or sweep). FORGE_V2_LR and the
        # opt-in FIELD calibration experiment keep precedence.
        try:
            _g2c_env_lr = float(os.environ.get("FORGE_V2_LR", "") or 0.0)
        except ValueError:
            _g2c_env_lr = 0.0
        _g2c_fixed = not field_calibration and _g2c_env_lr <= 0
        _g2c_prior = prior_lr
        if _g2c_fixed:
            prior_lr = GEMMA2_CHAMPION_PEAK_LR
            _lr_override = prior_lr
        _event_and_print("sft_v2_gemma2_champion_route", params_b=round(params_b, 3), rows=len(rows), train_n=len(train_ex),
                         dev_n=len(dev_ex), **(_g2c_geometry or {}), micro=geo.micro_batch, grad_accum=geo.grad_accum,
                         blocks_per_step=geo.eff_batch, rows_per_step=pack.rows_per_step(geo.eff_batch),
                         steps_per_epoch=packing.optimizer_steps(pack.n_blocks, geo.micro_batch, geo.grad_accum),
                         lr=prior_lr, lr_source="champion_fixed" if _g2c_fixed else ("calibration" if field_calibration else "env"),
                         prior_lr_before=_g2c_prior, min_epochs=GEMMA2_CHAMPION_MIN_EPOCHS)

    _beta1 = 0.9
    _beta2 = 0.999
    if gemma_adapter_small_batch:
        _beta2 = 0.999 ** (geo.eff_batch / max(1, _small_batch_default_eff))
        _event_and_print("sft_v2_small_batch_optimizer", default_eff=_small_batch_default_eff,
                         candidate_eff=geo.eff_batch, ratio=geo.eff_batch / max(1, _small_batch_default_eff),
                         beta1=_beta1, beta2=_beta2, eps=1e-8, weight_decay=0.0)

    def optimizer_factory(params: list, lr: float):
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(_beta1, _beta2), eps=1e-8)

    def make_args(num_epochs: float, learning_rate: float, warmup_steps: int, micro: int, accum: int, seed: int = 7) -> TrainingArguments:
        kwargs = dict(
            output_dir=workdir(spec), overwrite_output_dir=True, num_train_epochs=num_epochs,
            per_device_train_batch_size=micro, gradient_accumulation_steps=accum, learning_rate=learning_rate,
            lr_scheduler_type="cosine_with_min_lr", lr_scheduler_kwargs={"min_lr_rate": _min_lr_rate(0.25 if field else 0.1)}, warmup_steps=warmup_steps,
            weight_decay=0.0, adam_beta1=_beta1, adam_beta2=_beta2, adam_epsilon=1e-8,
            optim=geo.optim, max_grad_norm=1.0, bf16=True, fp16=False,
            gradient_checkpointing=False,  # enabled on the model directly above
            logging_steps=5, save_strategy="no", eval_strategy="no", report_to=[], remove_unused_columns=False,
            dataloader_num_workers=2, disable_tqdm=True, seed=seed, train_sampling_strategy="group_by_length", length_column_name="length",
            neftune_noise_alpha=_neftune_alpha(baseline_summary, strategy, escalate=field_hygiene),
        )
        return TrainingArguments(**compatible_dataclass_kwargs(TrainingArguments, kwargs, allow_removed={"overwrite_output_dir"}))

    def with_lengths(exs: list[dict]) -> Dataset:
        return Dataset.from_list([{**ex, "length": len(ex["input_ids"])} for ex in exs])

    train_ds = with_lengths(pack.features if pack else train_ex)

    # ---- learning-rate probe on cached batches from the real dataloader ----
    probe_trainer = Trainer(model=model, args=make_args(1.0, prior_lr, 10, geo.micro_batch, geo.grad_accum), train_dataset=train_ds, data_collator=pack.collator if pack else collator)
    steps_per_epoch = (packing.optimizer_steps(pack.n_blocks, geo.micro_batch, geo.grad_accum)
                       if pack else max(1, len(train_ex) // geo.eff_batch))
    remaining = deadline.remaining_hard()
    probe_budget = min(0.15 * remaining, 480.0) if _lr_override <= 0 else 0.0  # fixed LR: timing steps only
    batches: list[dict[str, Any]] = []
    # Cache enough distinct micro-batches that a 100-step probe never re-sees a
    # batch (re-seen batches are memorised and bias the sweep toward the fastest
    # memoriser rather than the best learner).
    n_probe_batches = min(400, max(100 * geo.grad_accum, 60))
    if field_calibration:
        n_probe_batches = 25 * geo.grad_accum  # one full, non-wrapping experimental panel prefix
    lr, t_per_step, diag = prior_lr, None, {}
    # D4: a fixed-LR full-FT default run (e.g. the late fallback rung 2) needs only the 3-step timing probe, so it runs
    # above TIMING_PROBE_MIN_REMAINING_S instead of the sweep's 20 min; without timing the planner would plan one full
    # epoch and the clock would stop it in warmup / near peak LR with no anneal. (The rung-2 entry floor stays
    # CHEAP_FULL_MIN_REMAINING_S: raising it would send late failures straight to the adapter, directive 4.)
    _probe_floor = (TIMING_PROBE_MIN_REMAINING_S if (recipe_key and _lr_override > 0 and not field_calibration)
                    else MIN_TASK_SECONDS_FOR_V2)
    if _probe_floor < MIN_TASK_SECONDS_FOR_V2 and remaining <= MIN_TASK_SECONDS_FOR_V2:
        n_probe_batches = min(n_probe_batches, max(8, 4 * geo.grad_accum))  # timing only: 3 steps
        _event_and_print("sft_v2_late_timing_probe", remaining_s=round(remaining, 1), cheap_full=bool(cheap_full),
                         batches=n_probe_batches)
    if remaining > _probe_floor:
        try:
            for i, b in enumerate(probe_trainer.get_train_dataloader()):
                if i >= n_probe_batches:
                    break
                batches.append({k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in b.items() if k != "length"})
            import random as _random

            _random.Random(11).shuffle(batches)  # the sampler front-loads the longest rows; timing needs a representative mix
            if field_calibration:
                from forge.tuning.field_lr_calibration import run_field_calibration

                _probe_reserve = EXPORT_RESERVE_S + 2.0 * _estimate_eval_seconds(len(dev_ex), None, geo) + _dev_pass_estimate(len(dev_ex), None, geo)
                lr, t_per_step, diag = run_field_calibration(
                    probe_trainer, batches, prior_lr=prior_lr, deadline=deadline,
                    finish_reserve_s=_probe_reserve,
                    log=lambda n, d: _event_and_print(n, **_flat(d)),
                )
            else:
                # 1) timing at the initial geometry (3 steps) at a safe LR: step time does
                #    not depend on the LR, and an un-warmed step at the full prior spikes
                #    a full fine-tune (Gemma: 0.23 -> 2.7); the sweep judges LRs properly
                t_micro, warm = lr_probe.measure_step_time(
                    model, batches, lr=0.1 * prior_lr, grad_accum=geo.grad_accum, optimizer_factory=optimizer_factory, autocast_bf16=autocast,
                )
                t_per_step = t_micro
                _event_and_print("sft_v2_timing", t_per_step=round(t_per_step, 4) if t_per_step else None, eff_batch=geo.eff_batch,
                                 losses=[round(x, 4) for x in warm[:3]])
                # 2) batch adaptation (below) needs the timing; 3) the sweep runs after it at the final geometry
        except Exception as exc:
            if field_calibration:
                # The experimental runner only returns after verified recovery.
                # Restoration failure must never train on partially probed state.
                raise
            if _is_oom(exc):
                _event_and_print("lr_probe_oom", micro=geo.micro_batch)
                if gemma_adapter_small_batch:
                    _event_and_print("sft_v2_small_batch_candidate_drift", reason="lr_probe_oom", micro=geo.micro_batch, grad_accum=geo.grad_accum)
                    gemma_adapter_small_batch = False
                    _small_batch_target_eff = None
                _free_cuda()
                geo = Geometry(max(1, geo.micro_batch // 2), geo.grad_accum * 2, geo.max_len, True, geo.fp32_master, geo.optim)
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                lr, t_per_step = prior_lr, None
            else:
                _event_and_print("lr_probe_failed", error=f"{type(exc).__name__}: {exc}")
                lr, t_per_step = prior_lr, None
    del probe_trainer
    gc.collect()
    _free_cuda()

    # ---- large-model batch rule (forge.tuning.large_model_rule; release-candidate-v9, default ON): in-scope full-FT
    # default tasks >= 3.0 B only. Below 3.0 B this block is not entered, so the run is 854bf7a's. Grows the effective
    # batch (same micro-batch, more accumulation) toward the champion's tokens per step and scales the fixed FIELD LR by
    # min(LR_FACTOR_MAX 2.0, (B_new / B_old) ** LR_ALPHA 0.5) (phase-ba arm C; c1f7501 used 0.75). ----
    if (large_model_rule_eligible(field=field, recipe_key=recipe_key, cheap_full=cheap_full, g2c=_g2c, strategy=strategy,
                                  params_b=params_b)
            and t_per_step and _lr_override > 0 and not field_calibration):
        _lm_sorted = sorted(lengths)
        _lm_len = large_model_rule.champion_len(_lm_sorted[len(_lm_sorted) // 2],
                                                _lm_sorted[min(len(_lm_sorted) - 1, int(0.99 * len(_lm_sorted)))])
        _lm_units = pack.n_blocks if pack else len(train_ex)
        _lm_tokens = float(sum(len(ex["input_ids"]) for ex in train_ex))
        _lm_window = deadline.remaining_hard() - (EXPORT_RESERVE_S + 2.0 * _estimate_eval_seconds(len(dev_ex), t_per_step, geo)
                                                  + _dev_pass_estimate(len(dev_ex), t_per_step, geo))
        _lm = large_model_rule.plan_growth(
            train_tokens=_lm_tokens, tokens_per_unit=_lm_tokens / max(1, _lm_units), eff_units=geo.eff_batch,
            micro=geo.micro_batch, t_step_s=t_per_step, window_s=_lm_window,
            max_epochs=(4.0 if len(train_ex) < 10000 else 3.0), champ_len=_lm_len,
            fill=float((pack.stats.get("fill") if pack else None) or large_model_rule.DEFAULT_FILL), gpu_gb=per_gpu_gb)
        if _lm is not None:
            _lm_new, _lm_info = _lm
            _lm_f = _lm_info["lr_factor"]
            t_per_step = t_per_step * _lm_new / geo.eff_batch
            geo = Geometry(geo.micro_batch, max(1, _lm_new // geo.micro_batch), geo.max_len, geo.gradient_checkpointing,
                           geo.fp32_master, geo.optim)
            lr = prior_lr * _lm_f
            _lr_override = lr
            _event_and_print("sft_v2_large_model_batch", params_b=round(params_b, 3), lr_old=prior_lr, lr_new=lr,
                             champ_len=_lm_len, **_lm_info)
        else:
            _event_and_print("sft_v2_large_model_batch_kept", params_b=round(params_b, 3), eff_units=geo.eff_batch,
                             t_per_step=round(t_per_step, 4), window_s=round(_lm_window, 1))

    # ---- adaptive effective batch: guarantee enough optimizer updates in short tasks ----
    # (not on the field route: the champion keeps its bucket and trains fewer steps)
    if t_per_step and not field:
        samples_per_s = geo.eff_batch / t_per_step
        window0 = deadline.remaining_hard() - (EXPORT_RESERVE_S + 2.0 * _estimate_eval_seconds(len(dev_ex), t_per_step, geo)
                                                + _dev_pass_estimate(len(dev_ex), t_per_step, geo))
        affordable = max(0.0, window0) * 0.85 * samples_per_s
        # update budget by strategy: production's adapter takes ~575 updates at batch
        # 16 on a 45-min task; the champion's full fine-tune takes ~300 at batch 64
        # (watcher archive, Sept 7 uploads) — full weights want the larger batch
        default_updates = 500 if strategy == "lora" else 200
        try:
            min_updates = int(os.environ.get("FORGE_V2_MIN_UPDATES", str(default_updates)))
        except ValueError:
            min_updates = default_updates
        eff_new = int(affordable / max(1, min_updates))
        eff_floor = _small_batch_target_eff if gemma_adapter_small_batch else 16
        eff_new = max(eff_floor, min(geo.eff_batch, eff_new))
        if gemma_adapter_small_batch:
            eff_new = max(geo.micro_batch, math.ceil(eff_new / geo.micro_batch) * geo.micro_batch)
        else:
            eff_new = max(geo.micro_batch, (eff_new // geo.micro_batch) * geo.micro_batch)
        if eff_new != geo.eff_batch:
            scale = math.sqrt(eff_new / geo.eff_batch)
            _event_and_print("sft_v2_eff_batch", old=geo.eff_batch, new=eff_new, lr_old=lr, lr_new=lr * scale,
                             affordable_samples=int(affordable), samples_per_s=round(samples_per_s, 2))
            t_per_step = t_per_step * eff_new / geo.eff_batch
            lr = lr * scale
            geo = Geometry(geo.micro_batch, eff_new // geo.micro_batch, geo.max_len, geo.gradient_checkpointing, geo.fp32_master, geo.optim)

    # ---- learning-rate sweep at the final geometry ----
    # Adapters skip the sweep (2026-09-12): its 51-step probes cannot resolve the
    # ~1% differences between 2.2e-4 and 3.2e-4 and pick the higher LR in about
    # half the runs (Gemma-2-2B: 3.16e-4 → 0.035777 / 0.035153 vs 2.24e-4 → 0.034361
    # with everything else equal), while the fixed prior is at or near the best on
    # Qwen2.5-3B as well. The 15% of the budget goes to training instead.
    # FORGE_V2_ADAPTER_SWEEP=1 re-enables it for experiments.
    if strategy == "lora" and os.environ.get("FORGE_V2_ADAPTER_SWEEP", "0") != "1" and _lr_override <= 0:
        _event_and_print("sft_v2_sweep_skipped", strategy=strategy, lr=prior_lr)
        _lr_override = prior_lr
    _event_and_print("sft_v2_sweep_gate", batches=len(batches), t_per_step=round(t_per_step, 4) if t_per_step else None,
                     lr_override=_lr_override, remaining_s=round(deadline.remaining_hard(), 1))
    if batches and t_per_step and _lr_override <= 0 and deadline.remaining_hard() > MIN_TASK_SECONDS_FOR_V2:
        try:
            # the champion stack pays for its LR search even on 45-min tasks; full
            # weights are far more LR-sensitive than an adapter, so give them more
            sweep_budget = min((0.18 if strategy == "full" else 0.15) * deadline.remaining_hard(), 480.0)
            lr, t_sw, diag = lr_probe.lr_search(
                model, batches, center_lr=lr, budget_s=sweep_budget, grad_accum=geo.grad_accum,
                optimizer_factory=optimizer_factory, autocast_bf16=autocast, log=lambda n, d: _event_and_print(n, **_flat(d)),
                known_t_per_step=t_per_step,
            )
        except Exception as exc:
            if _is_oom(exc):
                _event_and_print("lr_sweep_oom", micro=geo.micro_batch)
                _free_cuda()
            else:
                _event_and_print("lr_sweep_failed", error=f"{type(exc).__name__}: {exc}")
    del batches
    gc.collect()
    _free_cuda()

    # ---- field hygiene: Gaussian difficulty core once the measured coverage is known ----
    if field_hygiene:
        cov = hygiene.coverage(rows=len(train_ex), eff_batch=geo.eff_batch, t_per_step=t_per_step, budget_s=hours_total * 3600.0)
        kept_n = None
        try:
            kept, target = hygiene.difficulty_subsample(train_ex, coverage=cov, seed=7)
            if target is not None and len(kept) < len(train_ex):
                train_ex = kept
                train_ds = with_lengths(train_ex)
                kept_n = len(train_ex)
                telemetry.set_meta(train_n=len(train_ex))
        except Exception as exc:
            _event_and_print("sft_v2_hygiene_failed", stage="subsample", error=f"{type(exc).__name__}: {exc}"[:200])
        _event_and_print("sft_v2_hygiene", dedup_removed=hyg["dedup_removed"], groups=hyg["groups"], grouped_rows=hyg["grouped_rows"],
                         dedup=hyg["dedup"], subsample_kept=kept_n, plw=hyg["plw"], prompt_completion_ratio=hyg["ratio"],
                         neftune_alpha=_neftune_alpha(baseline_summary, strategy, escalate=field_hygiene), coverage=cov,
                         train_n=len(train_ex))

    # ---- planning constants ----
    dev_eval_est = _estimate_eval_seconds(len(dev_ex), t_per_step, geo)
    export_reserve = 60.0 if strategy == "lora" else EXPORT_RESERVE_S
    try:
        _soup_reserve_evals = float(os.environ.get("FORGE_V2_SOUP_RESERVE_EVALS", "0") or 0)  # experiment: keep time for the soup
    except ValueError:
        _soup_reserve_evals = 0.0
    finish_reserve = export_reserve + (1.5 + _soup_reserve_evals) * dev_eval_est + _dev_pass_estimate(len(dev_ex), t_per_step, geo)
    if recipe_key:
        # phase-ao fix (Qwen3.5-4B long rows: reserve 3,696 s vs 1,793 s left -> zero steps, rung 1 lost): drop the dev-pass
        # estimate from the reserve when keeping it would leave no training window. The dev pass is still gated at run
        # time on the remaining time. Full-FT default route only; inert whenever the window is not starved.
        _fr = starved_finish_reserve(remaining=deadline.remaining_hard(), finish_reserve=finish_reserve,
                                     dev_pass_s=_dev_pass_estimate(len(dev_ex), t_per_step, geo), t_step=t_per_step)
        if _fr != finish_reserve:
            _event_and_print("sft_v2_finish_reserve_dev_pass_dropped", finish_reserve_s=round(finish_reserve, 1),
                             new_s=round(_fr, 1), remaining_s=round(deadline.remaining_hard(), 1))
            finish_reserve = _fr
    max_epochs_total = 4.0 if len(train_ex) < 10000 else 3.0
    try:
        if os.environ.get("FORGE_V2_MAX_EPOCHS"):
            max_epochs_total = float(os.environ["FORGE_V2_MAX_EPOCHS"])
    except ValueError:
        pass
    steps_per_epoch = (packing.optimizer_steps(pack.n_blocks, geo.micro_batch, geo.grad_accum)
                       if pack else max(1, len(train_ex) // geo.eff_batch))

    # ---- training with validator-style dev selection ----
    # D2: the RAM gate reads the container's cgroup headroom (selection.available_ram); above 3 B params a FULL-weight
    # pool keeps at most 3 bf16 snapshots so pool + fp32 soup sum + export stay well inside G.O.D's 110 GiB container
    # (adapter snapshots are adapter-only and keep k = 8, cab930c)
    pool = selection.SnapshotPool(k=selection.snapshot_pool_k(float(params_b) * 1e9) if strategy == "full"
                                  else selection.SNAPSHOT_POOL_K)
    _event_and_print("sft_v2_snapshot_pool", k=pool.k, params_b=round(float(params_b), 3),
                     available_ram=selection.available_ram(), **selection.cgroup_memory_peak())
    stop_at = _StopAt(deadline, finish_reserve)
    state = {"best": float("inf"), "best_step": 0, "overfit": 0, "persisted_best": float("inf"), "persisted_step": 0,
             "eval_count": 0, "last_persist_at": 0.0, "step_offset": 0, "epochs_done": 0.0, "eval_secs": 0.0,
             "persist_secs": 0.0, "stopped_overfit": False, "last_phase_complete": False, "restarts": 0, "last_eval_step": -1}
    eval_bs = selection.eval_batch_for(max_len=max_len, vocab=vocab)
    dev_rows_for_selection = dev_ex

    def eval_dev(m: Any) -> float:
        t0 = time.perf_counter()
        loss, _ = selection.per_example_dev_loss(m, dev_rows_for_selection, collator, batch_size=eval_bs, autocast_bf16=autocast)
        state["eval_secs"] += time.perf_counter() - t0
        return loss

    def persist(m: Any, cpu_state: dict[str, Any] | None, step: int, truth: str, reason: str) -> bool:
        t0 = time.perf_counter()
        try:
            save_adapter(m, tokenizer, spec.output_dir, artifact_truth=truth, optimizer_step=step, truth_reason=reason,
                         state_dict=cpu_state)
            return True
        except Exception as exc:
            _event_and_print("sft_v2_persist_failed", step=step, error=f"{type(exc).__name__}: {exc}")
            return False
        finally:
            state["persist_secs"] += time.perf_counter() - t0

    row_loss = os.environ.get("FORGE_V2_ROW_LOSS", "0") == "1"

    try:
        _lora_plus = float(os.environ.get("FORGE_V2_LORA_PLUS", "0") or 0)
    except ValueError:
        _lora_plus = 0.0

    class LoraPlusTrainer(Trainer):
        """LoRA+: the B matrices train at `ratio` x the A-matrix learning rate.
        The configured learning rate is the B rate (the successful plain-LoRA
        setting); A gets lr / ratio."""

        def create_optimizer(self):
            if self.optimizer is None and _lora_plus > 1.0:
                import torch as _t

                a_params = [p for n, p in self.model.named_parameters() if p.requires_grad and "lora_A" in n]
                b_params = [p for n, p in self.model.named_parameters() if p.requires_grad and "lora_B" in n]
                rest = [p for n, p in self.model.named_parameters() if p.requires_grad and "lora_A" not in n and "lora_B" not in n]
                lr = self.args.learning_rate
                groups = [{"params": b_params, "lr": lr}, {"params": a_params, "lr": lr / _lora_plus}]
                if rest:
                    groups.append({"params": rest, "lr": lr})
                self.optimizer = _t.optim.AdamW(groups, lr=lr, betas=(_beta1, _beta2), eps=1e-8, weight_decay=0.0, fused=_t.cuda.is_available())
                _event_and_print("sft_v2_lora_plus", ratio=_lora_plus, lr_b=lr, lr_a=lr / _lora_plus, n_b=len(b_params), n_a=len(a_params))
            return super().create_optimizer()

    class RowLossTrainer(Trainer):
        """Per-example objective (validator statistic) instead of the token mean."""

        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            self.model_accepts_loss_kwargs = False  # the trainer then divides by the accumulation steps

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # noqa: ANN001
            from forge.tuning.rowloss import row_mean_loss

            try:
                loss = row_mean_loss(model, inputs, use_fused=use_liger)
            except Exception as exc:
                import traceback

                _event_and_print("sft_v2_row_loss_error", error=f"{type(exc).__name__}: {exc}",
                                 trace=traceback.format_exc()[-600:])
                raise
            return (loss, None) if return_outputs else loss

    TrainerCls = RowLossTrainer if row_loss else (LoraPlusTrainer if (_lora_plus > 1.0 and strategy == "lora") else Trainer)

    if gemma_adapter_small_batch:
        class OptimizerTelemetryTrainer(TrainerCls):
            """Emit the optimizer object’s actual groups and resolved betas."""

            def create_optimizer(self):
                result = super().create_optimizer()
                opt = self.optimizer
                if opt is None:
                    raise RuntimeError("Trainer created no optimizer")
                groups = list(getattr(opt, "param_groups", ()))
                group_rows = []
                for index, group in enumerate(groups):
                    betas = group.get("betas")
                    group_rows.append({
                        "index": index,
                        "params": len(group.get("params", ())),
                        "numel": sum(int(p.numel()) for p in group.get("params", ())),
                        "lr": float(group.get("lr")),
                        "beta1": float(betas[0]) if betas is not None else None,
                        "beta2": float(betas[1]) if betas is not None else None,
                        "eps": float(group.get("eps")),
                        "weight_decay": float(group.get("weight_decay")),
                    })
                betas = groups[0].get("betas") if groups else None
                _event_and_print(
                    "sft_v2_optimizer_created",
                    groups=len(groups),
                    trainable_params=sum(len(g.get("params", ())) for g in groups),
                    optimizer_groups=group_rows,
                    beta1=float(betas[0]) if betas is not None else None,
                    beta2=float(betas[1]) if betas is not None else None,
                    optimizer=type(opt).__name__,
                )
                return result

        TrainerCls = OptimizerTelemetryTrainer
    if row_loss:
        _event_and_print("sft_v2_row_loss", enabled=True, fused=use_liger)
    elif field_hygiene and hyg["plw"] > 0:
        TrainerCls = make_prompt_weight_trainer(Trainer, use_fused=use_liger)
        _event_and_print("sft_v2_prompt_loss_weight", plw=hyg["plw"], ratio=hyg["ratio"], fused=use_liger)

    try:
        _ema_frac = float(os.environ.get("FORGE_V2_TAIL_EMA", "0") or 0)  # fraction of the phase averaged (0 = off)
    except ValueError:
        _ema_frac = 0.0
    try:
        _seed_base = int(os.environ.get("FORGE_V2_SEED", "7") or 7)  # data-order seed (dev split stays seed 7)
    except ValueError:
        _seed_base = 7
    try:
        _restart_factor = float(os.environ.get("FORGE_V2_RESTART_LR_FACTOR", "0.5") or 0.5)  # later-phase LR = lr * f**phase
    except ValueError:
        _restart_factor = 0.5
    if strategy != "lora":
        _ema_frac = 0.0  # a full-weight fp32 average would not fit next to the optimizer state
    ema: dict[str, Any] = {}

    class SelectCallback(TrainerCallback):
        def __init__(self, eval_every: int, total_steps: int, window_s: float | None = None) -> None:
            self.eval_every = eval_every
            self.total_steps = total_steps
            self.window_s = window_s
            self.governed = False
            self.ema_start = int(total_steps * (1.0 - _ema_frac)) if _ema_frac > 0 else None
            self.ema_n = 0

        def _ema_update(self, m: Any, local_step: int) -> None:
            # Uniform (LAWA-style) running mean of the trainable weights over the
            # tail of the phase — a dense average the greedy soup can consider.
            if self.ema_start is None or local_step < self.ema_start:
                return
            import torch

            with torch.no_grad():
                self.ema_n += 1
                w = 1.0 / self.ema_n
                for n, p in m.named_parameters():
                    if not p.requires_grad:
                        continue
                    if n not in ema:
                        ema[n] = p.detach().float().clone()
                    else:
                        ema[n].mul_(1.0 - w).add_(p.detach().float(), alpha=w)

        def on_step_end(self, args, st, control, **kw):  # noqa: ANN001
            local_step = int(st.global_step)
            step = state["step_offset"] + local_step
            m = kw.get("model")
            if m is not None:
                self._ema_update(m, local_step)
            if stop_at.should_stop():
                control.should_training_stop = True
            end_step = int(getattr(st, "max_steps", 0) or self.total_steps)
            due = local_step > 0 and (local_step % self.eval_every == 0 or local_step >= end_step)
            if not due or m is None or not dev_rows_for_selection:
                return control
            t0 = time.perf_counter()
            loss = eval_dev(m)
            state["last_eval_step"] = step
            dt = time.perf_counter() - t0
            state["eval_count"] += 1
            telemetry.eval_point(step, loss)
            improved = loss < state["best"]
            state["phase_evals"] = state.get("phase_evals", 0) + 1
            if improved:
                state["best"], state["best_step"] = loss, step
                state["overfit"] = 0
            elif loss > state["best"] * 1.05 and state["phase_evals"] > 1:
                state["overfit"] += 1
                # PLACEHOLDER (release-candidate-v9; NOT IMPLEMENTED): phase-bg exposure arm slot (2 of 2): arms C/D
                # suppress this DEV overfit stop (FORGE_V2_NO_OVERFIT_STOP on 32984fb). Nothing here runs; see slot 1.
                if state["overfit"] >= 3:
                    _event_and_print("sft_v2_early_stop", step=step, best=state["best"], best_step=state["best_step"])
                    state["stopped_overfit"] = True
                    control.should_training_stop = True
            else:
                state["overfit"] = 0
            admitted = pool.consider(m, loss, step)
            since_persist = time.monotonic() - state["last_persist_at"]
            if improved and (since_persist > 120.0 or loss < state["persisted_best"] * 0.995 or state["persisted_best"] == float("inf")):
                snap = pool.best()["state"] if (admitted and pool.best() and pool.best()["step"] == step) else selection.snapshot_trainable(m)
                if persist(m, snap, step, ARTIFACT_PARTIAL_TRAINED_BEST, "dev_minimum"):
                    state["persisted_best"], state["persisted_step"] = loss, step
                    state["last_persist_at"] = time.monotonic()
            _event_and_print("sft_v2_eval", step=step, dev_loss=round(loss, 6), best=round(state["best"], 6), eval_s=round(dt, 1),
                             pool=len(pool.items), improved=improved)
            if not self.governed and t_step:
                # keep evaluation (incl. persistence) under ~6% of training time
                self.governed = True
                per_eval = dt + 0.5 * (state["persist_secs"] / max(1, state["eval_count"]))
                widened, slack_info = governed_eval_every(self.eval_every, per_eval, t_step, self.total_steps,
                                                          self.window_s if _eval_slack(_csq_field) else None)
                if slack_info:
                    _event_and_print("sft_v2_eval_governor_slack", eval_every=widened, **slack_info)
                if widened > self.eval_every:
                    _event_and_print("sft_v2_eval_governor", eval_every=widened, eval_s=round(dt, 1))
                    self.eval_every = widened
            return control

    _event_and_print("sft_v2_train_start", elapsed_s=round(time.monotonic() - t_start, 1), lr=lr, prior_lr=prior_lr,
                     t_per_step=t_per_step, steps_per_epoch=steps_per_epoch, finish_reserve_s=round(finish_reserve, 1),
                     **_flat({"probe": diag}))
    trainer = None
    t_step = t_per_step
    phase = 0
    oom_retries = 0
    oom_continue = False  # the next loop iteration is an OOM retry, not a new phase
    # Two-seed exact soup (2026-09-12): a second independently initialised adapter
    # trained for one annealed epoch in a different data order, combined with the
    # first as an exact equal-weight soup (rank 2r). On ru-AAQG the exact soup of
    # two seeds scored 0.557561 vs 0.558400 / 0.564622 for the seeds themselves
    # (the seed spread alone is ~1%). Only where a second epoch fits comfortably.
    two_seed = os.environ.get("FORGE_V2_TWO_SEED", "0") == "1" and strategy == "lora"
    fullft_extended = False  # window use: one DEV-gated continuation past the epoch cap (fullft_epoch_cap_extension)
    two_seed_result: tuple[float, int, str] | None = None
    seed_b_active = False
    seed_a: dict[str, Any] = {}
    while phase < 4:
        remaining = deadline.remaining_hard()
        window = remaining - finish_reserve
        min_window = max(240.0, 25.0 * t_step) if t_step else 240.0
        if state["stopped_overfit"]:
            # One warm restart from the best snapshot at a lower learning rate
            # when plenty of budget remains; a second overfit stop ends training.
            if field or state["restarts"] >= 1 or window < max(600.0, 60.0 * (t_step or 10.0)) or pool.best() is None:
                break
            selection.load_state(model, pool.best()["state"])
            state["restarts"] += 1
            state["stopped_overfit"] = False
            state["overfit"] = 0
            lr = lr * 0.3
            _event_and_print("sft_v2_restart", lr=lr, from_step=pool.best()["step"], window_s=round(window, 1))
        if phase > 0 and window < min_window:
            break
        if phase > 0 and field and not oom_continue:
            _ext = fullft_epoch_cap_extension(
                recipe_key=recipe_key, phase=phase, extended=fullft_extended, last_phase_complete=state["last_phase_complete"],
                stopped_overfit=state["stopped_overfit"], best_step=state["best_step"], step_offset=state["step_offset"],
                epochs_done=state["epochs_done"], max_epochs=max_epochs_total, window_s=window, min_window_s=min_window,
                t_step=t_step, steps_per_epoch=steps_per_epoch)
            if _ext <= 0:
                break  # the field route is one cosine: no continuation phases
            fullft_extended = True
            _event_and_print("sft_v2_fullft_epoch_cap_extended", cap=max_epochs_total, extra_epochs=_ext,
                             window_s=round(window, 1), best=state["best"], best_step=state["best_step"],
                             lr_next=lr * max(0.1, _restart_factor ** phase))
            max_epochs_total += _ext
        oom_continue = False
        if phase > 0 and os.environ.get("FORGE_V2_SOUP_FIRST", "0") == "1" and t_step:
            # experiment: a later phase only when the greedy soup (3 evals) and the
            # dev pass still fit after it; otherwise stop and soup the phase-0 pool
            soup_need = 3.0 * dev_eval_est + _dev_pass_estimate(len(dev_ex), t_step, geo) + 60.0
            if window - soup_need < max(10, int(0.15 * steps_per_epoch)) * t_step:
                _event_and_print("sft_v2_soup_first_stop", window_s=round(window, 1), soup_need_s=round(soup_need, 1))
                break
        if two_seed and phase == 1 and not seed_b_active and two_seed_result is None and state["last_phase_complete"] and t_step and pool.items:
            need = steps_per_epoch * t_step * 1.2 + 4.0 * dev_eval_est + 90.0
            if window > need:
                seed_b_no_fit = False
                try:
                    loss_a, state_a, members_a = selection.greedy_soup(
                        model, pool, eval_dev, time_budget_s=min(300.0, 0.2 * window), log=lambda n, d: _event_and_print(n, **_flat(d))
                    )
                    if state_a is None:
                        loss_a, state_a = pool.best()["loss"], pool.best()["state"]
                    seed_a = {"loss": loss_a, "state": state_a, "members": members_a, "step": state["step_offset"]}
                    _reinit_lora(model, _seed_base + 1000)
                    pool.items = []
                    state.update(best=float("inf"), best_step=0, overfit=0, stopped_overfit=False, last_eval_step=-1, phase_evals=0)
                    seed_b_active = True
                    _event_and_print("sft_v2_two_seed_a", dev_loss=round(loss_a, 6), members=members_a, need_s=round(need, 1), window_s=round(window, 1))
                    if _eval_slack():
                        # the soup and re-init spent part of `window`: plan seed B (and size the eval governor's slack)
                        # from the live deadline. The no-fit restore runs below, outside this handler (Codex re-review).
                        window, fits = seed_b_window(deadline, finish_reserve, min_window)
                        seed_b_no_fit = not fits
                except Exception as exc:
                    _event_and_print("sft_v2_two_seed_failed", stage="a", error=f"{type(exc).__name__}: {exc}")
                    two_seed = False
                if seed_b_no_fit:
                    _event_and_print("sft_v2_two_seed_b_no_fit", window_s=round(window, 1), min_window_s=round(min_window, 1))
                    restore_seed_a_or_fail(model, seed_a, pool, state)   # raises: never plan seed B on a re-initialised adapter
                    seed_b_active = False
                    two_seed = False
                    break
            else:
                _event_and_print("sft_v2_two_seed_skipped", need_s=round(need, 1), window_s=round(window, 1))
                two_seed = False
        # Each phase is a fully annealed cosine cycle of at most one epoch (the
        # affordable fraction when less fits), so every phase ends on a low
        # learning rate: the phase endpoints are the checkpoints the soup wants.
        # Without timing (probe OOM) phase 0 is one epoch and the clock guard
        # stops it if the epoch does not fit.
        if os.environ.get("FORGE_V2_RICH_EPOCHS") == "1" and t_step and steps_per_epoch * t_step * 6.0 < window and max_epochs_total < 8.0:
            # an epoch costs < 1/6 of the window: keep training (annealed cycles at
            # decreasing peak LR); the overfit early stop ends it when dev turns.
            max_epochs_total = 8.0
            _event_and_print("sft_v2_epoch_cap_raised", max_epochs=max_epochs_total, epoch_s=round(steps_per_epoch * t_step, 1))
        epochs_left = max_epochs_total - state["epochs_done"]
        if t_step:
            achievable = window * 0.85 / t_step
            if sharded:
                # device_map models run ~20% slower in steady state than the timing
                # probe says and their dev evals cost minutes (32B: 141 s each), so
                # plan the annealed cycle at 80% of the window to keep the final eval,
                # the greedy soup and the dev pass (dry-2v lost all three: 0.98 epoch,
                # clock stop, soup skipped).
                achievable *= 0.8
            if field and phase == 0:
                # field route: one cosine over everything that fits, planned 25% past
                # the window (champion planner) so the wall-clock stop lands near the
                # LR floor; the early stop and the best-snapshot pool cover the tail
                epochs = min(epochs_left, 1.25 * achievable / steps_per_epoch)
                if _g2c:
                    # damaged Gemma2 champion recipe: the champion planner's 1.3-epoch floor on the cosine horizon (on
                    # 59469903 it planned 390 steps and the clock stopped it at 250, LR 0.48 x peak)
                    epochs = min(epochs_left, max(GEMMA2_CHAMPION_MIN_EPOCHS, epochs))
            elif phase == 0 or state["restarts"] or seed_b_active:
                # first cycle: one fully annealed epoch (the strongest single
                # candidate on small data), also the timing measurement
                epochs = min(epochs_left, 1.0, achievable / steps_per_epoch)
            else:
                # then one continuous cosine over everything that still fits:
                # repeated one-epoch cycles restart at a high LR each time and
                # the dev loss spends most of each cycle recovering (Qwen2.5 run)
                epochs = min(epochs_left, achievable / steps_per_epoch)
        else:
            epochs = min(epochs_left, 1.0) if phase == 0 else 0.0
        epochs = round(max(0.0, epochs), 3)
        total_steps = int(math.ceil(steps_per_epoch * epochs))
        if total_steps < 1 or (phase > 0 and total_steps < max(10, int(0.15 * steps_per_epoch))):
            break
        warmup = min(200, max(3, int(0.03 * total_steps)))
        eval_every = max(10, min(max(1, steps_per_epoch // 8), max(1, total_steps // 8)))
        phase_lr = lr if (phase == 0 or state["restarts"] or seed_b_active) else lr * max(0.1, _restart_factor ** phase)
        state["overfit"] = 0
        state["phase_evals"] = 0
        _event_and_print("sft_v2_plan", phase=phase, lr=phase_lr, t_per_step=t_step, epochs=epochs, total_steps=total_steps,
                         warmup=warmup, eval_every=eval_every, window_s=round(window, 1), epochs_done=round(state["epochs_done"], 3))
        cb = SelectCallback(eval_every, total_steps, window_s=window)
        trainer = TrainerCls(
            model=model, args=make_args(epochs, phase_lr, warmup, geo.micro_batch, geo.grad_accum,
                                        seed=_seed_base + (1000 if seed_b_active else phase)),
            train_dataset=train_ds, data_collator=train_collator, callbacks=[cb, telemetry.make_trainer_callback(spec.output_dir)],
        )
        t_phase0 = time.monotonic()
        eval0, pers0 = state["eval_secs"], state["persist_secs"]
        try:
            trainer.train()
        except Exception as exc:
            if not _is_oom(exc):
                raise
            step = int(getattr(trainer.state, "global_step", 0) or 0)
            _event_and_print("sft_v2_train_oom", phase=phase, step=step, micro=geo.micro_batch,
                             gradient_checkpointing=geo.gradient_checkpointing)
            try:
                trainer.optimizer = None  # release optimizer state before retrying
            except Exception:
                pass
            del trainer
            trainer = None
            _free_cuda()
            # Preserve the incumbent native retry ladder if the candidate geometry
            # runs out of memory. The result is explicitly marked as candidate drift
            # and cannot be treated as a scored batch10 arm.
            if gemma_adapter_small_batch:
                _event_and_print("sft_v2_small_batch_candidate_drift", reason="train_oom", step=step,
                                 micro=geo.micro_batch, grad_accum=geo.grad_accum, eff_batch=geo.eff_batch)
                gemma_adapter_small_batch = False
                _small_batch_target_eff = None
            # Keep whatever was trained; shrink the geometry and continue in a new phase.
            state["step_offset"] += step
            state["epochs_done"] += step / steps_per_epoch
            if step > 0:
                phase += 1
            if oom_retries >= 3:
                break
            oom_retries += 1
            oom_continue = True
            new_micro = max(1, geo.micro_batch // 2) if (geo.gradient_checkpointing or geo.micro_batch > 1) else 1
            geo = Geometry(new_micro, max(1, geo.eff_batch // new_micro), geo.max_len, True, geo.fp32_master, geo.optim)
            if pack:
                steps_per_epoch = packing.optimizer_steps(pack.n_blocks, geo.micro_batch, geo.grad_accum)
            _apply_gc(True)
            continue
        if trainer is None:
            continue
        steps_done = int(getattr(trainer.state, "global_step", 0) or 0)
        planned_steps = int(getattr(trainer.state, "max_steps", 0) or total_steps)
        state["step_offset"] += steps_done
        state["epochs_done"] += steps_done / steps_per_epoch
        state["last_phase_complete"] = steps_done >= planned_steps
        try:  # release the optimizer state (fp32 Adam moments) before the next phase / soup / dev pass
            trainer.optimizer = None
            trainer.lr_scheduler = None
        except Exception:
            pass
        del trainer
        trainer = None
        model.zero_grad(set_to_none=True)
        gc.collect()
        _free_cuda()
        phase_wall = (time.monotonic() - t_phase0) - (state["eval_secs"] - eval0) - (state["persist_secs"] - pers0)
        if steps_done >= 5 and phase_wall > 0:
            t_step = phase_wall / steps_done
        _event_and_print("sft_v2_phase_end", phase=phase, steps=steps_done, planned=planned_steps, epochs_done=round(state["epochs_done"], 3),
                         t_per_step_measured=round(t_step, 3) if t_step else None, best=state["best"], best_step=state["best_step"],
                         remaining_s=round(deadline.remaining_hard(), 1))
        if ema and dev_rows_for_selection and deadline.remaining_hard() > finish_reserve + 2 * dev_eval_est:
            try:
                import torch as _torch

                with _torch.no_grad():
                    current = {n: p.detach().clone() for n, p in model.named_parameters() if n in ema}
                    for n, p in model.named_parameters():
                        if n in ema:
                            p.data.copy_(ema[n].to(p.dtype))
                ema_loss = eval_dev(model)
                admitted = pool.consider(model, ema_loss, state["step_offset"])
                if ema_loss < state["best"]:
                    state["best"], state["best_step"] = ema_loss, state["step_offset"]
                _event_and_print("sft_v2_tail_ema", phase=phase, n=cb.ema_n, dev_loss=round(ema_loss, 6), admitted=admitted,
                                 best=round(state["best"], 6))
                with _torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in current:
                            p.data.copy_(current[n])
                del current
            except Exception as exc:
                _event_and_print("sft_v2_tail_ema_failed", error=f"{type(exc).__name__}: {exc}")
            ema.clear()
            _free_cuda()
        if seed_b_active:
            seed_b_active = False
            try:
                two_seed_result = _finish_two_seed(model, pool, seed_a, eval_dev, state, deadline, finish_reserve, dev_eval_est,
                                                   log=_event_and_print, flat=_flat)
            except Exception as exc:
                _event_and_print("sft_v2_two_seed_failed", stage="b", error=f"{type(exc).__name__}: {exc}")
                two_seed_result = None
                if not pool.items and seed_a.get("state") is not None:
                    selection.load_state(model, seed_a["state"])
                    pool.items = [{"loss": seed_a["loss"], "step": seed_a["step"], "state": seed_a["state"]}]
                    state["best"], state["best_step"] = seed_a["loss"], seed_a["step"]
            if two_seed_result is not None:
                phase += 1
                break
        phase += 1
        if steps_done < planned_steps and not state["stopped_overfit"]:
            break  # stopped by the clock (or trainer); no time for another phase
    final_step = state["step_offset"]
    _event_and_print("sft_v2_train_end", step=final_step, best=state["best"], best_step=state["best_step"],
                     phases=phase, remaining_s=round(deadline.remaining_hard(), 1))
    if final_step <= 0:
        # never ship the untrained floor while a cheaper rung can still train (owner directive 4): fail loudly. Only on
        # the in-scope full-FT default route (recipe_key set, D3); every other route keeps cab930c (return, floor kept)
        _event_and_print("sft_v2_zero_trained_steps", phases=phase, remaining_s=round(deadline.remaining_hard(), 1),
                         cheap_full=bool(cheap_full), raised=bool(recipe_key))
        if recipe_key:
            raise ZeroTrainedStepsError("sft_v2 took no optimizer step")
        return  # cab930c: out-of-scope routes return with the floor kept

    if two_seed_result is not None:
        # the chosen candidate (single seed or the exact two-seed soup) is loaded in the model
        state["best"], state["best_step"], state["last_eval_step"] = two_seed_result[0], two_seed_result[1], final_step
        pool.items = []
    # ---- final evaluation of the last weights (may be the best) ----
    if dev_rows_for_selection and state["last_eval_step"] != final_step and deadline.remaining_hard() > export_reserve + dev_eval_est:
        try:  # D5: a failed final eval (e.g. OOM) must not discard the trained weights; the pool/best keep selection
            last_loss = eval_dev(model)
            telemetry.eval_point(final_step, last_loss)
            if last_loss < state["best"]:
                state["best"], state["best_step"] = last_loss, final_step
            pool.consider(model, last_loss, final_step)
        except Exception as exc:
            _event_and_print("sft_v2_final_eval_failed", error=f"{type(exc).__name__}: {exc}"[:300])
            _free_cuda()

    # ---- greedy soup ----
    chosen_state: dict[str, Any] | None = None
    chosen_loss = state["best"]
    chosen_step = state["best_step"]
    soup_budget = max(0.0, 0.5 * (deadline.remaining_hard() - export_reserve - _dev_pass_estimate(len(dev_ex), t_per_step, geo)))
    measured_eval = state["eval_secs"] / max(1, state["eval_count"]) if state["eval_count"] else dev_eval_est
    if pool.items and dev_rows_for_selection and len(pool.items) >= 2 and soup_budget < 2.5 * measured_eval:
        _event_and_print("sft_v2_soup_skipped", budget_s=round(soup_budget, 1), eval_s=round(measured_eval, 1))
        pool.items = pool.items[:1]  # keep the best only: the soup would not fit a single trial
    if pool.items and dev_rows_for_selection:
        try:
            soup_loss, soup_state, members = selection.greedy_soup(
                model, pool, eval_dev, time_budget_s=soup_budget, log=lambda n, d: _event_and_print(n, **_flat(d))
            )
            if soup_state is not None and soup_loss <= chosen_loss + 1e-9:
                chosen_state, chosen_loss, chosen_step = soup_state, soup_loss, final_step if members > 1 else pool.best()["step"]
            elif pool.best() is not None:
                chosen_state = pool.best()["state"]
                selection.load_state(model, chosen_state)
                chosen_loss, chosen_step = pool.best()["loss"], pool.best()["step"]
        except Exception as exc:
            _event_and_print("sft_v2_soup_failed", error=f"{type(exc).__name__}: {exc}")
            if pool.best() is not None:
                chosen_state = pool.best()["state"]
                selection.load_state(model, chosen_state)
    elif pool.best() is not None:
        chosen_state = pool.best()["state"]
        selection.load_state(model, chosen_state)

    # ---- dev pass: one low-LR epoch over the held-out slice from the chosen weights ----
    # Harness knobs: FORGE_DEV_PASS_EPOCHS (passes, default 1), FORGE_DEV_PASS_LR_MULT
    # (default 0.25), FORGE_V2_KEEP_STAGES=1 (export the pre-dev-pass weights and each
    # pass to <output_dir>-stages/<name> so the stages can be scored separately).
    dev_pass_s = _dev_pass_estimate(len(dev_ex), t_per_step, geo)
    try:
        _dp_epochs = max(1, int(os.environ.get("FORGE_DEV_PASS_EPOCHS", "1") or 1))
        _dp_mult = float(os.environ.get("FORGE_DEV_PASS_LR_MULT", "0.25") or 0.25)
    except ValueError:
        _dp_epochs, _dp_mult = 1, 0.25
    _keep_stages = os.environ.get("FORGE_V2_KEEP_STAGES", "0") == "1"

    def _keep_stage(name: str) -> None:
        if not _keep_stages:
            return
        try:
            d = spec.output_dir.rstrip("/") + "-stages/" + name
            save_adapter(model, tokenizer, d, artifact_truth=ARTIFACT_PARTIAL_TRAINED_BEST, optimizer_step=final_step,
                         truth_reason="stage_" + name)
            _event_and_print("sft_v2_stage_kept", stage=name, path=d)
        except Exception as exc:
            _event_and_print("sft_v2_stage_failed", stage=name, error=f"{type(exc).__name__}: {exc}")

    _dp_factory = dev_pass_optimizer_factory(geo, optimizer_factory, strategy=strategy, betas=(_beta1, _beta2))
    if _dp_factory is None and dev_ex and len(dev_ex) >= 100:
        _event_and_print("sft_v2_dev_pass_skipped", reason="bf16_weights_without_8bit_adamw", optim=geo.optim)
    if (_dp_factory is not None and dev_ex and len(dev_ex) >= 100 and deadline.remaining_hard() > export_reserve + dev_pass_s + 30
            and os.environ.get("FORGE_DEV_PASS", "1") != "0"):
        _keep_stage("soup")
        for _k in range(_dp_epochs):
            if _k > 0 and deadline.remaining_hard() <= export_reserve + dev_pass_s + 30:
                break
            _dp_pre = chosen_state if chosen_state is not None else (pool.best()["state"] if pool.best() is not None else None)
            try:
                if recipe_key:
                    dev_pass_with_oom_retry(model, dev_ex, collator, lr=lr * _dp_mult, geo=geo, autocast=autocast,
                                            optimizer_factory=_dp_factory, restore_state=_dp_pre,
                                            remaining_s=deadline.remaining_hard, export_reserve_s=export_reserve,
                                            dev_pass_s=dev_pass_s)
                else:
                    _dev_pass(model, dev_ex, collator, lr=lr * _dp_mult, geo=geo, autocast=autocast, optimizer_factory=_dp_factory)
                chosen_state = None  # weights in the model are now the final ones
                _event_and_print("sft_v2_dev_pass_done", rows=len(dev_ex), lr=lr * _dp_mult, pass_index=_k + 1)
                _keep_stage(f"devpass{_k + 1}")
            except Exception as exc:
                _event_and_print("sft_v2_dev_pass_failed", error=f"{type(exc).__name__}: {exc}")
                if pool.best() is not None:
                    chosen_state = pool.best()["state"]
                    selection.load_state(model, chosen_state)
                break

    # ---- final save (bf16 weights) ----
    truth = ARTIFACT_COMPLETE_BEST if state["last_phase_complete"] else ARTIFACT_PARTIAL_TRAINED_BEST
    if final_step <= 0:
        return
    final_state = chosen_state if chosen_state is not None else selection.snapshot_trainable(model)
    ok = persist(model, final_state, chosen_step or final_step, truth, "sft_v2_final")
    if not ok and recipe_key and not trained_artifact_present(spec):
        # D5: trained weights exist but nothing trained reached the output (only the floor would ship with a clean exit).
        # In-scope full-FT route: raise so the dispatcher's ladder retries (cheap full FT, then the adapter) or the task
        # fails loudly. Out-of-scope routes keep cab930c.
        _event_and_print("sft_v2_persist_failed_no_artifact", step=final_step, remaining_s=round(deadline.remaining_hard(), 1))
        raise PersistFailedError("sft_v2 trained but could not save any trained artifact")
    _event_and_print("sft_v2_final", saved=ok, chosen_loss=chosen_loss, chosen_step=chosen_step, truth=truth,
                    remaining_s=round(deadline.remaining_hard(), 1), **{f"cgroup_{k}": v for k, v in selection.cgroup_memory_peak().items()})
    telemetry.write_into(spec.output_dir)


def trained_artifact_present(spec: TaskSpec) -> bool:
    """True when v2 already persisted a trained (non-floor) artifact."""
    import json

    try:
        with open(os.path.join(spec.output_dir, "forge_artifact_truth.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload.get("truth") in (ARTIFACT_PARTIAL_TRAINED_BEST, ARTIFACT_COMPLETE_BEST)
    except Exception:
        return False


def _layer_weight_rms(model: Any) -> float | None:
    """Median RMS of 2-D weights inside transformer blocks (embeddings and the
    output head excluded, as in the public LR law)."""
    try:
        vals = []
        for name, p in model.named_parameters():
            if p.dim() != 2 or "embed" in name or "lm_head" in name or "layers." not in name:
                continue
            vals.append(p.detach().float().pow(2).mean().sqrt().item())
        if not vals:
            return None
        vals.sort()
        return vals[len(vals) // 2]
    except Exception:
        return None


def _reinit_lora(model: Any, seed: int) -> None:
    """Fresh LoRA initialisation (A: kaiming-uniform as PEFT does, B: zeros) under `seed`."""
    import torch

    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if ".lora_A." in name:
                torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            elif ".lora_B." in name:
                p.zero_()


def _rebuild_default_adapter(model: Any, new_r: int) -> None:
    """Replace the 'default' LoRA adapter by a fresh one of rank `new_r` (same
    alpha/targets/dropout), so an exact two-adapter soup can live in the model
    and be exported through the ordinary PEFT save path."""
    import copy

    cfg = copy.deepcopy(model.peft_config["default"])
    cfg.r = int(new_r)
    if isinstance(getattr(cfg, "rank_pattern", None), dict):
        cfg.rank_pattern = {}
    if isinstance(getattr(cfg, "alpha_pattern", None), dict):
        cfg.alpha_pattern = {}
    model.delete_adapter("default")
    model.add_adapter("default", cfg)
    model.set_adapter("default")
    for name, p in model.named_parameters():
        p.requires_grad_(".lora_" in name)


def _load_two_seed_soup(model: Any, state_a: dict[str, Any], state_b: dict[str, Any]) -> int:
    """Write A* = [A1; A2], B* = [B1 B2] into the (rank 2r) default adapter; alpha
    unchanged, so the update equals the equal-weight average of the two adapters."""
    import torch

    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in state_a and name in state_b:
                if ".lora_A." in name:
                    p.data.copy_(torch.cat([state_a[name], state_b[name]], dim=0).to(device=p.device, dtype=p.dtype))
                    n += 1
                elif ".lora_B." in name:
                    p.data.copy_(torch.cat([state_a[name], state_b[name]], dim=1).to(device=p.device, dtype=p.dtype))
                    n += 1
    return n


def _finish_two_seed(model: Any, pool: Any, seed_a: dict[str, Any], eval_dev: Callable[[Any], float], state: dict[str, Any],
                     deadline: Any, finish_reserve: float, dev_eval_est: float, *, log: Callable[..., None],
                     flat: Callable[[dict], dict]) -> tuple[float, int, str] | None:
    """After the seed-B epoch: soup B, build the exact A+B soup at rank 2r,
    keep the best of {A, B, A+B} by dev loss (near-ties go to the soup)."""
    window = deadline.remaining_hard() - finish_reserve
    if not pool.items or window < 4.0 * dev_eval_est + 30.0:
        return None
    loss_b, state_b, members_b = selection.greedy_soup(
        model, pool, eval_dev, time_budget_s=min(300.0, 0.3 * window), log=lambda n, d: log(n, **flat(d))
    )
    if state_b is None:
        loss_b, state_b = pool.best()["loss"], pool.best()["state"]
    step_b = state["step_offset"]
    loss_a, state_a = float(seed_a["loss"]), seed_a["state"]
    r = int(model.peft_config["default"].r)
    loss_ab: float | None = None
    try:
        _rebuild_default_adapter(model, 2 * r)
        n = _load_two_seed_soup(model, state_a, state_b)
        if n == 0:
            raise RuntimeError("no adapter tensors matched")
        loss_ab = float(eval_dev(model))
    except Exception as exc:
        log("sft_v2_two_seed_soup_failed", error=f"{type(exc).__name__}: {exc}")
        loss_ab = None
    best_single = min(loss_a, loss_b)
    if loss_ab is not None and math.isfinite(loss_ab) and loss_ab <= best_single * 1.003:
        chosen, chosen_loss, chosen_step = "soup", loss_ab, step_b
    else:
        if int(model.peft_config["default"].r) != r:
            _rebuild_default_adapter(model, r)
        if loss_a <= loss_b:
            selection.load_state(model, state_a)
            chosen, chosen_loss, chosen_step = "a", loss_a, int(seed_a["step"])
        else:
            selection.load_state(model, state_b)
            chosen, chosen_loss, chosen_step = "b", loss_b, step_b
    log("sft_v2_two_seed", loss_a=round(loss_a, 6), loss_b=round(loss_b, 6), loss_soup=round(loss_ab, 6) if loss_ab is not None else None,
        chosen=chosen, members_a=seed_a.get("members"), members_b=members_b, rank=int(model.peft_config["default"].r))
    return chosen_loss, chosen_step, chosen


def _min_lr_rate(default: float = 0.1) -> float:
    try:
        return float(os.environ.get("FORGE_V2_MIN_LR_RATE", str(default)))
    except ValueError:
        return default


def _neftune_alpha(summary: Any, strategy: str = "full", *, escalate: bool = False) -> float | None:
    """NEFTune alpha for the trainer. Adapters train without it (2026-09-11: on
    Qwen2.5-3B-Instruct/ru-AAQG, NEFTune off at the 2e-4 prior scores 0.559766
    vs 0.561–0.563 with it, and 0.517658 vs full FT 0.523538 at 52k rows; unpacked
    ~300-token rows carry ~3.7x the field's packed-4096 noise at the same alpha).
    Full weights keep alpha 1 (tested); with ``escalate`` (field hygiene) alpha 5 when
    the validator's gradient_noise_scale exceeds 1.0, as in the reference stack.
    FORGE_V2_NEFTUNE overrides (0 = off)."""
    forced = os.environ.get("FORGE_V2_NEFTUNE", "").strip()
    if forced:
        try:
            v = float(forced)
            return None if v <= 0 else v
        except ValueError:
            pass
    if strategy == "lora":
        return None
    if escalate:
        try:
            gns = getattr(summary, "gradient_noise_scale", None)
            if gns is not None and float(gns) > 1.0:
                return 5.0
        except Exception:
            pass
    return 1.0


def make_prompt_weight_trainer(trainer_cls: Any, *, use_fused: bool) -> Any:
    """Trainer subclass for the hygiene prompt-loss weight: the collator's per-token
    ``loss_weights`` drive a weighted token cross-entropy normalised by the weight sum
    (forge.data.hygiene.weighted_token_loss). Labels stay masked, so dev evaluation and
    the dev pass keep the plain objective."""

    class PromptWeightTrainer(trainer_cls):
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            self.model_accepts_loss_kwargs = False  # the trainer then divides by the accumulation steps

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # noqa: ANN001
            loss = hygiene.weighted_token_loss(model, inputs, use_fused=use_fused)
            return (loss, None) if return_outputs else loss

    return PromptWeightTrainer


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "_"))
        else:
            out[key] = v
    return out


def _estimate_eval_seconds(n_dev: int, t_per_step: float | None, geo: Geometry) -> float:
    if n_dev <= 0:
        return 0.0
    if t_per_step is None:
        return 60.0
    # forward-only per row ~ 1/3 of a training micro-row; batched eval is cheaper still
    per_row = (t_per_step / max(1, geo.eff_batch)) / 3.0
    return max(5.0, n_dev * per_row + 5.0)


def _dev_pass_estimate(n_dev: int, t_per_step: float | None, geo: Geometry) -> float:
    if n_dev <= 0:
        return 0.0
    if t_per_step is None:
        return 120.0
    return max(20.0, 1.5 * (n_dev / max(1, geo.eff_batch)) * t_per_step + 20.0)


def dev_pass_optimizer_factory(geo: Geometry, default_factory: Callable, *, strategy: str = "full",
                               betas: tuple[float, float] = (0.9, 0.999)) -> Callable | None:
    """Optimizer of the post-selection dev pass (release defect D6). fp32 master weights keep the run's torch AdamW. bf16
    weights take bitsandbytes' paged 8-bit AdamW, the optimizer the training run's memory probe reserved for (2 B/param);
    plain AdamW there allocates bf16 moments at 4 B/param, can OOM at the probed micro-batch (then the pass is silently
    dropped) and was never the trained recipe. Without 8-bit AdamW a bf16 run skips the dev pass (returns None)."""
    if strategy != "full" or geo.fp32_master:  # adapters train fp32 adapter weights: unchanged
        return default_factory
    if "8bit" not in str(geo.optim) or not _bnb_available():
        return None

    def factory(params: list, lr: float):
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW8bit(params, lr=lr, betas=betas, eps=1e-8, weight_decay=0.0)

    return factory


def starved_finish_reserve(*, remaining: float, finish_reserve: float, dev_pass_s: float, t_step: float | None) -> float:
    """phase-ao planner fix: the finish reserve without the dev-pass estimate when keeping it would leave less than
    max(25% of the remaining time, 3 measured steps) to train; otherwise unchanged."""
    floor = max(0.25 * remaining, 3.0 * (t_step or 0.0))
    if remaining - finish_reserve < floor and dev_pass_s > 0:
        return max(0.0, finish_reserve - dev_pass_s)
    return finish_reserve


DEV_PASS_RETRY_FACTOR = 1.5  # a half-micro retry is budgeted at 1.5 x the one-pass estimate (smaller micro runs slower)
DEV_PASS_RETRY_MARGIN_S = 60.0


def dev_pass_retry_fits(*, remaining_s: float, export_reserve_s: float, dev_pass_s: float) -> bool:
    """RESP-074-1 B1: a retry runs only if a conservative complete half-micro pass plus the export still fit."""
    return remaining_s > export_reserve_s + DEV_PASS_RETRY_FACTOR * max(0.0, dev_pass_s) + DEV_PASS_RETRY_MARGIN_S


def dev_pass_with_oom_retry(model: Any, dev: list[dict], collator: Any, *, lr: float, geo: Geometry, autocast: bool,
                            optimizer_factory: Callable, restore_state: dict[str, Any] | None,
                            remaining_s: Callable[[], float], export_reserve_s: float, dev_pass_s: float) -> None:
    """phase-ao fix (Qwen3.5-4B dev pass OOM at micro 24, vocabulary 248k): on a CUDA OOM, restore the pre-dev-pass
    weights and free memory; then retry once at half the micro-batch (same effective batch) ONLY if a conservative full
    retry plus the export reserve still fits the deadline (``remaining_s()`` read after the restore). Otherwise keep the
    restored trained weights and skip the retry. Other errors, and OOMs without a restorable state, propagate."""
    try:
        _dev_pass(model, dev, collator, lr=lr, geo=geo, autocast=autocast, optimizer_factory=optimizer_factory)
        return
    except Exception as exc:
        if not _is_oom(exc) or geo.micro_batch <= 1 or restore_state is None:
            raise
        _event_and_print("sft_v2_dev_pass_oom_retry", micro=geo.micro_batch, new_micro=max(1, geo.micro_batch // 2))
    try:
        model.zero_grad(set_to_none=True)
    except Exception:
        pass
    selection.load_state(model, restore_state)
    _free_cuda()
    left = float(remaining_s())
    if not dev_pass_retry_fits(remaining_s=left, export_reserve_s=export_reserve_s, dev_pass_s=dev_pass_s):
        _event_and_print("sft_v2_dev_pass_retry_skipped", remaining_s=round(left, 1), export_reserve_s=export_reserve_s,
                         dev_pass_s=round(dev_pass_s, 1))
        return  # the restored pre-dev-pass weights (soup or best) are the final weights
    half = max(1, geo.micro_batch // 2)
    geo2 = Geometry(half, max(1, math.ceil(geo.eff_batch / half)), geo.max_len, geo.gradient_checkpointing,
                    geo.fp32_master, geo.optim)
    _dev_pass(model, dev, collator, lr=lr, geo=geo2, autocast=autocast, optimizer_factory=optimizer_factory)


def _dev_pass(model: Any, dev: list[dict], collator: Any, *, lr: float, geo: Geometry, autocast: bool, optimizer_factory: Callable) -> None:
    import torch

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optimizer_factory(params, lr)
    device = next(model.parameters()).device
    rows = list(dev)
    micro = geo.micro_batch
    accum = geo.grad_accum
    n_micro = 0
    for start in range(0, len(rows), micro):
        batch = collator(rows[start : start + micro])
        batch = {k: v.to(device) for k, v in batch.items()}
        if autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / accum
        else:
            loss = model(**batch).loss / accum
        loss.backward()
        n_micro += 1
        if n_micro % accum == 0 or start + micro >= len(rows):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
    del opt
    model.zero_grad(set_to_none=True)
    _free_cuda()
