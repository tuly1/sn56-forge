"""VENDORED, test-only: G.O.D a2b14db validator/evaluation/model_checks.py::model_is_a_finetune (lines 21-147, verbatim).
Source sha256 f2c23de7e0e14a9e6da6d82799eddfe45441fa316772b13e2b6926e1c6d74479 (worktrees/god-a2b14db). Only the logger import is replaced (core.logging is not installed here).
The validator calls it as model_is_a_finetune(original_model, finetuned_model) on the model loaded by
AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=False) (evaluators/instruct_text.py:446-455)."""
from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM

logger = logging.getLogger("god_vendor.model_checks")


def model_is_a_finetune(
    original_repo: str,
    finetuned_model: AutoModelForCausalLM,
    local_files_only: bool = False,
) -> bool:
    from transformers import AutoConfig

    max_retries = 3
    base_delay = 2

    # For local files, try to load config directly from snapshot
    if local_files_only:
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        cache_path = os.path.join(cache_dir, "hub", f"models--{original_repo.replace('/', '--')}")

        if os.path.exists(cache_path):
            snapshots_dir = os.path.join(cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = sorted(os.listdir(snapshots_dir))

                for snapshot in snapshots:
                    snapshot_path = os.path.join(snapshots_dir, snapshot)
                    if ".no_exist" in snapshot_path:
                        continue
                    config_path = os.path.join(snapshot_path, "config.json")

                    if os.path.exists(config_path) and os.path.getsize(config_path) > 0:
                        logger.info(f"Loading original model config from snapshot: {snapshot}")
                        try:
                            original_config = AutoConfig.from_pretrained(
                                snapshot_path, local_files_only=True, trust_remote_code=False
                            )
                            logger.info("Successfully loaded config from snapshot")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load config from snapshot {snapshot}: {e}")
                            continue
                else:
                    logger.error(f"No valid config found in snapshots for {original_repo}")
                    return False
            else:
                logger.error(f"No snapshots directory found for {original_repo}")
                return False
        else:
            logger.error(f"No cache found for {original_repo}")
            return False
    else:
        # Standard online loading with retries
        for attempt in range(max_retries):
            try:
                kwargs = {"token": os.environ.get("HUGGINGFACE_TOKEN"), "trust_remote_code": False}

                original_config = AutoConfig.from_pretrained(original_repo, **kwargs)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e

                error_msg = str(e).lower()
                if any(
                    pattern in error_msg for pattern in ["connection", "timeout", "5xx", "too many requests", "couldn't connect"]
                ):
                    delay = base_delay * (2**attempt)
                    logger.info(
                        f"HuggingFace connection issue (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    raise e
    finetuned_config = finetuned_model.config

    try:
        architecture_classes_match = finetuned_config.architectures == original_config.architectures
    except Exception as e:
        logger.debug(f"There is an issue with checking the architecture classes {e}")
        architecture_classes_match = False

    attrs_to_compare = [
        "architectures",
        "hidden_size",
        "n_layer",
        "intermediate_size",
        "head_dim",
        "hidden_act",
        "model_type",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    ]
    architecture_same = True
    for attr in attrs_to_compare:
        if getattr(original_config, attr, None) is not None:
            if not hasattr(finetuned_config, attr):
                architecture_same = False
                logger.info(f"Architecture mismatch: {attr} missing from the submitted config")
                break
            if getattr(original_config, attr) != getattr(finetuned_config, attr):
                architecture_same = False
                logger.info(
                    f"Architecture mismatch on {attr}: base={getattr(original_config, attr)} "
                    f"submitted={getattr(finetuned_config, attr)}"
                )
                break

    # Context window is checked directionally rather than for equality. Declaring a SMALLER context
    # than the base lowers the eval sequence_len (_load_and_update_evaluation_config only ever
    # shrinks it), which makes axolotl drop the longer held-out rows - so the submission is scored
    # on a shorter subset than everyone else and the mean losses stop being comparable. A LARGER
    # declared context cannot shrink anyone's eval set, and rejecting it would fail benign pipelines
    # that rewrite the field when applying rope scaling. YaRN tasks are unaffected either way: the
    # validator bakes the extended value into the base model miners receive.
    context_window_ok = True
    original_max_pos = getattr(original_config, "max_position_embeddings", None)
    finetuned_max_pos = getattr(finetuned_config, "max_position_embeddings", None)
    if isinstance(original_max_pos, int) and isinstance(finetuned_max_pos, int):
        if finetuned_max_pos < original_max_pos:
            context_window_ok = False
            logger.info(
                f"Context window shrunk: base max_position_embeddings={original_max_pos} "
                f"submitted={finetuned_max_pos} - this would shorten the evaluation set"
            )

    logger.info(
        f"Architecture same: {architecture_same}, Architecture classes match: {architecture_classes_match}, "
        f"Context window ok: {context_window_ok}"
    )
    return architecture_same and architecture_classes_match and context_window_ok
