# Validator-contract training image for SN56 text tournaments.
#
# The validator builds this from the repo root (internet available at BUILD time)
# and runs it with no network, /cache read-only, and a wall-clock kill. The
# entrypoint receives the standardized CLI args and writes the model to
# /app/checkpoints/{task_id}/{expected_repo_name}.
#
# Base: PyTorch 2.5.1 + CUDA 12.4, which supports the H100 (sm_90) GPUs the
# validator provides. Torch ships in the base image; we pin the training stack.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TOKENIZERS_PARALLELISM=false \
    # Runtime has no network; fail fast on any incidental hub call instead of
    # hanging on retries against the wall clock.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Pinned so the versions that reach the validator GPU are exactly these. torch is
# already present in the base image and satisfies these packages' requirements,
# so pip won't reinstall it; letting pip resolve the rest pulls the correct
# tokenizers/huggingface_hub/etc. transitively.
RUN pip install \
        transformers==4.57.1 \
        peft==0.17.1 \
        trl==0.24.0 \
        accelerate==1.10.1 \
        "datasets>=3.0,<5.0" \
        "safetensors>=0.4" \
        sentencepiece \
        protobuf

WORKDIR /app
COPY forge /app/forge
COPY pyproject.toml LICENSE.md NOTICE /app/

# All args after the entrypoint flow straight into argparse in forge/cli.py.
ENTRYPOINT ["python", "-m", "forge.cli"]
