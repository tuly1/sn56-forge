# Validator-contract training image for SN56 text tournaments.
#
# The validator builds this from the repo root (internet available at BUILD time)
# and runs it with no network, /cache read-only, and a wall-clock kill. The
# entrypoint receives the standardized CLI args and writes the model to
# /app/checkpoints/{task_id}/{expected_repo_name}.
#
# Runtime-aligned with G.O.D's validator/model-prep base image
# (OCI source revision 0bda5a13e4d52ceec58104f44fabb7bd314f9c02):
# PyTorch 2.9.1, Transformers 5.12.1, PEFT 0.19.1, TRL 1.5.1, Accelerate 1.13,
# and the flash-linear-attention/fla-core stack required by the forced Quasar
# rounds. The multi-platform index digest is pinned; the validator is amd64.

FROM axolotlai/axolotl:main-20260701-py3.11-cu128-2.9.1@sha256:3aa6403f59f2268bb8f686f6c748d6fad2949580a0048cd25a605eb2de239ee5

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TOKENIZERS_PARALLELISM=false \
    # Runtime has no network; fail fast on any incidental hub call instead of
    # hanging on retries against the wall clock.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Quasar's hybrid architecture imports causal-conv1d at load time. Build it
# against the base Torch ABI, exactly as G.O.D does; isolated builds can link a
# different libc10_cuda and fail only on the tournament GPU.
RUN TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX" uv pip install \
        --python /workspace/axolotl-venv/bin/python \
        --no-cache --no-build-isolation causal-conv1d==1.6.2.post1

WORKDIR /app
COPY forge /app/forge
COPY pyproject.toml LICENSE.md NOTICE /app/

# All args after the entrypoint flow straight into argparse in forge/cli.py.
ENTRYPOINT ["/workspace/axolotl-venv/bin/python", "-m", "forge.cli"]
