# CPU twin of the submission image, used only for the local end-to-end smoke
# test on a GPU-less box. It exercises the exact entrypoint, argument flow, cache
# layout, output path, and dependency graph — everything except real GPU kernels
# — so a version conflict or import error surfaces here, not on the validator.
#
# Not a tournament artifact: the validator always builds the CUDA dockerfile.

FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=0 \
    TOKENIZERS_PARALLELISM=false

# CPU torch keeps the image small; the pinned training stack matches the GPU
# image so the smoke test validates the same versions that will ship.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 \
    && pip install \
        transformers==5.12.1 \
        peft==0.19.1 \
        trl==1.5.1 \
        accelerate==1.13.0 \
        "datasets==4.8.4" \
        "safetensors==0.8.0" \
        "sentencepiece==0.2.1" \
        "protobuf==7.35.1"

WORKDIR /app
COPY forge /app/forge
COPY pyproject.toml LICENSE.md NOTICE /app/

ENTRYPOINT ["python", "-m", "forge.cli"]
