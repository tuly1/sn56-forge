# BloomZ-560M matched LoRA/full experiment

This package answers one narrow question: on one newly frozen public fixture,
does full-weight BloomZ-560M training at `1e-4` beat a matched LoRA control at
`1.5e-4`? Both arms use the same ordered rows, tokenizer, unpacked
microbatch-1/gradient-accumulation-16 geometry, 2,048-token training, 4,096-token
dev nomination, 256 optimizer steps, and evaluations at steps 64, 128, 192, and
256. Exactly four artifacts per arm are externally scored. Trainer loss only
nominates artifacts; the digest-pinned external dev score decides.

This is local matched evidence only. It does not reconstruct private official
rows, evaluator deployment, or absolute official calibration. The source
dataset's license and provenance are unresolved, so do not redistribute the
generated fixture without separate rights clearance. No command below mutates a
GPU provider or starts/stops a rented instance.

## Frozen identities

- Dataset:
  `AlekseyKorshuk/evol-codealpaca-v1-dpo@31c087a1492db443a3ace4247ef1880678b27aa4`
- Dataset parquet SHA-256:
  `b7d98f92731ad075bd01c1088c59816f05cf0f49605856b7e3007482a419535a`
- Model/tokenizer:
  `bigscience/bloomz-560m@a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05`
- Training image:
  `axolotlai/axolotl@sha256:97fba6ae924a55059bf48c5996014f0675d569df1b9c96e0cb0a0f922f355883`
- Evaluator image:
  `gradientsio/text-evaluator:basilica@sha256:860d49c7317a82b68d93b7e0e257091d810fdea12eee3013f373903092d279d0`
- Fixture manifest SHA-256:
  `f5e7ffb590a05ba3bf4ab925b442be6bcd4f743d8ddc9603f8e3fa6c93c327c7`
- Confirmation-blind training-manifest SHA-256:
  `3efcd00e9cd8d70c15bb324c264723ea292b5ed8c64bcdbf669f4a034372c336`

The fixture has 38,346 train rows, 1,024 dev rows, and 512 untouched
confirmation rows. Duplicate grouping uses all six disjoint SimHash bands
(11/11/11/11/10/10 bits), unions exact and Hamming-distance-at-most-five
edges, and splits only whole connected groups.

## 1. Build and reproduce the CPU fixture

Use a clean committed checkout. Every output directory and receipt path in this
workflow must be new and outside the repository.

```bash
REPO=/absolute/path/to/clean/week12-lane-b-bloomz-v1
set -euo pipefail
EXP=$REPO/experiments/20260831-bloomz-memory-fullft-v1
PARQUET=/absolute/path/train-00000-of-00001.parquet
BASE=/absolute/path/bloomz-560m-pinned
FIXTURE=/absolute/outside/repo/bloomz-public-fixture-v1
FIXTURE_2=/absolute/outside/repo/bloomz-public-fixture-v1-rebuild
cd "$REPO"

python -B "$EXP/build_fixture.py" \
  --parquet "$PARQUET" \
  --tokenizer-dir "$BASE" \
  --output-dir "$FIXTURE"

python -B "$EXP/build_fixture.py" \
  --parquet "$PARQUET" \
  --tokenizer-dir "$BASE" \
  --output-dir "$FIXTURE_2"

diff -qr "$FIXTURE" "$FIXTURE_2"
shasum -a 256 "$FIXTURE/manifest.json"
```

The final command must print
`f5e7ffb590a05ba3bf4ab925b442be6bcd4f743d8ddc9603f8e3fa6c93c327c7`.
Keep `confirmation.jsonl` inaccessible to training and dev selection.

Stage the only directory training may see. It contains exactly five flat files;
neither confirmation bytes nor confirmation filenames/hashes are present.

```bash
TRAIN_FIXTURE=/absolute/outside/repo/bloomz-training-fixture-v1
install -d -m 0755 "$TRAIN_FIXTURE"
for NAME in training-manifest.json train.jsonl dev.jsonl dataset-type.json baseline-stats.json; do
  install -m 0444 "$FIXTURE/$NAME" "$TRAIN_FIXTURE/$NAME"
done
chmod 0555 "$TRAIN_FIXTURE"
test "$(find "$TRAIN_FIXTURE" -mindepth 1 -maxdepth 1 -type f | wc -l | tr -d ' ')" = 5
test -z "$(find "$TRAIN_FIXTURE" -mindepth 1 -maxdepth 1 ! -type f -print -quit)"
EXPECTED_NAMES=$'baseline-stats.json\ndataset-type.json\ndev.jsonl\ntrain.jsonl\ntraining-manifest.json'
ACTUAL_NAMES=$(find "$TRAIN_FIXTURE" -mindepth 1 -maxdepth 1 -type f -exec basename {} \; | LC_ALL=C sort)
test "$ACTUAL_NAMES" = "$EXPECTED_NAMES"
shasum -a 256 "$TRAIN_FIXTURE/training-manifest.json"
```

The last command must print
`3efcd00e9cd8d70c15bb324c264723ea292b5ed8c64bcdbf669f4a034372c336`.

## 2. Bind the clean source and training image

The inspector rejects a dirty checkout, records the commit and Git trees,
captures every tracked file below `forge/` and this experiment with exact mode,
Git blob, size, and SHA-256, and inspects the already-present image without
pulling it. Before allocating an H100, verify the static cap arithmetic:

```bash
.venv/bin/python -B - <<'PY'
from decimal import Decimal
from forge.tuning import bloomz
b = bloomz.lease_budget()
assert b["stage_seconds"] == 23_460
assert b["decision_reserve_seconds"] == 540
assert b["stage_seconds"] + b["decision_reserve_seconds"] == 24_000
assert b["bootstrap_start_allowance_seconds"] == 2_700
assert b["ceo_custody_close_reserve_seconds"] == 5_400
assert 2_700 + 24_000 + 5_400 == b["total_seconds"] == 32_100
assert Decimal(b["hourly_rate_usd"]) * Decimal(32_100) / Decimal(3_600) == Decimal(b["maximum_cost_usd"])
print(b)
PY
```

Before the externally managed lease starts, obtain its trusted absolute start
epoch from the CEO-owned lifecycle path. This package never creates, starts,
stops, or closes a provider instance.
The maxima are: admissions `2×600`, training `2×7200`, dev scoring `8×570`,
validation `8×270`, and optional confirmation scoring `2×570` seconds. These
total 23,460 seconds, leaving 540 seconds for scientific decisions. Runtime
inspection must finish and bind the actual science start no later than provider
start +2,700 seconds. The immutable decision deadline is actual science start
+24,000 seconds.
The CEO-owned trusted path exclusively owns at least the final 5,400 seconds for
runtime-zero enforcement, byte-verified off-host custody, and provider closure,
including absolute provider deletion by start +32,100 seconds (8 hours 55
minutes), or `$17.8932571675` at `$2.006720430/hour`.

```bash
EVIDENCE=/absolute/outside/repo/bloomz-evidence
mkdir -p "$EVIDENCE"
: "${PROVIDER_START_EPOCH:?export the trusted provider start epoch before this block}"
case "$PROVIDER_START_EPOCH" in (*[!0-9]*|'') exit 91;; esac
export PLANNED_STAGE_SECONDS=23460

authority_stage_fields() {
  .venv/bin/python -B - "$EVIDENCE/runtime-authority.json" <<'PY'
import sys
from forge.tuning import bloomz
_, authority, digest = bloomz.load_runtime_authority(sys.argv[1])
lease = authority["lease"]
print(
    lease["provider_start_epoch"],
    lease["science_start_deadline_epoch"],
    lease["science_started_epoch"],
    lease["decision_deadline_epoch"],
    lease["provider_deadline_epoch"],
    lease["budget_sha256"],
    digest,
)
PY
}

run_stage() {
  local label=$1 cap=$2
  shift 2
  local provider_start science_start_deadline science_started decision_deadline provider_deadline budget_sha authority_sha
  read -r provider_start science_start_deadline science_started decision_deadline provider_deadline budget_sha authority_sha < <(authority_stage_fields)
  local now remaining required
  now=$(date +%s)
  test "$science_start_deadline" -eq $((provider_start + 2700))
  test "$science_started" -ge "$provider_start"
  test "$science_started" -le "$science_start_deadline"
  test "$decision_deadline" -eq $((science_started + 24000))
  test "$provider_deadline" -eq $((provider_start + 32100))
  test $((decision_deadline + 5400)) -le "$provider_deadline"
  test "${#budget_sha}" -eq 64
  test "${#authority_sha}" -eq 64
  test "$now" -ge "$science_started"
  remaining=$((decision_deadline - now))
  required=$PLANNED_STAGE_SECONDS
  if [ "$remaining" -lt "$required" ]; then
    echo "STOP_NO_SCIENCE: $label cannot fit bound decision deadline" >&2
    return 90
  fi
  timeout --signal=TERM "$cap" "$@"
  local status=$?
  [ "$status" -eq 0 ] || return "$status"
  PLANNED_STAGE_SECONDS=$((PLANNED_STAGE_SECONDS - cap))
}

skip_stage() {
  local cap=$1
  PLANNED_STAGE_SECONDS=$((PLANNED_STAGE_SECONDS - cap))
}

authority_science_check() {
  .venv/bin/python -B - "$EVIDENCE/runtime-authority.json" <<'PY'
import sys
from forge.tuning import bloomz
_, authority, _ = bloomz.load_runtime_authority(sys.argv[1])
bloomz.require_science_stage(
    authority["lease"], stage_max_seconds=1, remaining_planned_seconds=1
)
PY
}

python -B "$EXP/inspect_runtime.py" \
  --repo "$REPO" \
  --output "$EVIDENCE/runtime-authority.json" \
  --git /usr/bin/git \
  --docker /usr/bin/docker \
  --provider-start-epoch "$PROVIDER_START_EPOCH"
unset PROVIDER_START_EPOCH
```

## 3. Run the measured H100 admission probes

The two probes must run on the same single H100 used for training. Each executes
a real batch-1/sequence-2,048 forward, backward, and fused-AdamW step, then an
optimizer-resident labeled batch-1/sequence-4,096 evaluation. The receipt fails
above the 70% reserved-memory limit.

```bash
TRAIN_IMAGE='axolotlai/axolotl@sha256:97fba6ae924a55059bf48c5996014f0675d569df1b9c96e0cb0a0f922f355883'
MODEL_PARENT=/absolute/path/whose/bloomz-560m-pinned-child-is-bigscience--bloomz-560m

for ARM in control full; do
  LR=1.5e-4
  if [ "$ARM" = full ]; then LR=1e-4; fi
  run_stage "admission-$ARM" 600 docker run --rm --pull never --network none --gpus 'device=0' \
    -e PYTHONPATH=/app \
    -v "$REPO:/app:ro" \
    -v "$MODEL_PARENT:/cache/models:ro" \
    -v "$EVIDENCE:/evidence" \
    --entrypoint /workspace/axolotl-venv/bin/python \
    "$TRAIN_IMAGE" -B /app/experiments/20260831-bloomz-memory-fullft-v1/gpu_memory_probe.py \
    --model-dir /cache/models/bigscience--bloomz-560m \
    --arm "$ARM" \
    --learning-rate "$LR" \
    --runtime-authority /evidence/runtime-authority.json \
    --receipt "/evidence/$ARM-gpu-admission.json" \
    --max-reserved-ratio 0.70
done
```

## 4. Train both matched arms

`RUNS` is the only writable checkpoint mount. Training sees only
`TRAIN_FIXTURE`; the full fixture is not mounted. The launcher accepts at most
2.0 hours per arm, reserves 180 seconds for finalization, writes no visible
fallback model, and exits nonzero unless the complete four-artifact inventory
is `EXTERNAL_SCORE_READY`.

```bash
RUNS=/absolute/outside/repo/bloomz-runs
mkdir -p "$RUNS"
DATASET_TYPE='{"field_instruction":"instruct","field_output":"output","field_system":"system","no_input_format":"{instruction}","system_format":"{system}"}'

for ARM in control full; do
  PHASE=control
  LR=1.5e-4
  FULL_ENV=()
  if [ "$ARM" = full ]; then
    PHASE=candidate
    LR=1e-4
    FULL_ENV=(-e FORGE_ENABLE_EXPERIMENTAL_FULL_FT=1)
  fi

  run_stage "training-$ARM" 7200 docker run --rm --pull never --network none --gpus 'device=0' \
    -e PYTHONPATH=/app \
    -e FORGE_BLOOMZ_EXPERIMENT_ARM="$ARM" \
    -e FORGE_BLOOMZ_PHASE="$PHASE" \
    -e FORGE_BLOOMZ_LR="$LR" \
    -e FORGE_BLOOMZ_MAX_STEPS=256 \
    -e FORGE_BLOOMZ_TRAINING_MANIFEST=/train-fixture/training-manifest.json \
    -e FORGE_BLOOMZ_RUNTIME_AUTHORITY=/evidence/runtime-authority.json \
    -e FORGE_BLOOMZ_GPU_ADMISSION_RECEIPT="/evidence/$ARM-gpu-admission.json" \
    "${FULL_ENV[@]}" \
    -v "$REPO:/app:ro" \
    -v "$BASE:/cache/models/bigscience--bloomz-560m:ro" \
    -v "$TRAIN_FIXTURE:/train-fixture:ro" \
    -v "$EVIDENCE:/evidence:ro" \
    -v "$RUNS:/app/checkpoints" \
    --entrypoint /workspace/axolotl-venv/bin/python \
    "$TRAIN_IMAGE" -B /app/experiments/20260831-bloomz-memory-fullft-v1/run_training.py \
    --task-id "bloomz-$ARM" \
    --model bigscience/bloomz-560m \
    --dataset /train-fixture/train.jsonl \
    --dataset-type "$DATASET_TYPE" \
    --task-type InstructTextTask \
    --file-format json \
    --expected-repo-name "$ARM" \
    --hours-to-complete 2 \
    --baseline-stats /train-fixture/baseline-stats.json
done
```

Inventories are written at
`$RUNS/bloomz-control/_work/bloomz-checkpoint-inventory.json` and
`$RUNS/bloomz-full/_work/bloomz-checkpoint-inventory.json`. Each contains
exactly four artifact paths. If a container path is recorded, map the
`/app/checkpoints` prefix to `$RUNS` on the host before scoring.

## 5. Externally score and validate all eight dev artifacts

The following Bash block reads exactly four paths from each accepted inventory,
maps the container checkpoint prefix to the host, establishes one control
fingerprint anchor, scores the other seven artifacts against it, and validates
every artifact. Every score output directory and validation receipt is new.

```bash
DRIVER=/absolute/path/to/SN56-text/lanes/01-instruct-chat/runtime-smoke/local_artifact_score_driver.py
SCORES=/absolute/outside/repo/bloomz-scores
VALIDATIONS=/absolute/outside/repo/bloomz-validations
mkdir -p "$SCORES" "$VALIDATIONS"

read_inventory() {
  python -B - "$1" "$RUNS" <<'PY'
import json, sys
inventory, host_root = sys.argv[1:]
data = json.load(open(inventory, encoding="utf-8"))
assert data["status"] == "EXTERNAL_SCORE_READY"
assert len(data["checkpoints"]) == 4
for item in data["checkpoints"]:
    path = item["path"]
    prefix = "/app/checkpoints"
    assert path == prefix or path.startswith(prefix + "/")
    print(host_root + path[len(prefix):])
PY
}

mapfile -t CONTROL_ARTIFACTS < <(read_inventory "$RUNS/bloomz-control/_work/bloomz-checkpoint-inventory.json")
mapfile -t CANDIDATE_ARTIFACTS < <(read_inventory "$RUNS/bloomz-full/_work/bloomz-checkpoint-inventory.json")
[ "${#CONTROL_ARTIFACTS[@]}" -eq 4 ]
[ "${#CANDIDATE_ARTIFACTS[@]}" -eq 4 ]

score_selection() {
  local artifact=$1 role=$2 transport=$3 output=$4 fingerprint=$5
  local fingerprint_args=(--expected-fingerprint "$fingerprint")
  if [ "$fingerprint" = ANCHOR ]; then
    fingerprint_args=(--selection-fingerprint-anchor)
  fi
  run_stage "dev-score-$role-$(basename "$output")" 570 python -B "$EXP/score_external.py" \
    --artifact "$artifact" \
    --artifact-role "$role" \
    --expected-transport "$transport" \
    --base "$BASE" \
    --dev "$FIXTURE/dev.jsonl" \
    --confirmation "$FIXTURE/confirmation.jsonl" \
    --score-driver "$DRIVER" \
    --runtime-authority "$EVIDENCE/runtime-authority.json" \
    --output-dir "$output" \
    --gpu 0 \
    --phase selection \
    --timeout-seconds 540 \
    "${fingerprint_args[@]}"
}

score_selection "${CONTROL_ARTIFACTS[0]}" control peft_adapter "$SCORES/control-1" ANCHOR
DEV_FINGERPRINT=$(python -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["eval_set_fingerprint"])' "$SCORES/control-1/receipt.json")
for INDEX in 1 2 3; do
  NUMBER=$((INDEX + 1))
  score_selection "${CONTROL_ARTIFACTS[$INDEX]}" control peft_adapter "$SCORES/control-$NUMBER" "$DEV_FINGERPRINT"
done
for INDEX in 0 1 2 3; do
  NUMBER=$((INDEX + 1))
  score_selection "${CANDIDATE_ARTIFACTS[$INDEX]}" candidate full_model "$SCORES/candidate-$NUMBER" "$DEV_FINGERPRINT"
done

validate_one() {
  local artifact=$1 score_dir=$2 receipt_name=$3
  run_stage "validation-$receipt_name" 270 docker run --rm --pull never --network none --gpus 'device=0' \
    -e PYTHONPATH=/app \
    -v "$REPO:/app:ro" \
    -v "$BASE:/base:ro" \
    -v "$artifact:/artifact:ro" \
    -v "$score_dir:/score:ro" \
    -v "$EVIDENCE:/evidence:ro" \
    -v "$VALIDATIONS:/validations" \
    --entrypoint /workspace/axolotl-venv/bin/python \
    "$TRAIN_IMAGE" -B /app/experiments/20260831-bloomz-memory-fullft-v1/validate_artifact.py \
    --base-dir /base \
    --artifact /artifact \
    --receipt "/validations/$receipt_name.json" \
    --runtime-authority /evidence/runtime-authority.json \
    --external-score-receipt /score/receipt.json
}

for INDEX in 0 1 2 3; do
  NUMBER=$((INDEX + 1))
  validate_one "${CONTROL_ARTIFACTS[$INDEX]}" "$SCORES/control-$NUMBER" "control-$NUMBER"
  validate_one "${CANDIDATE_ARTIFACTS[$INDEX]}" "$SCORES/candidate-$NUMBER" "candidate-$NUMBER"
done
```

Validation rejects alternate tensor formats and every nested artifact file,
scans every authorized serialized floating or complex tensor for finiteness,
performs a fresh offline native Bloom load, and checks loaded state and logits.

## 6. Deterministic dev decision

This command passes exactly four score and validation receipts per arm:

```bash
DECISION=/absolute/outside/repo/bloomz-dev-decision.json

authority_science_check
set +e
python -B "$EXP/decide_external.py" select \
  --control-inventory "$RUNS/bloomz-control/_work/bloomz-checkpoint-inventory.json" \
  --candidate-inventory "$RUNS/bloomz-full/_work/bloomz-checkpoint-inventory.json" \
  --control-receipt "$SCORES/control-1/receipt.json" \
  --control-receipt "$SCORES/control-2/receipt.json" \
  --control-receipt "$SCORES/control-3/receipt.json" \
  --control-receipt "$SCORES/control-4/receipt.json" \
  --candidate-receipt "$SCORES/candidate-1/receipt.json" \
  --candidate-receipt "$SCORES/candidate-2/receipt.json" \
  --candidate-receipt "$SCORES/candidate-3/receipt.json" \
  --candidate-receipt "$SCORES/candidate-4/receipt.json" \
  --control-validation "$VALIDATIONS/control-1.json" \
  --control-validation "$VALIDATIONS/control-2.json" \
  --control-validation "$VALIDATIONS/control-3.json" \
  --control-validation "$VALIDATIONS/control-4.json" \
  --candidate-validation "$VALIDATIONS/candidate-1.json" \
  --candidate-validation "$VALIDATIONS/candidate-2.json" \
  --candidate-validation "$VALIDATIONS/candidate-3.json" \
  --candidate-validation "$VALIDATIONS/candidate-4.json" \
  --output "$DECISION"
DECISION_RC=$?
set -e
test "$DECISION_RC" -eq 0 -o "$DECISION_RC" -eq 3
```

Ties are broken by artifact tree SHA-256. The selected control and candidate
then face the owner paired gate over their ordered per-row loss vectors:
deterministic seed `20260808`, 10,000 paired bootstrap resamples, and 99%
one-sided lower bounds. Confirmation is authorized only when the candidate
win-rate lower bound is at least 0.55 and the control-minus-candidate mean-gap
lower bound is at least `max(0.01 nat, 1% * abs(control mean))`.

## 7. One paired confirmation

Only after the decision prints `AUTHORIZED_FOR_ONE_CONFIRMATION`, resolve the
two already-validated winners and score each exactly once. Confirmation accepts
no selection fingerprint or anchor; the authorization binds its fixture,
artifact role/tree/transport, base, source, and runtime. The same paired
bootstrap gate is required for the final confirmation verdict.

```bash
read_winner() {
  python -B - "$DECISION" "$1" <<'PY'
import json, sys
decision, role = sys.argv[1:]
data = json.load(open(decision, encoding="utf-8"))
assert data["status"] == "AUTHORIZED_FOR_ONE_CONFIRMATION"
score = json.load(open(data["selection"][role]["receipt_path"], encoding="utf-8"))
print(score["artifact"]["root"])
PY
}
score_confirmation() {
  local artifact=$1 role=$2 transport=$3 output=$4
  run_stage "confirmation-score-$role" 570 python -B "$EXP/score_external.py" \
    --artifact "$artifact" \
    --artifact-role "$role" \
    --expected-transport "$transport" \
    --base "$BASE" \
    --dev "$FIXTURE/dev.jsonl" \
    --confirmation "$FIXTURE/confirmation.jsonl" \
    --score-driver "$DRIVER" \
    --runtime-authority "$EVIDENCE/runtime-authority.json" \
    --output-dir "$output" \
    --gpu 0 \
    --phase confirmation \
    --timeout-seconds 540 \
    --decision-authorization "$DECISION"
}

DECISION_STATUS=$(python -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$DECISION")
FINAL_DECISION=$DECISION
if [ "$DECISION_STATUS" = AUTHORIZED_FOR_ONE_CONFIRMATION ]; then
  CONTROL_WINNER=$(read_winner control)
  CANDIDATE_WINNER=$(read_winner candidate)
  score_confirmation "$CONTROL_WINNER" control peft_adapter "$SCORES/control-confirmation"
  score_confirmation "$CANDIDATE_WINNER" candidate full_model "$SCORES/candidate-confirmation"
  authority_science_check
  CONFIRMATION_VERDICT=/absolute/outside/repo/bloomz-confirmation-verdict.json
  set +e
  python -B "$EXP/decide_external.py" confirm \
    --authorization "$DECISION" \
    --control-receipt "$SCORES/control-confirmation/receipt.json" \
    --candidate-receipt "$SCORES/candidate-confirmation/receipt.json" \
    --output "$CONFIRMATION_VERDICT"
  CONFIRMATION_RC=$?
  set -e
  test "$CONFIRMATION_RC" -eq 0 -o "$CONFIRMATION_RC" -eq 3
  FINAL_DECISION=$CONFIRMATION_VERDICT
else
  skip_stage 1140
fi

authority_science_check
python -B "$EXP/decide_external.py" complete \
  --decision "$FINAL_DECISION" \
  --output /absolute/outside/repo/bloomz-decision-complete.json
test "$PLANNED_STAGE_SECONDS" = 0
```

A successful local result is `LOCAL_PAIRED_CANDIDATE_WIN`. Any failed
identity, admission, schedule, artifact, score, validation, authorization, or
paired comparison is the concrete scientific hold `STOP_NO_SCIENCE`.
The lane ends at the authority-bound `DECISION_COMPLETE` receipt. It makes no
runtime-zero, process-zero, GPU-zero, evidence-custody, sync, or provider-state
claim. The unchanged CEO-owned trusted path owns the bound 5,400-second
custody/close reserve and the authority's `provider_deadline_epoch`; there is
no provider mutation here.
