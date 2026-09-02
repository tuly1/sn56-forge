from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import pytest

from forge.data.schema import InstructColumns
from forge.tuning import bloomz
from forge.tuning.memory import (
    estimate_sft_memory,
    infer_output_width,
    require_sft_admission,
)
from forge.tuning.plan import TrainPlan


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "20260831-bloomz-memory-fullft-v1"


def _runtime_authority() -> dict[str, object]:
    return {
        "source": {
            "parent": "0" * 40,
            "commit": "1" * 40,
            "tree": "2" * 40,
            "forge_child_tree": "3" * 40,
            "experiment_child_tree": "4" * 40,
            "runtime_source_inventory_sha256": bloomz.runtime_source_inventory_sha256(
                ROOT
            ),
        },
        "training_image": {
            "reference": bloomz.TRAINING_IMAGE,
            "image_id": "sha256:" + "5" * 64,
            "os": "linux",
            "architecture": "amd64",
            "repo_digests": [bloomz.TRAINING_IMAGE],
        },
        "lease": bloomz.lease_authority(4_102_444_800, 4_102_445_400),
    }


def _gpu_admission(
    *,
    arm: str,
    learning_rate: float,
    runtime_path: Path,
    runtime_authority: dict[str, object],
    runtime_sha256: str,
) -> dict[str, object]:
    strategy = "full" if arm == "full" else "lora"
    card_bytes = 80_000_000_000
    analytic = {}
    for purpose, sequence_length in (
        ("train", bloomz.SEQUENCE_LENGTH),
        ("selection", bloomz.SELECTION_SEQUENCE_LENGTH),
    ):
        analytic[purpose] = estimate_sft_memory(
            params_b=bloomz.MODEL_PARAMS_B,
            vocab_size=250_880,
            sequence_length=sequence_length,
            microbatch=bloomz.MICROBATCH,
            strategy=strategy,
            gradient_checkpointing=True,
            card_gb=card_bytes / 1_000_000_000.0,
        ).telemetry_fields()
    return {
        "schema_version": 1,
        "status": "PASS",
        "proof": bloomz.GPU_ADMISSION_PROOF,
        "analytic_estimate_role": "admission_support_only_not_hardware_proof",
        "arm": arm,
        "strategy": strategy,
        "learning_rate": learning_rate,
        "geometry": dict(bloomz.GPU_ADMISSION_GEOMETRY),
        "model": {
            "repo": bloomz.MODEL_REPO,
            "revision": bloomz.MODEL_REVISION,
            "config_sha256": bloomz.MODEL_CONFIG_SHA256,
            "weights_sha256": bloomz.MODEL_WEIGHTS_SHA256,
            "output_width": 250_880,
            "params_b": bloomz.MODEL_PARAMS_B,
        },
        "gpu": {
            "name": "NVIDIA H100 80GB HBM3",
            "card_bytes": card_bytes,
            "train_peak_allocated_bytes": 39_000_000_000,
            "train_peak_reserved_bytes": 40_000_000_000,
            "train_peak_reserved_ratio": 0.5,
            "selection_peak_allocated_bytes": 44_000_000_000,
            "selection_peak_reserved_bytes": 45_000_000_000,
            "selection_peak_reserved_ratio": 0.5625,
            "free_before_bytes": 75_000_000_000,
            "train_free_after_bytes": 40_000_000_000,
            "free_after_bytes": 35_000_000_000,
            "max_reserved_ratio": 0.7,
        },
        "loss": 1.25,
        "selection_loss": 1.5,
        "optimizer_state_tensor_count": 4,
        "optimizer_state_bytes": 8_000_000_000,
        "analytic_estimate": analytic,
        "runtime_authority_path": str(runtime_path.resolve()),
        "authority": bloomz.authority_fields(runtime_authority, runtime_sha256),
        "runtime": {
            "python": "3.11.13",
            "platform": "Linux-6.8-x86_64",
            "torch": "2.9.1+cu128",
            "transformers": "5.12.1",
            "peft": "0.19.1",
        },
    }


def _load_script(name: str):
    path = EXPERIMENT / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan(strategy: str) -> TrainPlan:
    return TrainPlan(
        lora_r=32 if strategy == "lora" else 0,
        lora_alpha=64 if strategy == "lora" else 0,
        lora_dropout=0.05 if strategy == "lora" else 0.0,
        learning_rate=3e-5,
        per_device_batch_size=4,
        grad_accum_steps=4,
        max_seq_len=4096,
        num_epochs=2,
        warmup_ratio=0.03,
        weight_decay=0.0,
        optimizer="adamw_torch_fused",
        lr_scheduler="cosine_with_min_lr",
        gradient_checkpointing=False,
        bf16=True,
        fp16=False,
        strategy=strategy,
    )


def test_wide_head_memory_uses_output_projection_and_rejects_old_geometry() -> None:
    head = SimpleNamespace(out_features=250_880, weight=SimpleNamespace(shape=(250_880, 8)))
    model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=250_880),
        get_output_embeddings=lambda: head,
    )
    class Tokenizer:
        def __len__(self) -> int:
            return 250_680

    tokenizer = Tokenizer()
    assert infer_output_width(model, tokenizer) == 250_880

    new = require_sft_admission(
        params_b=0.56,
        vocab_size=250_880,
        sequence_length=2048,
        microbatch=1,
        strategy="full",
        gradient_checkpointing=True,
        card_gb=80.0,
    )
    old = estimate_sft_memory(
        params_b=0.56,
        vocab_size=250_880,
        sequence_length=4096,
        microbatch=4,
        strategy="lora",
        gradient_checkpointing=False,
        card_gb=80.0,
    )
    assert new.admitted is True
    assert old.admitted is False
    doubled = estimate_sft_memory(
        params_b=0.56,
        vocab_size=250_880,
        sequence_length=4096,
        microbatch=1,
        strategy="full",
        gradient_checkpointing=True,
        card_gb=80.0,
    )
    assert doubled.logit_loss_gb == pytest.approx(2 * new.logit_loss_gb)


@pytest.mark.parametrize(
    "arm,strategy,lr",
    [("control", "lora", 1.5e-4), ("full", "full", bloomz.FULL_LR)],
)
def test_bloom_plan_freezes_identical_safe_geometry(arm: str, strategy: str, lr: float) -> None:
    routed = bloomz.apply_plan(
        _plan(strategy), SimpleNamespace(arm=arm, learning_rate=lr)
    )
    assert routed.per_device_batch_size == 1
    assert routed.grad_accum_steps == 16
    assert routed.per_device_batch_size * routed.grad_accum_steps == 16
    assert routed.max_seq_len == 2048
    assert routed.gradient_checkpointing is True
    assert routed.learning_rate == lr


def _write_training_fixture(root: Path) -> tuple[Path, dict[str, object]]:
    root.mkdir()
    splits = {}
    for name in ("train", "dev"):
        path = root / f"{name}.jsonl"
        path.write_text(
            '{"system":"","instruct":"explain confirmation bias",'
            '"output":"confirmation can be ordinary training text"}\n'
        )
        splits[name] = {
            "filename": path.name,
            "row_count": 1,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    dataset_type = root / "dataset-type.json"
    dataset_type.write_text("{}\n", encoding="utf-8")
    baseline = root / "baseline-stats.json"
    baseline.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": "sn56.bloomz-training-fixture.v1",
        "identities": {
            "dataset": {
                "repo": bloomz.DATASET_REPO,
                "revision": bloomz.DATASET_REVISION,
                "parquet_sha256": bloomz.DATASET_PARQUET_SHA256,
            },
            "model": {
                "repo": bloomz.MODEL_REPO,
                "revision": bloomz.MODEL_REVISION,
                "config_sha256": bloomz.MODEL_CONFIG_SHA256,
                "tokenizer_json_sha256": bloomz.MODEL_TOKENIZER_SHA256,
            },
        },
        "splits": splits,
        "schema": {
            "fields": ["system", "instruct", "output"],
            "dataset_type": {
                "filename": dataset_type.name,
                "sha256": hashlib.sha256(dataset_type.read_bytes()).hexdigest(),
            },
        },
        "artifacts": {
            "baseline_stats": {
                "filename": baseline.name,
                "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
            }
        },
    }
    path = root / "training-manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_training_fixture_physically_excludes_confirmation(tmp_path, monkeypatch) -> None:
    training_root = tmp_path / "training"
    manifest, contract = _write_training_fixture(training_root)
    monkeypatch.setattr(
        bloomz,
        "TRAINING_MANIFEST_SHA256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(bloomz, "training_manifest_contract", lambda: contract)
    monkeypatch.setenv("FORGE_BLOOMZ_EXPERIMENT_ARM", "full")
    monkeypatch.setenv("FORGE_ENABLE_EXPERIMENTAL_FULL_FT", "1")
    monkeypatch.setenv("FORGE_BLOOMZ_LR", "1e-4")
    monkeypatch.setenv("FORGE_BLOOMZ_PHASE", "candidate")
    monkeypatch.setenv(
        "FORGE_BLOOMZ_MAX_STEPS", str(bloomz.MATCHED_DECISION_STEPS)
    )
    monkeypatch.setenv("FORGE_BLOOMZ_TRAINING_MANIFEST", str(manifest))
    runtime_path = tmp_path / "runtime-authority.json"
    monkeypatch.setenv("FORGE_BLOOMZ_RUNTIME_AUTHORITY", str(runtime_path))
    runtime_authority = _runtime_authority()
    runtime_sha = "a" * 64
    monkeypatch.setattr(
        bloomz,
        "load_runtime_authority",
        lambda raw: (Path(raw), runtime_authority, runtime_sha),
    )
    admission_path = tmp_path / "gpu-admission.json"
    admission_path.write_text(
        json.dumps(
            _gpu_admission(
                arm="full",
                learning_rate=1e-4,
                runtime_path=runtime_path,
                runtime_authority=runtime_authority,
                runtime_sha256=runtime_sha,
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "FORGE_BLOOMZ_GPU_ADMISSION_RECEIPT", str(admission_path)
    )
    request = bloomz.request_from_environment()
    assert request is not None
    assert request.arm == "full" and request.learning_rate == 1e-4
    assert request.train_path.name == "train.jsonl"
    assert request.dev_path.name == "dev.jsonl"
    assert request.gpu_admission_path == admission_path.resolve()
    assert request.gpu_admission_sha256 == hashlib.sha256(
        admission_path.read_bytes()
    ).hexdigest()
    assert "confirmation" in request.train_path.read_text(encoding="utf-8")
    assert not any("confirmation" in path.name for path in training_root.iterdir())
    forbidden = training_root / "confirmation.jsonl"
    forbidden.write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(bloomz.BloomzExperimentError, match="physically contain only"):
        bloomz.request_from_environment()
    forbidden.unlink()
    monkeypatch.delenv("FORGE_BLOOMZ_GPU_ADMISSION_RECEIPT")
    with pytest.raises(bloomz.BloomzExperimentError, match="GPU_ADMISSION_RECEIPT"):
        bloomz.request_from_environment()
    monkeypatch.setenv(
        "FORGE_BLOOMZ_GPU_ADMISSION_RECEIPT", str(admission_path)
    )
    monkeypatch.setenv("FORGE_BLOOMZ_PHASE", "lr_probe")
    with pytest.raises(bloomz.BloomzExperimentError, match="PHASE=candidate"):
        bloomz.request_from_environment()
    monkeypatch.setenv("FORGE_BLOOMZ_PHASE", "candidate")
    monkeypatch.setenv("FORGE_BLOOMZ_LR", "7e-5")
    with pytest.raises(bloomz.BloomzExperimentError, match="LR is frozen"):
        bloomz.request_from_environment()


def test_full_arm_has_one_predeclared_matched_schedule() -> None:
    config = bloomz.experiment_config()
    assert config["arms"]["full"] == {
        "strategy": "full",
        "learning_rate": bloomz.FULL_LR,
    }
    assert "lr_probe_steps" not in config["decision"]
    assert config["decision"]["owner_paired_gate"] == {
        "seed": 20_260_808,
        "bootstrap_resamples": 10_000,
        "confidence": 0.99,
        "one_sided_tail": 0.01,
        "candidate_win_rate_lower_bound": 0.55,
        "mean_gap_lower_bound_nat_floor": 0.01,
        "mean_gap_lower_bound_control_fraction": 0.01,
        "mean_gap_direction": "control_minus_candidate",
    }
    assert config["model"]["tokenizer_config"] == {
        "bytes": 222,
        "sha256": bloomz.MODEL_TOKENIZER_CONFIG_SHA256,
    }
    assert config["model"]["special_tokens_map"] == {
        "bytes": 85,
        "sha256": bloomz.MODEL_SPECIAL_TOKENS_MAP_SHA256,
    }
    assert config["model"]["forbidden_tokenizer_files"] == [
        "added_tokens.json",
        "chat_template.jinja",
    ]


def test_total_lease_cap_arithmetic_is_exact() -> None:
    from decimal import Decimal

    budget = bloomz.lease_budget()
    stage_seconds = sum(
        item["count"] * item["max_each_seconds"]
        for item in bloomz.LEASE_STAGE_MAXIMA.values()
    )
    assert stage_seconds == budget["stage_seconds"] == 23_460
    assert budget["decision_reserve_seconds"] == 540
    assert stage_seconds + budget["decision_reserve_seconds"] == 24_000
    assert budget["science_window_seconds"] == 24_000
    assert budget["bootstrap_start_allowance_seconds"] == 1_200
    assert budget["ceo_custody_close_reserve_seconds"] == 5_400
    assert (
        budget["bootstrap_start_allowance_seconds"]
        + budget["science_window_seconds"]
        + budget["ceo_custody_close_reserve_seconds"]
        == budget["total_seconds"]
        == 30_600
    )
    assert Decimal(budget["hourly_rate_usd"]) * Decimal("8.5") == Decimal(
        budget["maximum_cost_usd"]
    )
    authority = bloomz.lease_authority(10_000, 11_200)
    assert authority["science_start_deadline_epoch"] == 11_200
    assert authority["science_started_epoch"] == 11_200
    assert authority["decision_deadline_epoch"] == 35_200
    assert authority["provider_deadline_epoch"] == 40_600
    with pytest.raises(bloomz.BloomzExperimentError, match="bootstrap allowance"):
        bloomz.lease_authority(10_000, 11_201)


def test_network_classifier_accepts_exact_timed_out_ipv4_default_deny() -> None:
    assert bloomz.classify_outbound_connect_result(
        family="IPv4",
        connect_ex=11,
        connected=False,
        elapsed_seconds=2.002144169000019,
        timeout_seconds=2.0,
    ) is True
    assert bloomz.classify_outbound_connect_result(
        family="IPv4",
        connect_ex=errno.ENETUNREACH,
        connected=False,
        elapsed_seconds=0.00001987,
        timeout_seconds=2.0,
    ) is True


def test_network_classifier_rejects_connections_and_ambiguous_eagain() -> None:
    assert bloomz.classify_outbound_connect_result(
        family="IPv4",
        connect_ex=0,
        connected=True,
        elapsed_seconds=0.01,
        timeout_seconds=2.0,
    ) is False
    assert bloomz.classify_outbound_connect_result(
        family="IPv4",
        connect_ex=errno.EAGAIN,
        connected=False,
        elapsed_seconds=1.999999,
        timeout_seconds=2.0,
    ) is False
    assert bloomz.classify_outbound_connect_result(
        family="IPv6",
        connect_ex=errno.EAGAIN,
        connected=False,
        elapsed_seconds=2.1,
        timeout_seconds=2.0,
    ) is False
    with pytest.raises(bloomz.BloomzExperimentError, match="inconsistent"):
        bloomz.classify_outbound_connect_result(
            family="IPv4",
            connect_ex=0,
            connected=False,
            elapsed_seconds=2.1,
            timeout_seconds=2.0,
        )


def test_shell_deadline_drift_is_rejected() -> None:
    lease = bloomz.lease_authority(10_000, 11_000)
    with pytest.raises(bloomz.BloomzExperimentError, match="shell decision deadline"):
        bloomz.require_science_stage(
            lease,
            stage_max_seconds=600,
            remaining_planned_seconds=600,
            now_epoch=11_001,
            claimed_decision_deadline_epoch=lease["decision_deadline_epoch"] + 1,
        )


def test_post_cutoff_science_stage_is_rejected() -> None:
    lease = bloomz.lease_authority(10_000, 11_000)
    with pytest.raises(bloomz.BloomzExperimentError, match="outside"):
        bloomz.require_science_stage(
            lease,
            stage_max_seconds=300,
            remaining_planned_seconds=300,
            now_epoch=lease["decision_deadline_epoch"],
        )


def test_task_contract_is_exact_standardized_schema() -> None:
    spec = SimpleNamespace(
        task_type="InstructTextTask",
        use_kl=False,
        model=bloomz.MODEL_REPO,
        instruct=InstructColumns(
            instruction="instruct", output="output", system="system"
        ),
    )
    bloomz.validate_task_contract(spec)
    spec.instruct = InstructColumns(instruction="question", output="output")
    with pytest.raises(bloomz.BloomzExperimentError, match="frozen"):
        bloomz.validate_task_contract(spec)


def test_model_identity_binds_all_tokenizer_files(tmp_path, monkeypatch) -> None:
    root = tmp_path / "base"
    root.mkdir()
    config = {
        "architectures": ["BloomForCausalLM"],
        "model_type": "bloom",
        "n_embed": 1024,
        "n_layer": 24,
        "num_attention_heads": 16,
        "seq_length": 2048,
        "vocab_size": 250880,
        "unk_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        "use_cache": True,
    }
    files = {
        "config.json": json.dumps(config).encode(),
        "tokenizer.json": b"tokenizer",
        "tokenizer_config.json": b"tokenizer-config",
        "special_tokens_map.json": b"special-tokens",
        "model.safetensors": b"weights",
    }
    constants = {
        "config.json": "MODEL_CONFIG_SHA256",
        "tokenizer.json": "MODEL_TOKENIZER_SHA256",
        "tokenizer_config.json": "MODEL_TOKENIZER_CONFIG_SHA256",
        "special_tokens_map.json": "MODEL_SPECIAL_TOKENS_MAP_SHA256",
        "model.safetensors": "MODEL_WEIGHTS_SHA256",
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
        monkeypatch.setattr(
            bloomz, constants[name], hashlib.sha256(content).hexdigest()
        )
    monkeypatch.setattr(
        bloomz, "MODEL_TOKENIZER_CONFIG_BYTES", len(files["tokenizer_config.json"])
    )
    monkeypatch.setattr(
        bloomz,
        "MODEL_SPECIAL_TOKENS_MAP_BYTES",
        len(files["special_tokens_map.json"]),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="bloom",
            architectures=["BloomForCausalLM"],
            vocab_size=250880,
        )
    )
    bloomz.validate_model_identity(model, root)
    (root / "chat_template.jinja").write_text("unsafe", encoding="utf-8")
    with pytest.raises(bloomz.BloomzExperimentError, match="unexpectedly contains"):
        bloomz.validate_model_identity(model, root)


def test_checkpoint_pool_retains_four_artifact_directories(tmp_path, monkeypatch) -> None:
    root = tmp_path / "bloomz-decision-checkpoints"
    pool = bloomz.BloomCheckpointPool(root=root)

    def fake_export(model, tokenizer, output, source):
        target = Path(output)
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}")

    monkeypatch.setattr(bloomz, "save_full_export", fake_export)
    callback = bloomz.make_checkpoint_callback(
        pool,
        tokenizer=object(),
        strategy="full",
        metadata_source_dir="/base",
    )
    control = SimpleNamespace()
    losses = [5.0, 4.0, 3.0, 2.0, 1.0]
    for step, loss in enumerate(losses, start=1):
        callback.on_evaluate(
            None,
            SimpleNamespace(global_step=step),
            control,
            metrics={"eval_loss": loss},
            model=object(),
        )
    assert pool.eval_count == 5
    assert len(pool.entries) == 4
    assert [entry.eval_loss for entry in pool.sorted_entries()] == [1.0, 2.0, 3.0, 4.0]
    assert len(list(root.iterdir())) == 4
    inventory = tmp_path / "inventory.json"
    runtime_path = tmp_path / "runtime-authority.json"
    runtime_authority = _runtime_authority()
    admission_path = tmp_path / "gpu-admission.json"
    admission = _gpu_admission(
        arm="full",
        learning_rate=1e-4,
        runtime_path=runtime_path,
        runtime_authority=runtime_authority,
        runtime_sha256="a" * 64,
    )
    assert bloomz.write_checkpoint_inventory(
        pool,
        inventory,
        strategy="full",
        phase="candidate",
        schedule_completed=True,
        planned_steps=5,
        final_step=5,
        runtime_authority_path=runtime_path,
        runtime_authority_sha256="a" * 64,
        runtime_authority=runtime_authority,
        gpu_admission_path=admission_path,
        gpu_admission_sha256="b" * 64,
        gpu_admission=admission,
    ) is True
    payload = json.loads(inventory.read_text())
    assert payload["status"] == "EXTERNAL_SCORE_READY"
    assert payload["gpu_admission"]["receipt_sha256"] == "b" * 64
    assert payload["gpu_admission"]["authority"] == payload["authority"]


def test_full_export_restores_metadata_inside_shared_atomic_call(tmp_path, monkeypatch) -> None:
    from forge.tasks import common

    source = tmp_path / "base"
    source.mkdir()
    (source / "config.json").write_text('{"use_cache":true}\n')
    (source / "tokenizer.json").write_text('{"original":true}\n')
    output = tmp_path / "out"

    class Model:
        def save_pretrained(self, path, **kwargs):
            target = Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "config.json").write_text('{"use_cache":false}\n')

    class Tokenizer:
        def save_pretrained(self, path, **kwargs):
            (Path(path) / "tokenizer.json").write_text('{"padding_side":"right"}\n')
            (Path(path) / "chat_template.jinja").write_text("invented")

    def fake_atomic(model, tokenizer, destination):
        staged = tmp_path / "staged"
        model.save_pretrained(staged)
        tokenizer.save_pretrained(staged)
        shutil.copytree(staged, destination)

    monkeypatch.setattr(common, "save_adapter", fake_atomic)
    bloomz.save_full_export(Model(), Tokenizer(), str(output), str(source))
    assert (output / "config.json").read_bytes() == (source / "config.json").read_bytes()
    assert (output / "tokenizer.json").read_bytes() == (source / "tokenizer.json").read_bytes()
    assert not (output / "adapter_config.json").exists()
    assert not (output / "chat_template.jinja").exists()


def test_gpu_probe_lr_contract_is_fail_closed() -> None:
    probe = _load_script("gpu_memory_probe.py")
    assert probe._frozen_lr("control", None) == 1.5e-4
    assert probe._frozen_lr("full", bloomz.FULL_LR) == bloomz.FULL_LR
    with pytest.raises(probe.ProbeError, match="outside"):
        probe._frozen_lr("full", 5e-5)
    with pytest.raises(probe.ProbeError, match="outside"):
        probe._frozen_lr("full", 7e-5)


def test_runtime_authority_binds_config_live_source_and_inspected_image(tmp_path) -> None:
    inventoried = {item["path"] for item in bloomz.runtime_source_inventory(ROOT)}
    assert {
        "forge/baseline.py",
        "forge/data/tokenize.py",
        "forge/tuning/bloomz.py",
        "ops/docker/standalone-text-trainer.dockerfile",
        f"{bloomz.EXPERIMENT_PATH}/run_training.py",
        f"{bloomz.EXPERIMENT_PATH}/score_external.py",
        f"{bloomz.EXPERIMENT_PATH}/validate_artifact.py",
    } <= inventoried
    payload = {
        "schema_version": "sn56.bloomz-runtime-authority.v2",
        "status": "PASS",
        "experiment_config_sha256": bloomz.experiment_config_sha256(),
        "source": {
            "clean": True,
            "parent": "0" * 40,
            "commit": "1" * 40,
            "tree": "2" * 40,
            "forge_child_tree": "3" * 40,
            "experiment_child_tree": "4" * 40,
            "forge_inventory": bloomz.source_child_inventory(ROOT, "forge"),
            "experiment_inventory": bloomz.source_child_inventory(
                ROOT, bloomz.EXPERIMENT_PATH
            ),
            "runtime_source_inventory_sha256": bloomz.runtime_source_inventory_sha256(
                ROOT
            ),
        },
        "training_image": {
            "reference": bloomz.TRAINING_IMAGE,
            "image_id": "sha256:" + "5" * 64,
            "os": "linux",
            "architecture": "amd64",
            "repo_digests": [bloomz.TRAINING_IMAGE],
        },
        "lease": bloomz.lease_authority(4_102_444_800, 4_102_445_400),
    }
    path = tmp_path / "runtime-authority.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    resolved, loaded, receipt_sha = bloomz.load_runtime_authority(
        path, source_root=ROOT
    )
    assert resolved == path.resolve()
    assert loaded == payload
    assert receipt_sha == hashlib.sha256(path.read_bytes()).hexdigest()

    payload["training_image"]["image_id"] = "latest"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(bloomz.BloomzExperimentError, match="image"):
        bloomz.load_runtime_authority(path, source_root=ROOT)

    payload["training_image"]["image_id"] = "sha256:" + "5" * 64
    payload["source"]["forge_inventory"][0]["sha256"] = "3" * 64
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(bloomz.BloomzExperimentError, match="source"):
        bloomz.load_runtime_authority(path, source_root=ROOT)


def test_tracked_source_inventory_ignores_only_python_cache(tmp_path: Path) -> None:
    source = tmp_path / "forge"
    source.mkdir()
    code = source / "route.py"
    code.write_text("VALUE = 7\n", encoding="utf-8")
    recorded = bloomz.source_child_inventory(tmp_path, "forge")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "route.cpython-311.pyc").write_bytes(b"ignored-bytecode")
    bloomz.verify_source_child_inventory(tmp_path, "forge", recorded)
    code.write_text("VALUE = 8\n", encoding="utf-8")
    with pytest.raises(bloomz.BloomzExperimentError, match="bytes drift"):
        bloomz.verify_source_child_inventory(tmp_path, "forge", recorded)


@pytest.mark.parametrize(
    "arm,learning_rate",
    (("control", bloomz.CONTROL_LR), ("full", 1e-4)),
)
def test_gpu_admission_is_exact_and_authority_bound(
    tmp_path, arm: str, learning_rate: float
) -> None:
    runtime_path = tmp_path / "runtime-authority.json"
    runtime_authority = _runtime_authority()
    runtime_sha = "a" * 64
    receipt = _gpu_admission(
        arm=arm,
        learning_rate=learning_rate,
        runtime_path=runtime_path,
        runtime_authority=runtime_authority,
        runtime_sha256=runtime_sha,
    )
    path = tmp_path / "gpu-admission.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    resolved, loaded, digest = bloomz.load_gpu_admission_receipt(
        path,
        arm=arm,
        learning_rate=learning_rate,
        runtime_authority_path=runtime_path,
        runtime_authority=runtime_authority,
        runtime_authority_sha256=runtime_sha,
    )
    assert resolved == path.resolve()
    assert loaded == receipt
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    receipt["learning_rate"] = 7e-5
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(bloomz.BloomzExperimentError, match="contract drift"):
        bloomz.load_gpu_admission_receipt(
            path,
            arm=arm,
            learning_rate=learning_rate,
            runtime_authority_path=runtime_path,
            runtime_authority=runtime_authority,
            runtime_authority_sha256=runtime_sha,
        )


def test_live_h100_must_match_admission_receipt(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda index: "NVIDIA H100 80GB HBM3"
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=80_000_000_000),
    )
    admission = {
        "gpu": {
            "name": "NVIDIA H100 80GB HBM3",
            "card_bytes": 80_000_000_000,
        }
    }
    bloomz.require_single_h100(
        n_gpus=1,
        per_gpu_gb=80.0,
        gpu_admission=admission,
    )
    admission["gpu"]["card_bytes"] = 79_000_000_000
    with pytest.raises(bloomz.BloomzExperimentError, match="differs"):
        bloomz.require_single_h100(
            n_gpus=1,
            per_gpu_gb=80.0,
            gpu_admission=admission,
        )


def test_checkpoint_tree_matches_external_scorer_inventory_hash(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    nested = checkpoint / "nested"
    nested.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (nested / "weights.bin").write_bytes(b"weights")
    files = []
    for item in sorted(
        candidate for candidate in checkpoint.rglob("*") if candidate.is_file()
    ):
        files.append(
            {
                "path": item.relative_to(checkpoint).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
            }
        )
    expected = hashlib.sha256(
        json.dumps(
            files,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert bloomz._tree_sha256(checkpoint) == expected

    (checkpoint / "unsafe-link").symlink_to(checkpoint / "config.json")
    with pytest.raises(bloomz.BloomzExperimentError, match="symlink"):
        bloomz._tree_sha256(checkpoint)


def test_fresh_artifact_scan_checks_parameters_and_buffers() -> None:
    import torch

    validator = _load_script("validate_artifact.py")

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.register_buffer("running", torch.tensor([3.0]))

    model = Tiny()
    summary = validator.finite_state_summary(model, chunk_elements=1)
    assert summary["all_tensors_finite"] is True
    assert summary["tensor_count"] == 2
    assert summary["total_numel"] == 3
    model.running[0] = torch.nan
    with pytest.raises(validator.ValidationError, match="nonfinite serialized tensor"):
        validator.finite_state_summary(model, chunk_elements=1)


def test_serialized_scan_rejects_extra_and_nonfinite_adapter_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from safetensors.torch import save_file

    validator = _load_script("validate_artifact.py")
    target = "transformer.h.0.self_attention.dense"
    monkeypatch.setattr(bloomz, "EXPECTED_LORA_TARGETS", frozenset({target}))
    expected = {
        f"base_model.model.{target}.lora_A.weight": torch.ones(2, 3),
        f"base_model.model.{target}.lora_B.weight": torch.ones(3, 2),
    }
    artifact = tmp_path / "adapter"
    base = tmp_path / "base"
    artifact.mkdir()
    base.mkdir()
    save_file(expected, artifact / "adapter_model.safetensors")
    summary = validator.serialized_state_summary(
        artifact,
        "peft_adapter",
        base,
        chunk_elements=1,
    )
    assert summary["all_tensors_finite"] is True
    assert summary["tensor_count"] == 2

    nested = artifact / "nested"
    nested.mkdir()
    save_file({"poison": torch.tensor([float("nan")])}, nested / "poison.safetensors")
    with pytest.raises(validator.ValidationError, match="unexpected nested artifact file"):
        validator.serialized_state_summary(artifact, "peft_adapter", base)
    (nested / "poison.safetensors").unlink()
    nested.rmdir()

    expected["unexpected.weight"] = torch.ones(1)
    save_file(expected, artifact / "adapter_model.safetensors")
    with pytest.raises(validator.ValidationError, match="key inventory drift"):
        validator.serialized_state_summary(artifact, "peft_adapter", base)

    del expected["unexpected.weight"]
    expected[f"base_model.model.{target}.lora_A.weight"][0, 0] = torch.nan
    save_file(expected, artifact / "adapter_model.safetensors")
    with pytest.raises(validator.ValidationError, match="nonfinite serialized tensor"):
        validator.serialized_state_summary(artifact, "peft_adapter", base, chunk_elements=1)


def test_experiment_launcher_propagates_training_failure_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_script("run_training.py")
    output = tmp_path / "visible-output"
    fake_spec = SimpleNamespace(
        task_id="control",
        task_type="InstructTextTask",
        model=bloomz.MODEL_REPO,
        file_format="json",
        output_dir=str(output),
    )
    request = SimpleNamespace(
        arm="control",
        phase="control",
        runtime_authority={
            "lease": bloomz.lease_authority(int(time.time()) - 2, int(time.time()) - 1)
        },
    )
    monkeypatch.setattr(launcher.bloomz, "request_from_environment", lambda: request)
    monkeypatch.setattr(launcher.bloomz, "validate_task_contract", lambda spec: None)
    monkeypatch.setattr(launcher.TaskSpec, "build", lambda **kwargs: fake_spec)

    def fail(_spec, _deadline):
        raise RuntimeError("training failed loudly")

    monkeypatch.setattr(launcher.bloomz, "run_matched_training", fail)
    with pytest.raises(RuntimeError, match="failed loudly"):
        launcher.main(
            [
                "--task-id", "control",
                "--model", bloomz.MODEL_REPO,
                "--dataset", "/fixture/train.jsonl",
                "--dataset-type", "{}",
                "--task-type", "InstructTextTask",
                "--file-format", "json",
                "--expected-repo-name", "control",
                "--hours-to-complete", "1",
            ]
        )
    assert not output.exists()
