#!/usr/bin/env python3
"""Create the immutable git/source/training-image authority for one GPU run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence

from forge.tuning import bloomz


SCHEMA_VERSION = "sn56.bloomz-runtime-authority.v1"
EXPERIMENT_PATH = "experiments/20260831-bloomz-memory-fullft-v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


class RuntimeInspectionError(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeInspectionError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def run_checked(argv: Sequence[str], *, timeout: int = 120) -> str:
    completed = subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    require(
        completed.returncode == 0,
        f"command failed ({completed.returncode}): {list(argv)!r}: "
        f"{(completed.stdout + completed.stderr)[-2000:]}",
    )
    return completed.stdout


def git_value(git: str, repository: Path, *arguments: str) -> str:
    value = run_checked((git, "-C", str(repository), *arguments), timeout=30).strip()
    require(bool(value), f"git returned an empty value for {arguments!r}")
    return value


def git_child_inventory(git: str, repository: Path, prefix: str) -> list[dict[str, Any]]:
    """Capture every tracked regular file beneath one committed source child."""
    raw = run_checked(
        (git, "-C", str(repository), "ls-files", "-s", "-z", "--", prefix),
        timeout=30,
    )
    tracked: list[tuple[str, str]] = []
    expected_blobs: dict[str, str] = {}
    for record in raw.split("\0"):
        if not record:
            continue
        metadata, separator, path = record.partition("\t")
        parts = metadata.split()
        require(separator == "\t" and len(parts) == 3, "invalid git index record")
        mode, blob, stage = parts
        require(stage == "0" and HEX40.fullmatch(blob) is not None, "unmerged source index")
        require(mode in {"100644", "100755"}, f"unsafe tracked source mode: {mode}")
        tracked.append((path, mode))
        expected_blobs[path] = blob
    inventory = bloomz.source_child_inventory(repository, prefix, tracked)
    require(
        all(item["git_blob_sha1"] == expected_blobs[item["path"]] for item in inventory),
        f"live {prefix} bytes differ from the clean Git index",
    )
    return inventory


def inspect_source(git: str, repository: Path) -> dict[str, Any]:
    top = Path(git_value(git, repository, "rev-parse", "--show-toplevel")).resolve()
    require(top == repository, f"repository root drift: {top} != {repository}")
    status_argv = (
        git,
        "-C",
        str(repository),
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    status = run_checked(status_argv, timeout=30)
    require(status == "", "runtime authority requires a clean committed checkout")
    commit = git_value(git, repository, "rev-parse", "HEAD")
    tree = git_value(git, repository, "rev-parse", "HEAD^{tree}")
    parent = git_value(git, repository, "rev-parse", "HEAD^")
    forge_tree = git_value(git, repository, "rev-parse", "HEAD:forge")
    experiment_tree = git_value(
        git,
        repository,
        "rev-parse",
        f"HEAD:{EXPERIMENT_PATH}",
    )
    dockerfile_blob = git_value(
        git,
        repository,
        "rev-parse",
        "HEAD:ops/docker/standalone-text-trainer.dockerfile",
    )
    for label, value in (
        ("commit", commit),
        ("tree", tree),
        ("parent", parent),
        ("forge child tree", forge_tree),
        ("experiment child tree", experiment_tree),
        ("Dockerfile blob", dockerfile_blob),
    ):
        require(HEX40.fullmatch(value) is not None, f"invalid {label}: {value!r}")
    inventory = bloomz.runtime_source_inventory(repository)
    forge_inventory = git_child_inventory(git, repository, "forge")
    experiment_inventory = git_child_inventory(
        git, repository, EXPERIMENT_PATH
    )
    return {
        "clean": True,
        "commit": commit,
        "tree": tree,
        "parent": parent,
        "forge_child_tree": forge_tree,
        "experiment_child_tree": experiment_tree,
        "dockerfile_blob": dockerfile_blob,
        "forge_inventory": forge_inventory,
        "experiment_inventory": experiment_inventory,
        "runtime_source_inventory": inventory,
        "runtime_source_inventory_sha256": bloomz.runtime_source_inventory_sha256(
            repository
        ),
        "git_status_argv": list(status_argv),
    }


def inspect_image(docker: str, image: str) -> dict[str, Any]:
    inspect_argv = (docker, "image", "inspect", image)
    raw = json.loads(run_checked(inspect_argv, timeout=60))
    require(isinstance(raw, list) and len(raw) == 1, "Docker inspect returned !=1 image")
    record = raw[0]
    require(isinstance(record, dict), "Docker image record is not an object")
    image_id = record.get("Id")
    require(
        isinstance(image_id, str) and SHA256_ID.fullmatch(image_id) is not None,
        "Docker image ID is not a SHA-256 identity",
    )
    require(record.get("Os") == "linux", "training image OS is not linux")
    require(record.get("Architecture") == "amd64", "training image is not amd64")
    repo_digests = sorted(record.get("RepoDigests") or [])
    require(image in repo_digests, "inspected image RepoDigests do not bind the reference")
    version_program = (
        "import json,sys,torch,transformers,peft,trl;"
        "print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,"
        "'transformers':transformers.__version__,'peft':peft.__version__,"
        "'trl':trl.__version__},sort_keys=True))"
    )
    version_argv = (
        docker,
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--entrypoint",
        "/workspace/axolotl-venv/bin/python",
        image,
        "-B",
        "-c",
        version_program,
    )
    versions = json.loads(run_checked(version_argv, timeout=180).strip())
    require(isinstance(versions, dict), "training runtime version probe is not an object")
    return {
        "reference": image,
        "image_id": image_id,
        "repo_digests": repo_digests,
        "inspect_record_sha256": hashlib.sha256(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "rootfs_layers": list((record.get("RootFS") or {}).get("Layers") or []),
        "os": record["Os"],
        "architecture": record["Architecture"],
        "created": record.get("Created"),
        "inspect_argv": list(inspect_argv),
        "version_probe_argv": list(version_argv),
        "versions": versions,
    }


def write_exclusive(path: Path, value: dict[str, Any]) -> str:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RuntimeInspectionError(f"refusing to overwrite authority: {path}") from exc
    return hashlib.sha256(payload).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docker", default="/usr/bin/docker")
    parser.add_argument("--git", default="/usr/bin/git")
    parser.add_argument("--training-image", default=bloomz.TRAINING_IMAGE)
    parser.add_argument("--provider-start-epoch", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repository = args.repo.expanduser().resolve(strict=True)
    require(repository.is_dir(), "repository is not a directory")
    require(args.training_image == bloomz.TRAINING_IMAGE, "training image reference drift")
    require(Path(args.docker).is_absolute(), "docker executable must be absolute")
    require(Path(args.git).is_absolute(), "git executable must be absolute")
    source = inspect_source(args.git, repository)
    training_image = inspect_image(args.docker, args.training_image)
    lease = bloomz.lease_authority(args.provider_start_epoch)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "experiment_config": bloomz.experiment_config(),
        "experiment_config_sha256": bloomz.experiment_config_sha256(),
        "source": source,
        "training_image": training_image,
        "lease": lease,
    }
    output = args.output.expanduser().resolve(strict=False)
    require(
        output != repository and repository not in output.parents,
        "runtime authority must be written outside the source checkout",
    )
    receipt_sha = write_exclusive(output, receipt)
    print(
        json.dumps(
            {"status": "PASS", "path": str(output), "sha256": receipt_sha},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOOMZ_RUNTIME_INSPECTION=FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
