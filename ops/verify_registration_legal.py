"""Fail-closed check for the SN56 registration LICENSE/NOTICE gate.

G.O.D clones a submission's default branch and compares its first recognized
LICENSE and NOTICE files with the validator repository after trimming trailing
whitespace on each line.  This utility reproduces that behavior locally while
also pinning the validator source that defined the check.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


EXPECTED_VALIDATOR_COMMIT = "a2b14db2a7f51c06b147a413c25f5901dd5a0247"
EXPECTED_VALIDATOR_TREE = "9c5664dd241e329dc6dd880fd7150890d99d77ab"
EXPECTED_VALIDATOR_CHECK_SHA256 = (
    "ba3cc564441cba705f4d389500963cf2e3126937d910a439fd0343078910f2f9"
)
EXPECTED_LICENSE_SHA256 = (
    "6ad6353ec71a92944b5e97adc783c01564044eee65e628a04efc8505db9506f8"
)
EXPECTED_NOTICE_SHA256 = (
    "3e316950fc1bc25c91a74614fa6ceb57e0ff14e20cf518fcc2628ee5f780469e"
)

LICENSE_NAMES = ("LICENSE.md", "LICENSE", "license.md", "license", "License.md", "License")
NOTICE_NAMES = ("NOTICE", "NOTICE.txt", "notice.txt", "Notice.txt", "notice", "Notice")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized(path: Path) -> str:
    return "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())


def _first(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise RuntimeError(f"none of {names!r} exists under {root}")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def verify(repo_root: Path, validator_root: Path) -> dict[str, str]:
    validator_commit = _git(validator_root, "rev-parse", "HEAD")
    validator_tree = _git(validator_root, "rev-parse", "HEAD^{tree}")
    if validator_commit != EXPECTED_VALIDATOR_COMMIT:
        raise RuntimeError(f"validator commit drift: {validator_commit}")
    if validator_tree != EXPECTED_VALIDATOR_TREE:
        raise RuntimeError(f"validator tree drift: {validator_tree}")

    check_source = validator_root / "validator/tournament/github_validation.py"
    if _sha256(check_source) != EXPECTED_VALIDATOR_CHECK_SHA256:
        raise RuntimeError("validator registration-check source hash drift")

    expected_license = _first(validator_root, LICENSE_NAMES)
    expected_notice = _first(validator_root, NOTICE_NAMES)
    actual_license = _first(repo_root, LICENSE_NAMES)
    actual_notice = _first(repo_root, NOTICE_NAMES)

    if _sha256(expected_license) != EXPECTED_LICENSE_SHA256:
        raise RuntimeError("pinned validator LICENSE byte hash drift")
    if _sha256(expected_notice) != EXPECTED_NOTICE_SHA256:
        raise RuntimeError("pinned validator NOTICE byte hash drift")
    if _normalized(actual_license) != _normalized(expected_license):
        raise RuntimeError(f"registration LICENSE mismatch: {actual_license}")
    if _normalized(actual_notice) != _normalized(expected_notice):
        raise RuntimeError(f"registration NOTICE mismatch: {actual_notice}")

    third_party = repo_root / "THIRD_PARTY_NOTICES.md"
    if not third_party.is_file() or "Axolotl" not in third_party.read_text(encoding="utf-8"):
        raise RuntimeError("Axolotl attribution was not preserved outside exact-match NOTICE")

    return {
        "validator_commit": validator_commit,
        "validator_tree": validator_tree,
        "validator_check_sha256": _sha256(check_source),
        "license_sha256": _sha256(actual_license),
        "notice_sha256": _sha256(actual_notice),
        "third_party_notices_sha256": _sha256(third_party),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--validator-root", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.repo_root.resolve(), args.validator_root.resolve())
    for key, value in result.items():
        print(f"{key}={value}")
    print("REGISTRATION_LEGAL_CHECK=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
