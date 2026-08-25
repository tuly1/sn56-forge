from pathlib import Path

from ops.verify_registration_legal import _normalized


ROOT = Path(__file__).resolve().parents[1]


def test_exact_match_files_and_preserved_third_party_notice():
    expected_notice = """Gradients-on-Demand
Copyright 2025 Grads LLC

This product includes software developed by Grads LLC.

When using, redistributing, or modifying this software, you must include
attribution to Gradients.io (https://gradients.io) in any documentation,
user interfaces, or other materials that accompany the software or its
derivatives.

Gradients.io is owned by Grads LLC.
"""
    assert _normalized(ROOT / "NOTICE") == "\n".join(
        line.rstrip() for line in expected_notice.splitlines()
    )
    third_party = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "Axolotl" in third_party
    assert "0bda5a13e4d52ceec58104f44fabb7bd314f9c02" in third_party
