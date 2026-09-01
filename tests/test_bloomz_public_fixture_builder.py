from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "20260831-bloomz-memory-fullft-v1"
    / "build_fixture.py"
)
SPEC = importlib.util.spec_from_file_location("bloomz_public_fixture_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _raw(question: str, *, suffix: str = "") -> dict[str, str]:
    return {
        "system": "Be precise. ",
        "question": question,
        "chosen": f"answer {suffix}",
        "rejected": f"bad {suffix}",
    }


def _synthetic_rows() -> list[dict[str, str]]:
    long = " ".join(f"token{i}" for i in range(32))
    near = " ".join([*(f"token{i}" for i in range(31)), "replacement"])
    return [
        _raw("  Unicode   NORMALIZATION  ", suffix="exact-a"),
        _raw("unicode normalization", suffix="exact-b"),
        _raw(long, suffix="near-a"),
        _raw(near, suffix="near-b"),
        *[_raw(f"independent prompt number {index}", suffix=str(index)) for index in range(12)],
    ]


def test_standardization_is_exact_and_omits_rejected() -> None:
    standardized = builder.standardize_rows([_raw("question", suffix="one")])
    assert standardized == [
        {"system": "Be precise. ", "instruct": "question", "output": "answer one"}
    ]
    assert set(standardized[0]) == {"system", "instruct", "output"}

    broken = _raw("question")
    del broken["rejected"]
    with pytest.raises(builder.FixtureError, match="missing 'rejected'"):
        builder.standardize_rows([broken])


def test_group_split_is_deterministic_exact_and_nonleaking() -> None:
    rows = builder.standardize_rows(_synthetic_rows())
    grouping = builder.build_groups(rows)
    # NFKC/case/whitespace duplicate and a high-overlap 5-gram near duplicate.
    assert grouping.row_group_ids[0] == grouping.row_group_ids[1]
    assert grouping.row_group_ids[2] == grouping.row_group_ids[3]
    assert grouping.exact_edge_count == 1
    assert grouping.near_edge_count >= 1

    lengths = [(24 + (index % 5), 4 + (index % 3)) for index in range(len(rows))]
    first = builder.make_split_plan(grouping, lengths, confirmation_rows=3, dev_rows=4)
    second = builder.make_split_plan(grouping, lengths, confirmation_rows=3, dev_rows=4)
    assert first == second
    assert len(first["positions"]["confirmation"]) == 3
    assert len(first["positions"]["dev"]) == 4
    assert len(first["positions"]["train"]) == len(rows) - 7

    owner_by_group: dict[str, str] = {}
    for split_name, positions in first["positions"].items():
        for position in positions:
            group_id = grouping.row_group_ids[position]
            assert owner_by_group.setdefault(group_id, split_name) == split_name
    assert sorted(
        first["positions"]["train"]
        + first["positions"]["dev"]
        + first["positions"]["confirmation"]
    ) == list(range(len(rows)))


def test_near_duplicate_bridge_unions_every_qualifying_candidate() -> None:
    # A and B are below the Jaccard threshold with one another. C is within the
    # Jaccard/SimHash/LSH gate of both. The former first-match loop joined only
    # A-C and leaked B into a separate component; all qualifying edges must join.
    base = [f"word{index}" for index in range(150)]
    left = list(base)
    left[37:40] = ["lefta", "leftb", "leftc"]
    right = list(base)
    right[112:115] = ["righta", "rightb", "rightc"]
    raw = [
        _raw(" ".join(left), suffix="left"),
        _raw(" ".join(right), suffix="right"),
        _raw(" ".join(base), suffix="bridge"),
    ]
    grouping = builder.build_groups(builder.standardize_rows(raw))
    assert len(set(grouping.row_group_ids)) == 1
    assert grouping.near_edge_count == 2


def test_hamming_five_pair_with_changes_in_all_old_bands_is_not_missed() -> None:
    base = [f"word{index}" for index in range(127)]
    changed = list(base)
    changed[63] = "adversarial21"
    left = " ".join(base)
    right = " ".join(changed)
    left_shingles = builder.prompt_shingles(left)
    right_shingles = builder.prompt_shingles(right)
    xor = builder.simhash64(left_shingles) ^ builder.simhash64(right_shingles)
    assert xor.bit_count() == 5
    assert all((xor >> (16 * band)) & 0xFFFF for band in range(4))
    assert len(left_shingles & right_shingles) / len(left_shingles | right_shingles) == 0.921875

    grouping = builder.build_groups(
        builder.standardize_rows([_raw(left), _raw(right)])
    )
    assert grouping.row_group_ids[0] == grouping.row_group_ids[1]
    assert grouping.near_edge_count == 1


def test_candidate_bucket_has_no_arbitrary_recency_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    names = ["left", *(f"filler-{index}" for index in range(120)), "right"]
    shingle_map = {
        "left": frozenset(range(100)),
        "right": frozenset([*range(95), *range(200, 205)]),
        **{
            f"filler-{index}": frozenset({1000 + index})
            for index in range(120)
        },
    }
    hash_map = {
        shingle_map["left"]: 0,
        shingle_map["right"]: sum(1 << bit for bit in (0, 11, 22, 33, 44)),
        **{
            shingle_map[f"filler-{index}"]: (index + 1) << 1
            for index in range(120)
        },
    }
    monkeypatch.setattr(builder, "prompt_shingles", shingle_map.__getitem__)
    monkeypatch.setattr(builder, "simhash64", hash_map.__getitem__)

    grouping = builder.build_groups(
        builder.standardize_rows([_raw(name, suffix=name) for name in names])
    )
    assert grouping.row_group_ids[0] == grouping.row_group_ids[-1]


def test_materialized_splits_keep_source_order_and_standard_schema() -> None:
    rows = builder.standardize_rows(_synthetic_rows())
    grouping = builder.build_groups(rows)
    lengths = [(40 + index, 5) for index in range(len(rows))]
    plan = builder.make_split_plan(grouping, lengths, confirmation_rows=2, dev_rows=3)
    materialized = builder.materialize_splits(rows, plan)

    assert {name: len(value) for name, value in materialized.items()} == {
        "train": len(rows) - 5,
        "dev": 3,
        "confirmation": 2,
    }
    for split_name, split_rows in materialized.items():
        assert all(set(row) == {"system", "instruct", "output"} for row in split_rows)
        assert [rows.index(row) for row in split_rows] == plan["positions"][split_name]


def test_config_constraints_reject_context_or_padding_drift() -> None:
    config = dict(builder.MODEL_CONFIG_CONSTRAINTS)
    builder.validate_model_config(config)
    config["max_position_embeddings"] = 1024
    with pytest.raises(builder.FixtureError, match="must not invent"):
        builder.validate_model_config(config)

    tokenizer_config = dict(builder.TOKENIZER_CONFIG_CONSTRAINTS)
    special = {
        key: tokenizer_config[key]
        for key in ("unk_token", "bos_token", "eos_token", "pad_token")
    }
    builder.validate_tokenizer_config(tokenizer_config, special)
    tokenizer_config["padding_side"] = "right"
    with pytest.raises(builder.FixtureError, match="padding_side"):
        builder.validate_tokenizer_config(tokenizer_config, special)


def test_manifest_contract_constants_are_stable_and_rights_are_unresolved() -> None:
    assert builder.SCHEMA_VERSION == "sn56.bloomz-public-fixture.v1"
    assert set(builder.DATASET) >= {
        "repo",
        "revision",
        "parquet_sha256",
        "parquet_bytes",
        "row_count",
    }
    assert set(builder.MODEL) >= {"repo", "revision", "config_sha256", "tokenizer_json_sha256"}
    assert builder.RIGHTS["dataset_license_status"] == "UNRESOLVED"
    assert builder.EXPECTED_SPLITS["dev"]["index_sha256"] == (
        "4df2d1886f1b46e12fa1792c76fc6cc899f01cd34f066bb41b17b39c069b1088"
    )
    assert builder.EXPECTED_SPLITS["confirmation"]["index_sha256"] == (
        "288b501c1606ef12c1800e9f74778b9b55c45db0f071c26340cfecdf8a5db58c"
    )
    assert builder.EXPECTED_GROUP_STATISTICS == {
        "group_count": 39_758,
        "exact_edge_count": 0,
        "near_edge_count": 141,
    }
    assert builder.FIELD_TARGETS == {
        "authority": "NONLOCAL_FUTURE_OFFICIAL_ONLY",
        "runner_up_score": 1.3646068573,
        "top_score": 1.3348402977,
        "local_matched_ab_is_absolute_calibration": False,
    }


def test_generated_fixture_is_refused_inside_repository() -> None:
    repository_output = SCRIPT.parents[2] / "generated-fixture-must-not-land-here"
    with pytest.raises(builder.FixtureError, match="outside repository"):
        builder._assert_output_outside_repo(repository_output)
