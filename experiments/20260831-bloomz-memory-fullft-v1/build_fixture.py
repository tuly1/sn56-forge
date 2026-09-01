#!/usr/bin/env python3
"""Freeze the public, group-safe BloomZ matched-A/B fixture.

This builder deliberately accepts only the immutable public parquet and the
immutable BloomZ tokenizer snapshot named below.  It creates a *new* matched
A/B authority; it does not reconstruct the inaccessible official tournament
row order or private evaluation data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Sequence
import unicodedata


SCHEMA_VERSION = "sn56.bloomz-public-fixture.v1"

DATASET = {
    "repo": "AlekseyKorshuk/evol-codealpaca-v1-dpo",
    "revision": "31c087a1492db443a3ace4247ef1880678b27aa4",
    "parquet_filename": "data/train-00000-of-00001.parquet",
    "parquet_sha256": "b7d98f92731ad075bd01c1088c59816f05cf0f49605856b7e3007482a419535a",
    "parquet_bytes": 76_890_709,
    "row_count": 39_882,
}

MODEL = {
    "repo": "bigscience/bloomz-560m",
    "revision": "a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05",
    "config_sha256": "ee4ce2e30325d9b0e2969748bc9945081be52e68a10f2aa66ce9bb33759c70bb",
    "config_bytes": 715,
    "tokenizer_json_sha256": "3fa39cd4b1500feb205bcce3b9703a4373414cafe4970e0657b413f7ddd2a9d3",
    "tokenizer_json_bytes": 14_500_438,
    "tokenizer_config_sha256": "ae85f7ec32efe4ba09f3914743b0187528eab0322fe90c4e077a9229d1de64a9",
    "tokenizer_config_bytes": 222,
    "special_tokens_map_sha256": "bb7068de1150661a10b55f9e4b12a0e77af8bf91f5e45e1b58afaf1d0e17f675",
    "special_tokens_map_bytes": 85,
}

MODEL_CONFIG_CONSTRAINTS = {
    "architectures": ["BloomForCausalLM"],
    "model_type": "bloom",
    "n_embed": 1024,
    "n_layer": 24,
    "num_attention_heads": 16,
    "seq_length": 2048,
    "vocab_size": 250_880,
    "unk_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 3,
    "use_cache": True,
}

TOKENIZER_CONFIG_CONSTRAINTS = {
    "tokenizer_class": "BloomTokenizerFast",
    "padding_side": "left",
    "unk_token": "<unk>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
}

SPECIAL_TOKEN_IDS = {
    "unk_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 3,
}

CONFIRMATION_ROWS = 512
DEV_ROWS = 1024
CONFIRMATION_SEED = "sn56-week12-bloomz-confirm-v1"
DEV_SEED = "sn56-week12-bloomz-dev-v1"

FIELD_TARGETS = {
    "authority": "NONLOCAL_FUTURE_OFFICIAL_ONLY",
    "runner_up_score": 1.3646068573,
    "top_score": 1.3348402977,
    "local_matched_ab_is_absolute_calibration": False,
}

FIELD_OBSERVED = {
    "evidence_label": "FIELD_OBSERVED",
    "official_task_id": "597dc05e-2593-4056-bcbd-16839cca5af1",
    "official_task_object_sha256": "356e2fd9bc57cf94dae71f7f9d36846e4d492552684feda0e36c0230b3ae948f",
    "official_task_object_path": "/opt/sn56-watcher/archive-20260831/objects/sha256/35/356e2fd9bc57cf94dae71f7f9d36846e4d492552684feda0e36c0230b3ae948f",
    "official_source_row_count": 38_882,
    "official_tokenized_row_count": 38_823,
    "official_train_row_count": 38_567,
    "official_validation_row_count": 256,
    "public_fixture_is_exact_official_rows_or_order": False,
    "private_train_and_test_objects_require_aws4_auth": True,
}

EXPECTED_SPLITS = {
    "train": {
        "row_count": 38_346,
        "sha256": "e4a0e2d83c9b39d0388931e868e04fdc0cc45288a29c18228ad13555ee1c52c0",
        "index_sha256": "3296793cc212761ed1583efcc64f0e1e8cb20f3f5a9bc6c5d82fce03ce998576",
        "group_ids_sha256": "b43920c2eb3a2844347bb2981746a00fbf4ca14fad6c1fcb5be77b7d50a0f4ab",
    },
    "dev": {
        "row_count": 1_024,
        "sha256": "f5548b1864a55c208f9f8061cb0e1d2471a6e58b976bb532ffdbb7a584bbfad6",
        "index_sha256": "4df2d1886f1b46e12fa1792c76fc6cc899f01cd34f066bb41b17b39c069b1088",
        "group_ids_sha256": "adea3e6f83bfadbf9a6d1a471f245bf9008fdff974b843f31ac2720618a5b13d",
    },
    "confirmation": {
        "row_count": 512,
        "sha256": "2b1a788ed12051688402d6709f75c7e1727d26711f4a52c9925d9eff5892c7ae",
        "index_sha256": "288b501c1606ef12c1800e9f74778b9b55c45db0f071c26340cfecdf8a5db58c",
        "group_ids_sha256": "2018a20b80e63297741b8619f4758d77a98e69f5ef7ab756bac87a620ad45349",
    },
}
EXPECTED_GROUP_STATISTICS = {
    "group_count": 39_758,
    "exact_edge_count": 0,
    "near_edge_count": 141,
}

DATASET_TYPE = {
    "field_system": "system",
    "field_instruction": "instruct",
    "field_output": "output",
    "system_format": "{system}",
    "no_input_format": "{instruction}",
}

RIGHTS = {
    "dataset_license_status": "UNRESOLVED",
    "notice": (
        "The pinned public dataset repository exposes no license or provenance. "
        "Use this fixture only for the scoped scientific matched A/B; do not "
        "redistribute it or claim permissive rights without separate clearance."
    ),
    "model_license": "BigScience BLOOM RAIL 1.0",
}

TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
GROUP_ALGORITHM = (
    "question_nfkc_casefold_whitespace_exact_plus_word_punctuation_5gram_"
    "blake2b64_simhash64_lsh6_pigeonhole_complete_hamming5_jaccard0.84_all_v3"
)


class FixtureError(RuntimeError):
    """The frozen fixture contract was not met."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def canonical_sha(value: Any, *, newline: bool = False) -> str:
    return hashlib.sha256(canonical_bytes(value, newline=newline)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixtureError(f"cannot parse {path.name}: {exc}") from exc
    require(isinstance(value, dict), f"{path.name} must contain a JSON object")
    return value


def _verify_file(path: Path, *, expected_sha: str, expected_bytes: int) -> None:
    require(path.is_file(), f"missing immutable tokenizer file: {path}")
    actual_bytes = path.stat().st_size
    require(
        actual_bytes == expected_bytes,
        f"{path.name} byte mismatch: {actual_bytes} != {expected_bytes}",
    )
    actual_sha = file_sha256(path)
    require(
        actual_sha == expected_sha,
        f"{path.name} SHA-256 mismatch: {actual_sha} != {expected_sha}",
    )


def validate_model_config(config: dict[str, Any]) -> None:
    for key, expected in MODEL_CONFIG_CONSTRAINTS.items():
        require(config.get(key) == expected, f"config {key!r} is not pinned value {expected!r}")
    require(
        "max_position_embeddings" not in config,
        "pinned Bloom config must not invent max_position_embeddings",
    )
    require("auto_map" not in config, "pinned Bloom config must not enable remote code")


def validate_tokenizer_config(
    tokenizer_config: dict[str, Any], special_tokens_map: dict[str, Any]
) -> None:
    for key, expected in TOKENIZER_CONFIG_CONSTRAINTS.items():
        require(
            tokenizer_config.get(key) == expected,
            f"tokenizer_config {key!r} is not pinned value {expected!r}",
        )
    for key in ("unk_token", "bos_token", "eos_token", "pad_token"):
        require(
            special_tokens_map.get(key) == TOKENIZER_CONFIG_CONSTRAINTS[key],
            f"special_tokens_map {key!r} does not match tokenizer_config",
        )


def load_verified_tokenizer(tokenizer_dir: Path) -> tuple[Any, dict[str, Any]]:
    tokenizer_dir = tokenizer_dir.resolve(strict=True)
    expected_files = {
        "config.json": (MODEL["config_sha256"], MODEL["config_bytes"]),
        "tokenizer.json": (
            MODEL["tokenizer_json_sha256"],
            MODEL["tokenizer_json_bytes"],
        ),
        "tokenizer_config.json": (
            MODEL["tokenizer_config_sha256"],
            MODEL["tokenizer_config_bytes"],
        ),
        "special_tokens_map.json": (
            MODEL["special_tokens_map_sha256"],
            MODEL["special_tokens_map_bytes"],
        ),
    }
    for filename, (expected_sha, expected_bytes) in expected_files.items():
        _verify_file(
            tokenizer_dir / filename,
            expected_sha=str(expected_sha),
            expected_bytes=int(expected_bytes),
        )

    config = _json_file(tokenizer_dir / "config.json")
    tokenizer_config = _json_file(tokenizer_dir / "tokenizer_config.json")
    special_tokens_map = _json_file(tokenizer_dir / "special_tokens_map.json")
    validate_model_config(config)
    validate_tokenizer_config(tokenizer_config, special_tokens_map)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, trust_remote_code=False
    )
    # Transformers 5 loads tokenizer.json through its generic
    # ``TokenizersBackend`` even though the immutable tokenizer_config still
    # declares BloomTokenizerFast.  The exact file hashes plus behavioural
    # checks below are the authority across the pinned 4.x/5.x loader boundary.
    require(
        type(tokenizer).__name__ in {"BloomTokenizerFast", "TokenizersBackend"},
        f"unexpected tokenizer class {type(tokenizer).__name__!r}",
    )
    require(tokenizer.vocab_size == 250_680, "Bloom tokenizer vocab_size must be 250680")
    require(len(tokenizer) == 250_680, "Bloom tokenizer must not contain added tokens")
    require(tokenizer.padding_side == "left", "Bloom tokenizer padding_side must remain left")
    for field, expected in SPECIAL_TOKEN_IDS.items():
        require(getattr(tokenizer, field) == expected, f"tokenizer {field} must be {expected}")

    return tokenizer, {
        "class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "length": len(tokenizer),
        "padding_side": tokenizer.padding_side,
        **SPECIAL_TOKEN_IDS,
    }


def load_verified_parquet(path: Path) -> list[dict[str, Any]]:
    path = path.resolve(strict=True)
    require(path.is_file(), f"parquet path is not a file: {path}")
    require(
        path.stat().st_size == DATASET["parquet_bytes"],
        f"parquet byte mismatch: {path.stat().st_size} != {DATASET['parquet_bytes']}",
    )
    actual_sha = file_sha256(path)
    require(
        actual_sha == DATASET["parquet_sha256"],
        f"parquet SHA-256 mismatch: {actual_sha} != {DATASET['parquet_sha256']}",
    )

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise FixtureError("pyarrow is required (it is supplied by datasets==4.8.4)") from exc

    parquet = pq.ParquetFile(path)
    expected_columns = ["system", "question", "chosen", "rejected"]
    require(
        parquet.schema_arrow.names == expected_columns,
        f"unexpected parquet columns/order: {parquet.schema_arrow.names}",
    )
    for field in parquet.schema_arrow:
        require(
            pa.types.is_string(field.type) or pa.types.is_large_string(field.type),
            f"parquet field {field.name!r} must be a string",
        )
    require(
        parquet.metadata.num_rows == DATASET["row_count"],
        f"parquet row mismatch: {parquet.metadata.num_rows} != {DATASET['row_count']}",
    )
    rows = parquet.read().to_pylist()
    require(len(rows) == DATASET["row_count"], "materialized parquet row count changed")
    return rows


def standardize_rows(raw_rows: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Map public DPO columns onto the official standardized instruct schema."""
    required = ("system", "question", "chosen", "rejected")
    standardized: list[dict[str, str]] = []
    for position, row in enumerate(raw_rows):
        require(isinstance(row, dict), f"source row {position} is not an object")
        for field in required:
            require(field in row, f"source row {position} is missing {field!r}")
            require(
                isinstance(row[field], str),
                f"source row {position} field {field!r} is not a string",
            )
        standardized.append(
            {
                "system": row["system"],
                "instruct": row["question"],
                "output": row["chosen"],
            }
        )
    return standardized


def normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def prompt_shingles(normalized: str) -> frozenset[int]:
    tokens = TOKEN_RE.findall(normalized)
    if not tokens:
        grams = [""]
    elif len(tokens) < 5:
        grams = ["\x1f".join(tokens)]
    else:
        grams = ["\x1f".join(tokens[index : index + 5]) for index in range(len(tokens) - 4)]
    return frozenset(
        int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")
        for gram in grams
    )


def simhash64(shingles: frozenset[int]) -> int:
    require(bool(shingles), "simhash requires at least one shingle")
    result = 0
    for bit in range(64):
        ones = sum((value >> bit) & 1 for value in shingles)
        if 2 * ones >= len(shingles):
            result |= 1 << bit
    return result


# Six disjoint blocks make the candidate index complete for Hamming distance
# <= 5: by the pigeonhole principle at least one block must be unchanged.  The
# unequal final widths cover all 64 bits exactly once.
SIMHASH_BANDS = ((0, 11), (11, 11), (22, 11), (33, 11), (44, 10), (54, 10))


def simhash_band_keys(value: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (band, (value >> offset) & ((1 << width) - 1))
        for band, (offset, width) in enumerate(SIMHASH_BANDS)
    )


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


@dataclass(frozen=True)
class Grouping:
    row_group_ids: tuple[str, ...]
    groups: dict[str, tuple[int, ...]]
    normalized_sha256: tuple[str, ...]
    exact_edge_count: int
    near_edge_count: int


def build_groups(rows: Sequence[dict[str, str]]) -> Grouping:
    """Build exact/near prompt components without inspecting completions."""
    uf = UnionFind(len(rows))
    normalized = [normalize_question(row["instruct"]) for row in rows]
    normalized_sha = [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in normalized]
    exact_first: dict[str, int] = {}
    shingles: list[frozenset[int]] = []
    simhashes: list[int] = []
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    exact_edges = 0
    near_edges = 0

    for position, value in enumerate(normalized):
        previous = exact_first.setdefault(value, position)
        if previous != position:
            uf.union(position, previous)
            exact_edges += 1

        row_shingles = prompt_shingles(value)
        row_simhash = simhash64(row_shingles)
        candidates: set[int] = set()
        for key in simhash_band_keys(row_simhash):
            candidates.update(buckets[key])
        for candidate in sorted(candidates):
            if normalized_sha[position] == normalized_sha[candidate]:
                continue
            if (row_simhash ^ simhashes[candidate]).bit_count() > 5:
                continue
            other = shingles[candidate]
            union_size = len(row_shingles | other)
            similarity = len(row_shingles & other) / union_size if union_size else 1.0
            if similarity >= 0.84:
                uf.union(position, candidate)
                near_edges += 1
        shingles.append(row_shingles)
        simhashes.append(row_simhash)
        for key in simhash_band_keys(row_simhash):
            buckets[key].append(position)

    components: dict[int, list[int]] = defaultdict(list)
    for position in range(len(rows)):
        components[uf.find(position)].append(position)
    row_group_ids = [""] * len(rows)
    groups: dict[str, tuple[int, ...]] = {}
    for positions in components.values():
        group_id = hashlib.sha256(
            canonical_bytes(sorted(normalized_sha[position] for position in positions))
        ).hexdigest()
        require(group_id not in groups, "canonical group-id collision")
        groups[group_id] = tuple(positions)
        for position in positions:
            row_group_ids[position] = group_id
    return Grouping(
        row_group_ids=tuple(row_group_ids),
        groups=groups,
        normalized_sha256=tuple(normalized_sha),
        exact_edge_count=exact_edges,
        near_edge_count=near_edges,
    )


def tokenize_lengths(
    rows: Sequence[dict[str, str]], tokenizer: Any, *, batch_size: int = 256
) -> list[tuple[int, int]]:
    """Measure the same separately-tokenized prompt/completion boundary as SFT."""
    require(batch_size > 0, "tokenizer batch_size must be positive")
    eos_extra = 1 if tokenizer.eos_token_id is not None else 0
    lengths: list[tuple[int, int]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [row["system"] + row["instruct"] for row in batch]
        completions = [row["output"] for row in batch]
        prompt_ids = tokenizer(
            prompts, add_special_tokens=True, truncation=False, padding=False
        )["input_ids"]
        completion_ids = tokenizer(
            completions, add_special_tokens=False, truncation=False, padding=False
        )["input_ids"]
        for prompt, completion in zip(prompt_ids, completion_ids, strict=True):
            completion_length = len(completion) + eos_extra
            lengths.append((len(prompt) + completion_length, completion_length))
    require(len(lengths) == len(rows), "token length projection changed row count")
    return lengths


def _upper_median(values: Iterable[Any]) -> Any:
    ordered = sorted(values)
    require(bool(ordered), "cannot take median of an empty group")
    return ordered[len(ordered) // 2]


def _group_metadata(
    group_ids: Iterable[str],
    grouping: Grouping,
    lengths: Sequence[tuple[int, int]],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for group_id in group_ids:
        positions = grouping.groups[group_id]
        totals = [lengths[position][0] for position in positions]
        ratios = [lengths[position][1] / lengths[position][0] for position in positions]
        total_median = int(_upper_median(totals))
        ratio_median = float(_upper_median(ratios))
        stratum = (
            min(15, max(0, total_median.bit_length() - 1)),
            min(3, max(0, int(ratio_median * 4))),
        )
        metadata[group_id] = {
            "row_count": len(positions),
            "median_total_tokens": total_median,
            "median_completion_ratio": ratio_median,
            "stratum": stratum,
        }
    return metadata


def select_exact_groups(
    eligible_group_ids: Iterable[str],
    grouping: Grouping,
    lengths: Sequence[tuple[int, int]],
    *,
    target_rows: int,
    seed: str,
) -> tuple[set[str], dict[str, Any]]:
    """Interleave deterministic length strata and take whole groups to capacity."""
    require(target_rows >= 0, "target row count must be nonnegative")
    eligible = sorted(set(eligible_group_ids))
    metadata = _group_metadata(eligible, grouping, lengths)
    strata: dict[tuple[int, int], list[str]] = defaultdict(list)
    for group_id, item in metadata.items():
        strata[item["stratum"]].append(group_id)

    ranked: list[tuple[Fraction, tuple[int, int], bytes, str]] = []
    for stratum, group_ids in sorted(strata.items()):
        def digest(group_id: str) -> bytes:
            return hashlib.blake2b(
                f"{seed}:{stratum[0]}:{stratum[1]}:{group_id}".encode("utf-8"),
                digest_size=16,
            ).digest()

        ordered = sorted(group_ids, key=lambda group_id: (digest(group_id), group_id))
        for rank, group_id in enumerate(ordered):
            ranked.append(
                (Fraction(2 * rank + 1, 2 * len(ordered)), stratum, digest(group_id), group_id)
            )
    ranked.sort()

    selected: set[str] = set()
    selected_rows = 0
    for _quantile, _stratum, _digest, group_id in ranked:
        group_size = metadata[group_id]["row_count"]
        if selected_rows + group_size > target_rows:
            continue
        selected.add(group_id)
        selected_rows += group_size
        if selected_rows == target_rows:
            break
    require(
        selected_rows == target_rows,
        f"cannot reach exact whole-group capacity {target_rows}; reached {selected_rows}",
    )
    selected_strata: dict[str, int] = defaultdict(int)
    for group_id in selected:
        stratum = metadata[group_id]["stratum"]
        selected_strata[f"{stratum[0]}:{stratum[1]}"] += metadata[group_id]["row_count"]
    return selected, {
        "seed": seed,
        "target_rows": target_rows,
        "selected_group_count": len(selected),
        "selected_strata_rows": dict(sorted(selected_strata.items())),
    }


def make_split_plan(
    grouping: Grouping,
    lengths: Sequence[tuple[int, int]],
    *,
    confirmation_rows: int = CONFIRMATION_ROWS,
    dev_rows: int = DEV_ROWS,
) -> dict[str, Any]:
    require(len(grouping.row_group_ids) == len(lengths), "group/length row count mismatch")
    all_groups = set(grouping.groups)
    confirmation_groups, confirmation_diagnostics = select_exact_groups(
        all_groups,
        grouping,
        lengths,
        target_rows=confirmation_rows,
        seed=CONFIRMATION_SEED,
    )
    dev_groups, dev_diagnostics = select_exact_groups(
        all_groups - confirmation_groups,
        grouping,
        lengths,
        target_rows=dev_rows,
        seed=DEV_SEED,
    )
    train_groups = all_groups - confirmation_groups - dev_groups
    group_sets = {
        "train": train_groups,
        "dev": dev_groups,
        "confirmation": confirmation_groups,
    }
    positions = {
        name: sorted(position for group_id in group_ids for position in grouping.groups[group_id])
        for name, group_ids in group_sets.items()
    }
    require(len(positions["confirmation"]) == confirmation_rows, "confirmation count drift")
    require(len(positions["dev"]) == dev_rows, "dev count drift")
    all_positions = positions["train"] + positions["dev"] + positions["confirmation"]
    require(sorted(all_positions) == list(range(len(lengths))), "split coverage/uniqueness failure")
    return {
        "positions": positions,
        "group_ids": {name: sorted(group_ids) for name, group_ids in group_sets.items()},
        "diagnostics": {
            "confirmation": confirmation_diagnostics,
            "dev": dev_diagnostics,
            "reservation_order": ["confirmation", "dev", "train_remainder"],
        },
    }


def materialize_splits(
    rows: Sequence[dict[str, str]], split_plan: dict[str, Any]
) -> dict[str, list[dict[str, str]]]:
    return {
        name: [rows[position] for position in split_plan["positions"][name]]
        for name in ("train", "dev", "confirmation")
    }


def _jsonl_sha(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_bytes(row, newline=True))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> str:
    path.write_bytes(canonical_bytes(value, newline=True))
    return file_sha256(path)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            line = canonical_bytes(row, newline=True)
            handle.write(line)
            digest.update(line)
    return digest.hexdigest()


def _quantile(values: Sequence[int], quantile: float) -> int:
    require(bool(values), "cannot summarize empty token lengths")
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return int(ordered[rank])


def make_baseline_stats(
    train_positions: Sequence[int],
    lengths: Sequence[tuple[int, int]],
    grouping: Grouping,
) -> dict[str, Any]:
    totals = [lengths[position][0] for position in train_positions]
    train_group_count = len({grouping.row_group_ids[position] for position in train_positions})
    redundant_rows = len(train_positions) - train_group_count
    return {
        "task_type": "instruct",
        "dataset": {
            "num_records": len(train_positions),
            "total_tokens": sum(totals),
            "near_duplicate_rate": redundant_rows / len(train_positions),
            "seq_length_distribution": {
                "p50": _quantile(totals, 0.50),
                "p95": _quantile(totals, 0.95),
                "p99": _quantile(totals, 0.99),
                "max": max(totals),
            },
        },
        "weights": {},
        "training": {},
    }


def _assert_output_outside_repo(output_dir: Path) -> Path:
    output = output_dir.expanduser().resolve(strict=False)
    repo_root = Path(__file__).resolve().parents[2]
    try:
        output.relative_to(repo_root)
    except ValueError:
        return output
    raise FixtureError(f"generated fixture output must be outside repository: {output}")


def write_fixture(
    output_dir: Path,
    *,
    raw_rows: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, str]],
    lengths: Sequence[tuple[int, int]],
    grouping: Grouping,
    split_plan: dict[str, Any],
    tokenizer_facts: dict[str, Any],
) -> tuple[Path, str]:
    output = _assert_output_outside_repo(output_dir)
    require(not output.exists(), f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        split_rows = materialize_splits(rows, split_plan)
        split_contract: dict[str, dict[str, Any]] = {}
        for name in ("train", "dev", "confirmation"):
            filename = f"{name}.jsonl"
            index_filename = f"{name}.indices.json"
            group_ids_filename = f"{name}.groups.json"
            row_sha = _write_jsonl(temporary / filename, split_rows[name])
            index_sha = _write_json(
                temporary / index_filename, split_plan["positions"][name]
            )
            group_ids_sha = _write_json(
                temporary / group_ids_filename, split_plan["group_ids"][name]
            )
            split_contract[name] = {
                "filename": filename,
                "file_format": "jsonl",
                "row_count": len(split_rows[name]),
                "sha256": row_sha,
                "index_filename": index_filename,
                "index_sha256": index_sha,
                "group_ids_filename": group_ids_filename,
                "group_ids_sha256": group_ids_sha,
            }
        split_contract["confirmation"]["access_policy"] = (
            "UNTOUCHED_UNTIL_DEV_MEMBERSHIP_IS_IMMUTABLY_RECORDED"
        )

        group_manifest = {
            "algorithm": GROUP_ALGORITHM,
            "row_group_ids": list(grouping.row_group_ids),
            "groups": [
                {
                    "group_id": group_id,
                    "source_indices": list(grouping.groups[group_id]),
                    "normalized_question_sha256": [
                        grouping.normalized_sha256[position]
                        for position in grouping.groups[group_id]
                    ],
                }
                for group_id in sorted(grouping.groups)
            ],
            "exact_edge_count": grouping.exact_edge_count,
            "near_edge_count": grouping.near_edge_count,
        }
        groups_sha = _write_json(temporary / "groups.json", group_manifest)
        token_lengths_sha = _write_json(
            temporary / "token-lengths.json",
            [
                {"source_index": position, "total": total, "completion": completion}
                for position, (total, completion) in enumerate(lengths)
            ],
        )
        dataset_type_sha = _write_json(temporary / "dataset-type.json", DATASET_TYPE)
        baseline = make_baseline_stats(
            split_plan["positions"]["train"], lengths, grouping
        )
        baseline_sha = _write_json(temporary / "baseline-stats.json", baseline)

        try:
            transformers_version = importlib.metadata.version("transformers")
            datasets_version = importlib.metadata.version("datasets")
            pyarrow_version = importlib.metadata.version("pyarrow")
        except importlib.metadata.PackageNotFoundError as exc:
            raise FixtureError(f"missing pinned builder dependency: {exc}") from exc

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "identities": {
                "dataset": dict(DATASET),
                "model": {
                    **MODEL,
                    "config_constraints": MODEL_CONFIG_CONSTRAINTS,
                    "tokenizer_config_constraints": TOKENIZER_CONFIG_CONSTRAINTS,
                    "verified_tokenizer": tokenizer_facts,
                },
            },
            "splits": split_contract,
            "field_targets": FIELD_TARGETS,
            "field_observed": FIELD_OBSERVED,
            "authority": {
                "kind": "NEW_PUBLIC_MATCHED_AB",
                "claim_exact_official_week11_rows_evaluator_or_absolute_calibration": False,
                "reason": (
                    "Official private rows/order and deployed evaluator digest are not exposed; "
                    "this fixture binds the immutable public revision for a matched comparison."
                ),
            },
            "rights": RIGHTS,
            "schema": {
                "fields": ["system", "instruct", "output"],
                "mapping": {
                    "system": "system",
                    "question": "instruct",
                    "chosen": "output",
                    "rejected": "omitted",
                },
                "dataset_type_filename": "dataset-type.json",
                "dataset_type_sha256": dataset_type_sha,
            },
            "algorithms": {
                "grouping": GROUP_ALGORITHM,
                "stratification": (
                    "whole_group_upper_median_total_bitlength_cap15_x_"
                    "upper_median_completion_ratio_quartile_v1"
                ),
                "reservation": split_plan["diagnostics"],
                "prompt_assembly": "system_plus_instruct_direct_then_separate_output_plus_eos_v1",
            },
            "source": {
                "raw_canonical_jsonl_sha256": _jsonl_sha(raw_rows),
                "standardized_canonical_jsonl_sha256": _jsonl_sha(rows),
                "row_count": len(rows),
            },
            "statistics": {
                "group_count": len(grouping.groups),
                "exact_edge_count": grouping.exact_edge_count,
                "near_edge_count": grouping.near_edge_count,
            },
            "artifacts": {
                "groups": {"filename": "groups.json", "sha256": groups_sha},
                "token_lengths": {
                    "filename": "token-lengths.json",
                    "sha256": token_lengths_sha,
                },
                "baseline_stats": {
                    "filename": "baseline-stats.json",
                    "sha256": baseline_sha,
                },
            },
            "runtime": {
                "transformers": transformers_version,
                "datasets": datasets_version,
                "pyarrow": pyarrow_version,
            },
        }
        manifest_sha = _write_json(temporary / "manifest.json", manifest)
        (temporary / "manifest.sha256").write_text(
            f"{manifest_sha}  manifest.json\n", encoding="ascii"
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output, manifest_sha


def build(parquet_path: Path, tokenizer_dir: Path, output_dir: Path) -> tuple[Path, str]:
    raw_rows = load_verified_parquet(parquet_path)
    rows = standardize_rows(raw_rows)
    require(len(rows) == DATASET["row_count"], "standardization changed public row count")
    tokenizer, tokenizer_facts = load_verified_tokenizer(tokenizer_dir)
    lengths = tokenize_lengths(rows, tokenizer)
    for position, (total, completion) in enumerate(lengths):
        require(total > 0, f"row {position} has no tokens")
        require(0 < completion <= total, f"row {position} has invalid completion length")
    grouping = build_groups(rows)
    actual_group_statistics = {
        "group_count": len(grouping.groups),
        "exact_edge_count": grouping.exact_edge_count,
        "near_edge_count": grouping.near_edge_count,
    }
    require(
        actual_group_statistics == EXPECTED_GROUP_STATISTICS,
        f"frozen group statistics drift: {actual_group_statistics!r}",
    )
    split_plan = make_split_plan(grouping, lengths)
    split_rows = materialize_splits(rows, split_plan)
    for name, expected in EXPECTED_SPLITS.items():
        actual = {
            "row_count": len(split_rows[name]),
            "sha256": _jsonl_sha(split_rows[name]),
            "index_sha256": canonical_sha(
                split_plan["positions"][name], newline=True
            ),
            "group_ids_sha256": canonical_sha(
                split_plan["group_ids"][name], newline=True
            ),
        }
        require(
            actual == expected,
            f"frozen {name} split drift: {actual!r} != {expected!r}",
        )
    return write_fixture(
        output_dir,
        raw_rows=raw_rows,
        rows=rows,
        lengths=lengths,
        grouping=grouping,
        split_plan=split_plan,
        tokenizer_facts=tokenizer_facts,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True, help="exact pinned public parquet")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        required=True,
        help="local exact BloomZ tokenizer snapshot (config files included)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new output directory outside this repository",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output, manifest_sha = build(args.parquet, args.tokenizer_dir, args.output_dir)
    print(
        json.dumps(
            {
                "status": "FROZEN_PUBLIC_MATCHED_AB_READY",
                "output_dir": str(output),
                "manifest_sha256": manifest_sha,
                "train_rows": DATASET["row_count"] - CONFIRMATION_ROWS - DEV_ROWS,
                "dev_rows": DEV_ROWS,
                "confirmation_rows": CONFIRMATION_ROWS,
                "dataset_rights": "UNRESOLVED",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
