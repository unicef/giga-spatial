import gzip
import json
from pathlib import Path
import pytest

from gigaspatial.core.io import LocalDataStore
from gigaspatial.core.io.readers import (
    iter_field_from_gzipped_jsonl,
    read_field_from_gzipped_jsonl,
)


@pytest.fixture
def sample_gzipped_jsonl(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample.jsonl.gz"
    records = [
        {"id": 1, "name": "building_a", "geometry": {"type": "Point", "coordinates": [0, 0]}},
        {"id": 2, "name": "building_b", "geometry": {"type": "Point", "coordinates": [1, 1]}},
        {"id": 3, "geometry": {"type": "Point", "coordinates": [2, 2]}},  # missing 'name'
    ]

    with gzip.open(file_path, "wb") as f:
        for record in records:
            f.write(json.dumps(record).encode("utf-8") + b"\n")
            f.write(b"\n")  # Blank line to test empty line handling

    return file_path


def test_iter_field_basic(sample_gzipped_jsonl: Path):
    store = LocalDataStore()
    ids = list(iter_field_from_gzipped_jsonl(sample_gzipped_jsonl, field="id", data_store=store))
    assert ids == [1, 2, 3]


def test_iter_field_missing_ignore(sample_gzipped_jsonl: Path):
    names = list(
        iter_field_from_gzipped_jsonl(
            sample_gzipped_jsonl, field="name", ignore_missing=True
        )
    )
    assert names == ["building_a", "building_b"]


def test_iter_field_missing_default(sample_gzipped_jsonl: Path):
    names = list(
        iter_field_from_gzipped_jsonl(
            sample_gzipped_jsonl, field="name", default="unknown", ignore_missing=False
        )
    )
    assert names == ["building_a", "building_b", "unknown"]


def test_iter_field_alias(sample_gzipped_jsonl: Path):
    assert read_field_from_gzipped_jsonl is iter_field_from_gzipped_jsonl
