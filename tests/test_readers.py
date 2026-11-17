import gzip
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
import pandas as pd
import pytest

from gigaspatial.core.io import LocalDataStore
from gigaspatial.core.io.readers import (
    iter_field_from_gzipped_jsonl,
    read_dataset,
    read_field_from_gzipped_jsonl,
    read_kmz,
)
from gigaspatial.core.io.writers import write_dataset


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


def _make_kmz(files: dict[str, bytes]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    buf.seek(0)
    return buf


_SAMPLE_MULTI_LAYER_KML = b"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <Folder>
    <name>Schools</name>
    <Placemark>
      <name>School A</name>
      <Point><coordinates>36.8,-1.2,0</coordinates></Point>
    </Placemark>
  </Folder>
  <Folder>
    <name>Zones</name>
    <Placemark>
      <name>Zone 1</name>
      <Polygon>
        <outerBoundaryIs><LinearRing><coordinates>
          36.0,-1.0,0 36.1,-1.0,0 36.1,-1.1,0 36.0,-1.1,0 36.0,-1.0,0
        </coordinates></LinearRing></outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Folder>
  <Folder>
    <name>EmptyFolder</name>
  </Folder>
</Document>
</kml>
"""


def test_read_kmz_merges_non_empty_layers():
    kmz = _make_kmz({"doc.kml": _SAMPLE_MULTI_LAYER_KML})
    gdf = read_kmz(kmz)

    assert list(gdf["Name"]) == ["School A", "Zone 1"]
    assert list(gdf["_kml_file"]) == ["doc.kml", "doc.kml"]
    assert list(gdf["_layer_name"]) == ["Schools", "Zones"]
    assert gdf.crs is not None


def test_read_kmz_specific_layer():
    kmz = _make_kmz({"doc.kml": _SAMPLE_MULTI_LAYER_KML})
    gdf = read_kmz(kmz, layer="Zones")

    assert list(gdf["Name"]) == ["Zone 1"]
    assert list(gdf["_layer_name"]) == ["Zones"]


def test_read_kmz_empty_archive_raises():
    kmz = _make_kmz({"readme.txt": b"no kml here"})
    with pytest.raises(ValueError, match="No KML files found"):
        read_kmz(kmz)


def test_readers_and_poi_generator_import_without_fiona():
    repo_root = str(Path(__file__).resolve().parents[1])
    code = (
        "import sys; "
        f"sys.path.insert(0, {repo_root!r}); "
        "sys.modules['fiona'] = None; "
        "from gigaspatial.core.io.readers import read_kmz; "
        "from gigaspatial import PoiViewGenerator; "
        "assert read_kmz is not None and PoiViewGenerator is not None"
    )
    subprocess.run([sys.executable, "-W", "ignore", "-c", code], check=True)


def test_read_write_xlsx_missing_openpyxl_raises_helpful_import_error(tmp_path: Path, monkeypatch):
    xlsx_path = tmp_path / "sample.xlsx"
    df = pd.DataFrame({"a": [1, 2]})
    write_dataset(df, path=str(xlsx_path), index=False)
    pd.testing.assert_frame_equal(read_dataset(str(xlsx_path)), df)

    monkeypatch.setitem(sys.modules, "openpyxl", None)
    with pytest.raises(ImportError, match="giga-spatial\\[excel\\]"):
        read_dataset(str(xlsx_path))
    with pytest.raises(ImportError, match="giga-spatial\\[excel\\]"):
        write_dataset(df, path=str(tmp_path / "out.xlsx"), index=False)


def test_read_xls_missing_xlrd_raises_helpful_import_error(tmp_path: Path, monkeypatch):
    xls_path = tmp_path / "sample.xls"
    xls_path.write_bytes(b"placeholder")

    monkeypatch.setitem(sys.modules, "xlrd", None)
    with pytest.raises(ImportError, match="giga-spatial\\[excel\\]"):
        read_dataset(str(xls_path))
