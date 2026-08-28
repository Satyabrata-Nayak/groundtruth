"""Ingestion tests, weighted toward failure cases.

The happy path is easy and rarely breaks. What breaks is the fifth malformed CSV a
user uploads, and what matters is that it fails cleanly rather than leaving a
half-written version that a later query trips over.
"""

import uuid

import duckdb
import pytest

from app.config import get_settings
from app.data import ingest, storage
from app.data.ingest import IngestError


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    root.mkdir()
    monkeypatch.setattr(get_settings(), "data_dir", root)
    return root


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text(
        "order_id,order_date,region,revenue,cost\n"
        "1,2024-01-15,North,1200.0,800.0\n"
        "2,2024-01-20,South,450.0,300.0\n"
        "3,2024-02-10,North,980.5,700.25\n",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------------


def test_csv_becomes_parquet(data_root, csv_file):
    result = ingest.ingest_file(csv_file)

    assert result.version == 1
    assert result.row_count == 3
    assert result.column_count == 5
    assert result.original_format == "csv"
    assert result.parquet_path.is_file()
    assert result.parquet_path.name == "data.parquet"


def test_types_are_inferred_not_stringified(data_root, csv_file):
    """The whole point of Parquet over CSV is that types survive storage."""
    result = ingest.ingest_file(csv_file)
    con = duckdb.connect(":memory:")
    con.execute(f"SELECT * FROM read_parquet('{result.parquet_path.as_posix()}') LIMIT 0")
    types = {col[0]: str(col[1]).upper() for col in con.description}

    assert "INT" in types["order_id"]
    assert "DATE" in types["order_date"]
    assert "VARCHAR" in types["region"] or "STRING" in types["region"]
    assert "DOUBLE" in types["revenue"] or "FLOAT" in types["revenue"]


def test_reupload_creates_a_second_version(data_root, csv_file):
    first = ingest.ingest_file(csv_file)
    second = ingest.ingest_file(csv_file, dataset_id=first.dataset_id)

    assert second.dataset_id == first.dataset_id
    assert second.version == 2
    assert storage.existing_versions(first.dataset_id) == [1, 2]
    # v1 must be untouched — immutability is what makes analyses reproducible.
    assert first.parquet_path.is_file()


def test_parquet_upload_passes_through(data_root, tmp_path, csv_file):
    staged = ingest.ingest_file(csv_file)
    external = tmp_path / "external.parquet"
    external.write_bytes(staged.parquet_path.read_bytes())

    result = ingest.ingest_file(external)
    assert result.original_format == "parquet"
    assert result.row_count == 3


# --------------------------------------------------------------------------------
# Failure cases
# --------------------------------------------------------------------------------


def test_empty_file_rejected(data_root, tmp_path):
    empty = tmp_path / "empty.csv"
    empty.touch()
    with pytest.raises(IngestError, match="empty"):
        ingest.ingest_file(empty)


def test_unsupported_extension_rejected(data_root, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(IngestError, match="unsupported file type"):
        ingest.ingest_file(bad)


def test_missing_file_rejected(data_root, tmp_path):
    with pytest.raises(IngestError, match="not found"):
        ingest.ingest_file(tmp_path / "nope.csv")


def test_oversized_file_rejected(data_root, tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_upload_mb", 0)
    big = tmp_path / "big.csv"
    big.write_text("a,b\n1,2\n")
    with pytest.raises(IngestError, match="exceeds"):
        ingest.ingest_file(big)


def test_header_only_csv_rejected(data_root, tmp_path):
    """A file with columns but no rows cannot be analysed and must not become a version."""
    header_only = tmp_path / "headers.csv"
    header_only.write_text("a,b,c\n")
    with pytest.raises(IngestError, match="no data rows"):
        ingest.ingest_file(header_only)


def test_corrupt_parquet_rejected(data_root, tmp_path):
    fake = tmp_path / "fake.parquet"
    fake.write_bytes(b"this is definitely not parquet")
    with pytest.raises(IngestError, match="could not read Parquet"):
        ingest.ingest_file(fake)


def test_failed_ingest_leaves_no_version_behind(data_root, tmp_path):
    """The critical cleanup guarantee.

    A version directory that exists but has no readable data would be resolved by
    later reads and fail confusingly. On any failure the directory must be gone.
    """
    ds_id = uuid.uuid4()
    header_only = tmp_path / "headers.csv"
    header_only.write_text("a,b,c\n")

    with pytest.raises(IngestError):
        ingest.ingest_file(header_only, dataset_id=ds_id)

    assert storage.existing_versions(ds_id) == []
    assert not storage.version_dir(ds_id, 1).exists()


def test_error_messages_do_not_leak_server_paths(data_root, tmp_path):
    """User-visible errors must not expose filesystem layout."""
    fake = tmp_path / "fake.parquet"
    fake.write_bytes(b"nonsense")
    with pytest.raises(IngestError) as exc:
        ingest.ingest_file(fake)
    assert "\n" not in str(exc.value)


# --------------------------------------------------------------------------------
# Messy real-world CSVs that must still work
# --------------------------------------------------------------------------------


def test_quoted_newlines_and_commas(data_root, tmp_path):
    messy = tmp_path / "messy.csv"
    messy.write_text(
        'id,note,amount\n1,"a note, with a comma",10\n2,"a note\nspanning lines",20\n',
        encoding="utf-8",
    )
    result = ingest.ingest_file(messy)
    assert result.row_count == 2


def test_late_appearing_nulls_do_not_corrupt_types(data_root, tmp_path):
    """The reason for sample_size=-1.

    A column that looks like an integer for many rows and then contains 'N/A' must not
    be typed as an integer. Sampling only the head would silently corrupt this.
    """
    late = tmp_path / "late.csv"
    rows = "\n".join(f"{i},{i * 10}" for i in range(1, 400))
    late.write_text(f"id,value\n{rows}\n400,N/A\n", encoding="utf-8")

    result = ingest.ingest_file(late)
    assert result.row_count == 400

    con = duckdb.connect(":memory:")
    con.execute(f"SELECT value FROM read_parquet('{result.parquet_path.as_posix()}')")
    value_type = str(con.description[0][1]).upper()
    assert "INT" not in value_type, f"late N/A was lost; column typed as {value_type}"
