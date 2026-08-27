"""End-to-end dataset lifecycle against real Postgres.

This is M2's exit criterion as a test: upload a dataset, get a correct profile back,
query it — with no LLM anywhere in the path.
"""

from __future__ import annotations

import uuid

import pytest

from app.data import sandbox, service
from app.data.ingest import IngestError
from app.db.models import ColumnProfile, Dataset, DatasetVersion
from app.db.session import session_scope

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------------
# The exit criterion
# --------------------------------------------------------------------------------


def test_upload_profile_and_query_end_to_end(db, data_root, sales_csv):
    created = service.create_dataset(sales_csv, name="Sales")

    assert created.row_count == 4
    assert created.column_count == 6
    assert created.version == 1

    # the profile survives the process that computed it
    with session_scope() as session:
        profile = service.get_stored_profile(session, created.dataset_id)

    assert profile is not None
    assert profile.row_count == 4
    revenue = next(c for c in profile.columns if c.name == "revenue")
    assert revenue.null_count == 1
    assert revenue.semantic_type == "numeric"
    assert next(c for c in profile.columns if c.name == "flag").is_constant is True

    # and the data itself is queryable through the sandbox
    result = sandbox.execute_sql(
        created.dataset_id, created.version,
        f"SELECT region, SUM(revenue) AS r FROM {sandbox.TABLE_NAME} "
        "GROUP BY region ORDER BY r DESC",
    )
    assert result.as_dicts()[0] == {"region": "North", "r": pytest.approx(2180.0)}


def test_stored_profile_matches_recomputed_profile(db, data_root, sales_csv):
    """Persisting must not distort the numbers. If the round-trip through Postgres
    changed a value, every stored profile would be quietly wrong."""
    from app.data.profile import profile_parquet
    from app.data.storage import resolve_existing_parquet

    created = service.create_dataset(sales_csv)
    fresh = profile_parquet(resolve_existing_parquet(created.dataset_id, 1))

    with session_scope() as session:
        stored = service.get_stored_profile(session, created.dataset_id)

    assert stored.row_count == fresh.row_count
    assert stored.duplicate_row_count == fresh.duplicate_row_count
    for a, b in zip(stored.columns, fresh.columns, strict=True):
        assert a.name == b.name
        assert a.position == b.position
        assert a.null_count == b.null_count
        assert a.distinct_count == b.distinct_count
        assert a.is_constant == b.is_constant
        if b.mean_value is not None:
            assert a.mean_value == pytest.approx(b.mean_value)
        else:
            assert a.mean_value is None


# --------------------------------------------------------------------------------
# Versioning
# --------------------------------------------------------------------------------


def test_second_version_is_added_not_replaced(db, data_root, sales_csv, tmp_path):
    first = service.create_dataset(sales_csv, name="Sales")

    bigger = tmp_path / "sales_v2.csv"
    bigger.write_text(
        sales_csv.read_text(encoding="utf-8") + "5,2024-03-01,West,700.0,500.0,X\n",
        encoding="utf-8",
    )
    second = service.create_dataset(bigger, dataset_id=first.dataset_id)

    assert second.dataset_id == first.dataset_id
    assert second.version == 2
    assert second.row_count == 5

    with session_scope() as session:
        assert service.get_version(session, first.dataset_id, 1).row_count == 4
        assert service.get_version(session, first.dataset_id, 2).row_count == 5
        # version=None means latest
        assert service.get_version(session, first.dataset_id).version == 2


def test_versions_query_independently(db, data_root, sales_csv, tmp_path):
    first = service.create_dataset(sales_csv)
    bigger = tmp_path / "v2.csv"
    bigger.write_text(
        sales_csv.read_text(encoding="utf-8") + "5,2024-03-01,West,700.0,500.0,X\n",
        encoding="utf-8",
    )
    service.create_dataset(bigger, dataset_id=first.dataset_id)

    q = f"SELECT count(*) FROM {sandbox.TABLE_NAME}"
    assert sandbox.execute_sql(first.dataset_id, 1, q).rows[0][0] == 4
    assert sandbox.execute_sql(first.dataset_id, 2, q).rows[0][0] == 5


# --------------------------------------------------------------------------------
# Listing and lookup
# --------------------------------------------------------------------------------


def test_listing_returns_newest_first(db, data_root, sales_csv):
    service.create_dataset(sales_csv, name="First")
    service.create_dataset(sales_csv, name="Second")

    with session_scope() as session:
        names = [d.name for d in service.list_datasets(session)]

    assert names[:2] == ["Second", "First"]


def test_lookup_of_unknown_dataset_returns_none(db, data_root):
    with session_scope() as session:
        assert service.get_dataset(session, uuid.uuid4()) is None
        assert service.get_version(session, uuid.uuid4()) is None
        assert service.get_stored_profile(session, uuid.uuid4()) is None


# --------------------------------------------------------------------------------
# Consistency between the two stores
# --------------------------------------------------------------------------------


def test_failed_ingest_writes_nothing_anywhere(db, data_root, tmp_path):
    """A rejected upload must leave neither a file nor a row."""
    bad = tmp_path / "headers_only.csv"
    bad.write_text("a,b,c\n", encoding="utf-8")

    with pytest.raises(IngestError):
        service.create_dataset(bad)

    with session_scope() as session:
        assert service.list_datasets(session) == []
    assert list(data_root.iterdir()) == []


def test_parquet_is_removed_if_the_database_write_fails(
    db, data_root, sales_csv, monkeypatch
):
    """The consistency guarantee that matters.

    A Parquet file with no metadata row is invisible to every listing, occupies disk
    forever, and is findable only by walking directories. If the commit fails, the
    file must go.
    """
    from app.data import service as service_module

    def boom(*args, **kwargs):
        raise RuntimeError("simulated database failure")

    monkeypatch.setattr(service_module, "_get_or_create_dataset", boom)

    with pytest.raises(RuntimeError, match="simulated"):
        service.create_dataset(sales_csv)

    with session_scope() as session:
        assert service.list_datasets(session) == []
    # no version directory survived
    leftover = [p for p in data_root.rglob("*.parquet")]
    assert leftover == [], f"orphaned parquet left behind: {leftover}"


def test_delete_removes_both_stores_and_cascades(db, data_root, sales_csv):
    created = service.create_dataset(sales_csv)

    with session_scope() as session:
        assert session.query(DatasetVersion).count() == 1
        assert session.query(ColumnProfile).count() == 6

    assert service.delete_dataset(created.dataset_id) is True

    with session_scope() as session:
        assert session.query(Dataset).count() == 0
        # ON DELETE CASCADE must have taken the children with it
        assert session.query(DatasetVersion).count() == 0
        assert session.query(ColumnProfile).count() == 0

    assert not any(data_root.rglob("*.parquet"))


def test_deleting_an_unknown_dataset_is_not_an_error(db, data_root):
    assert service.delete_dataset(uuid.uuid4()) is False


# --------------------------------------------------------------------------------
# Constraints the database itself enforces
# --------------------------------------------------------------------------------


def test_duplicate_version_number_is_rejected_by_the_database(db, data_root, sales_csv):
    """The filesystem already prevents this via mkdir(exist_ok=False). The unique
    constraint is the same invariant expressed where the database can enforce it too —
    defence in depth against a future code path that bypasses storage."""
    from sqlalchemy.exc import IntegrityError

    created = service.create_dataset(sales_csv)

    with pytest.raises(IntegrityError):
        with session_scope() as session:
            session.add(
                DatasetVersion(
                    dataset_id=created.dataset_id,
                    version=1,  # already taken
                    original_filename="dupe.csv",
                    original_format="csv",
                    source_bytes=1,
                    parquet_bytes=1,
                    row_count=1,
                    column_count=1,
                )
            )
