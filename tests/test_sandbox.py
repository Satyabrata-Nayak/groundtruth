"""Sandbox tests, including the attack corpus.

In M5 the SQL executed here is written by a language model, which can be steered by
text inside the dataset itself. Every query in ATTACKS must be refused. "The model
would never write that" is not a security control.

Each attack is annotated with what it would achieve if it succeeded, so a reader can
tell which are catastrophic and which are merely wrong.
"""

import time
import uuid

import duckdb
import pytest

from app.config import get_settings
from app.data import ingest, sandbox
from app.data.sandbox import (
    SqlExecutionError,
    SqlValidationError,
    execute_sql,
    get_schema,
    validate_sql,
)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    root.mkdir()
    monkeypatch.setattr(get_settings(), "data_dir", root)
    return root


@pytest.fixture
def secret_file(tmp_path):
    """A file the query layer must never be able to read."""
    path = tmp_path / "secret.txt"
    path.write_text("TOPSECRET_CREDENTIAL", encoding="utf-8")
    return path


@pytest.fixture
def dataset(data_root, tmp_path):
    csv = tmp_path / "sales.csv"
    csv.write_text(
        "order_id,order_date,region,revenue,cost\n"
        "1,2024-01-15,North,1200.0,800.0\n"
        "2,2024-01-20,South,450.0,300.0\n"
        "3,2024-02-10,North,980.0,700.0\n"
        "4,2024-02-14,East,620.0,410.0\n",
        encoding="utf-8",
    )
    result = ingest.ingest_file(csv)
    return result.dataset_id, result.version


# ================================================================================
# Queries that must WORK
# ================================================================================


def test_simple_aggregate(dataset):
    ds, v = dataset
    result = execute_sql(ds, v, f"SELECT SUM(revenue) AS total FROM {sandbox.TABLE_NAME}")
    assert result.rows[0][0] == pytest.approx(3250.0)
    assert result.columns == ["total"]
    assert result.truncated is False


def test_group_by(dataset):
    ds, v = dataset
    result = execute_sql(
        ds, v,
        f"SELECT region, SUM(revenue) AS r FROM {sandbox.TABLE_NAME} "
        "GROUP BY region ORDER BY r DESC",
    )
    assert result.row_count == 3
    assert result.as_dicts()[0]["region"] == "North"


def test_cte_is_allowed(dataset):
    """WITH is a legitimate analytical construct and must not be collateral damage."""
    ds, v = dataset
    result = execute_sql(
        ds, v,
        f"WITH per_region AS (SELECT region, SUM(revenue) r FROM {sandbox.TABLE_NAME} "
        "GROUP BY region) SELECT MAX(r) FROM per_region",
    )
    assert result.rows[0][0] == pytest.approx(2180.0)


def test_subquery_and_derived_profit(dataset):
    ds, v = dataset
    result = execute_sql(
        ds, v,
        f"SELECT SUM(revenue - cost) FROM {sandbox.TABLE_NAME} WHERE region IN "
        f"(SELECT region FROM {sandbox.TABLE_NAME} WHERE revenue > 900)",
    )
    assert result.rows[0][0] == pytest.approx(680.0)


def test_union_is_allowed(dataset):
    ds, v = dataset
    result = execute_sql(
        ds, v,
        f"SELECT region FROM {sandbox.TABLE_NAME} WHERE revenue > 1000 "
        f"UNION SELECT region FROM {sandbox.TABLE_NAME} WHERE cost < 350",
    )
    assert result.row_count == 2


def test_trailing_semicolon_is_fine(dataset):
    """A trailing semicolon is normal SQL, not an attempt at a second statement."""
    ds, v = dataset
    result = execute_sql(ds, v, f"SELECT count(*) FROM {sandbox.TABLE_NAME};")
    assert result.rows[0][0] == 4


def test_schema_matches_the_queryable_view(dataset):
    ds, v = dataset
    schema = dict(get_schema(ds, v))
    assert set(schema) == {"order_id", "order_date", "region", "revenue", "cost"}
    assert "DATE" in schema["order_date"].upper()


# ================================================================================
# THE ATTACK CORPUS — every one must be refused
# ================================================================================

ATTACKS = [
    # --- data destruction / modification -------------------------------------
    ("DROP TABLE dataset",                          "destroy the dataset view"),
    ("DELETE FROM dataset",                         "delete rows"),
    ("UPDATE dataset SET revenue = 0",              "falsify data"),
    ("INSERT INTO dataset VALUES (9,'2024-01-01','X',1,1)", "inject rows"),
    ("CREATE TABLE evil (a INT)",                   "create state"),
    ("ALTER TABLE dataset RENAME TO other",         "modify schema"),
    ("TRUNCATE dataset",                            "empty the table"),

    # --- reading files outside the dataset -----------------------------------
    ("SELECT * FROM read_csv('C:/Windows/win.ini')",       "read a system file"),
    ("SELECT * FROM read_text('/etc/passwd')",             "read a system file"),
    ("SELECT * FROM read_blob('C:/Users/nsaty/.ssh/id_rsa')", "exfiltrate a private key"),
    ("SELECT * FROM read_parquet('C:/other/data.parquet')", "read another dataset"),
    ("SELECT * FROM glob('C:/Users/*')",                   "enumerate the filesystem"),
    ("SELECT * FROM read_json_auto('C:/secrets.json')",    "read a system file"),

    # --- writing out / exfiltration ------------------------------------------
    ("COPY (SELECT * FROM dataset) TO 'C:/tmp/leak.csv'",  "exfiltrate to disk"),
    ("EXPORT DATABASE 'C:/tmp/dump'",                      "dump everything"),

    # --- escaping the sandbox ------------------------------------------------
    ("ATTACH 'C:/other.db' AS other",               "mount another database"),
    ("INSTALL httpfs",                              "add network capability"),
    ("LOAD httpfs",                                 "add network capability"),
    ("SET enable_external_access=true",             "undo the lockdown"),
    ("PRAGMA disable_verification",                 "change engine behaviour"),

    # --- multiple statements, the classic injection shape ---------------------
    ("SELECT 1; DROP TABLE dataset",                        "smuggle a second statement"),
    ("SELECT 1; /* comment */ DELETE FROM dataset",         "comment-smuggled statement"),
    ("SELECT 1;;DROP TABLE dataset",                        "empty statement between"),

    # --- hiding a write inside a legitimate-looking query ---------------------
    ("WITH x AS (SELECT 1) DELETE FROM dataset",            "CTE wrapping a delete"),
    ("WITH x AS (SELECT * FROM read_csv('/etc/passwd')) SELECT * FROM x",
     "CTE wrapping a file read"),
    ("SELECT * FROM dataset WHERE region IN (SELECT * FROM read_text('/etc/passwd'))",
     "subquery reading a file"),
    ("SELECT (SELECT content FROM read_text('/etc/passwd')) AS leaked",
     "scalar subquery reading a file"),

    # --- malformed ------------------------------------------------------------
    ("",                                            "empty query"),
    ("   ",                                         "whitespace only"),
    ("NOT SQL AT ALL !!!",                          "garbage"),
]


@pytest.mark.parametrize(("sql", "intent"), ATTACKS, ids=[a[1] for a in ATTACKS])
def test_attack_is_refused(dataset, secret_file, sql, intent):
    """Every attack must raise before or during execution — never return data."""
    ds, v = dataset
    with pytest.raises((SqlValidationError, SqlExecutionError)):
        execute_sql(ds, v, sql)


def test_secret_file_is_unreadable_even_with_a_real_path(dataset, secret_file):
    """The strongest form of the test: a real, existing, readable-by-the-process file.

    A path that does not exist would fail for the wrong reason. This one exists and
    contains a known string, so passing proves the sandbox refused rather than that
    the file was simply absent.
    """
    ds, v = dataset
    assert secret_file.read_text(encoding="utf-8") == "TOPSECRET_CREDENTIAL"

    for sql in (
        f"SELECT * FROM read_text('{secret_file.as_posix()}')",
        f"SELECT * FROM read_csv('{secret_file.as_posix()}')",
        f"SELECT * FROM read_blob('{secret_file.as_posix()}')",
    ):
        with pytest.raises((SqlValidationError, SqlExecutionError)):
            execute_sql(ds, v, sql)


def test_lockdown_holds_at_engine_level_even_if_l1_were_bypassed(dataset, secret_file):
    """Defence in depth: L2 alone must stop a file read.

    This calls the confined connection directly, skipping the sqlglot allowlist, to
    prove the parser is not the only thing standing between a model and the filesystem.
    If this ever fails, L1 has become load-bearing on its own — which is exactly the
    fragile design the layering exists to avoid.
    """
    from app.data import storage

    ds, v = dataset
    parquet = storage.resolve_existing_parquet(ds, v)
    con = sandbox._open_confined_connection(parquet)
    try:
        # duckdb.Error specifically — a blind `Exception` would also pass if the test
        # itself had a typo, which would silently stop testing the sandbox at all.
        with pytest.raises(duckdb.Error):
            con.execute(f"SELECT * FROM read_text('{secret_file.as_posix()}')").fetchall()
        # ...and the dataset itself is still readable.
        assert con.execute(f'SELECT count(*) FROM "{sandbox.TABLE_NAME}"').fetchone()[0] == 4
    finally:
        con.close()


# ================================================================================
# Limits
# ================================================================================


def test_row_cap_truncates_and_reports_it(dataset):
    """Truncation must be visible. A silently truncated result is a wrong answer."""
    ds, v = dataset
    result = execute_sql(ds, v, f"SELECT * FROM {sandbox.TABLE_NAME}", max_rows=2)
    assert result.row_count == 2
    assert result.truncated is True


def test_exact_fit_is_not_reported_as_truncated(dataset):
    """Off-by-one guard: 4 rows with a cap of 4 is complete, not truncated."""
    ds, v = dataset
    result = execute_sql(ds, v, f"SELECT * FROM {sandbox.TABLE_NAME}", max_rows=4)
    assert result.row_count == 4
    assert result.truncated is False


def test_timeout_interrupts_a_runaway_query(dataset):
    """A recursive CTE counting to 100 million: entirely valid SQL, unbounded runtime.

    Deliberately NOT a `range()` cross join — table functions are rejected by L1 now,
    so that query would fail validation and never exercise the watchdog. This one is
    a realistic accident for a model to write, and the only thing standing between it
    and a hung worker is the timeout.
    """
    ds, v = dataset
    runaway = (
        "WITH RECURSIVE loop(n) AS ("
        "  SELECT 1 UNION ALL SELECT n + 1 FROM loop WHERE n < 100000000"
        ") SELECT count(*) FROM loop"
    )
    started = time.perf_counter()
    with pytest.raises(SqlExecutionError, match="time limit"):
        execute_sql(ds, v, runaway, timeout_s=2.0)
    # The watchdog must actually fire, not merely let a fast query through.
    assert time.perf_counter() - started < 20.0


# ================================================================================
# Dataset scoping
# ================================================================================


def test_unknown_dataset_is_a_clean_error(data_root):
    from app.data.storage import DatasetNotFoundError

    with pytest.raises(DatasetNotFoundError):
        execute_sql(uuid.uuid4(), 1, f"SELECT 1 FROM {sandbox.TABLE_NAME}")


def test_hostile_dataset_id_is_refused(data_root):
    from app.data.storage import InvalidDatasetIdError

    with pytest.raises(InvalidDatasetIdError):
        execute_sql("../../../etc/passwd", 1, f"SELECT * FROM {sandbox.TABLE_NAME}")


def test_other_versions_are_not_visible(data_root, tmp_path):
    """Each query sees exactly one version. Immutability is worthless if v1 can read v2."""
    csv = tmp_path / "a.csv"
    csv.write_text("a\n1\n", encoding="utf-8")
    first = ingest.ingest_file(csv)

    csv.write_text("a\n1\n2\n3\n", encoding="utf-8")
    ingest.ingest_file(csv, dataset_id=first.dataset_id)

    v1 = execute_sql(first.dataset_id, 1, f"SELECT count(*) FROM {sandbox.TABLE_NAME}")
    v2 = execute_sql(first.dataset_id, 2, f"SELECT count(*) FROM {sandbox.TABLE_NAME}")
    assert v1.rows[0][0] == 1
    assert v2.rows[0][0] == 3


# ================================================================================
# Validator unit tests
# ================================================================================


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "SELECT * FROM dataset",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "SELECT a FROM dataset UNION SELECT b FROM dataset",
        "SELECT a FROM dataset EXCEPT SELECT b FROM dataset",
    ],
)
def test_validator_accepts_read_only_queries(sql):
    assert validate_sql(sql) is not None


def test_validator_error_messages_are_actionable(dataset):
    """Messages are fed back to a model for repair in M5, so they must say what is
    wrong, not merely that something is."""
    ds, v = dataset
    with pytest.raises(SqlValidationError, match="only SELECT"):
        execute_sql(ds, v, "DELETE FROM dataset")
    with pytest.raises(SqlValidationError, match="exactly one statement"):
        execute_sql(ds, v, "SELECT 1; SELECT 2")
    with pytest.raises(SqlValidationError, match="not permitted"):
        execute_sql(ds, v, "SELECT * FROM read_csv('x.csv')")
