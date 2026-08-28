"""Tests for the dataset path trust boundary.

The attack cases matter more than the happy path. In M5 an LLM supplies the dataset
id; if a path-shaped id could reach the filesystem, the model could read any file the
server process can. Every one of these must be rejected before a path is built.
"""

import uuid

import pytest

from app.config import get_settings
from app.data import storage
from app.data.storage import (
    DatasetNotFoundError,
    InvalidDatasetIdError,
)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Point the settings singleton at a temp directory for the duration of a test."""
    settings = get_settings()
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


# --------------------------------------------------------------------------------
# Rejecting hostile ids
# --------------------------------------------------------------------------------

HOSTILE_IDS = [
    "../../../../etc/passwd",
    "..\\..\\..\\Windows\\win.ini",
    "C:/Users/nsaty/.ssh/id_rsa",
    "/etc/shadow",
    "....//....//secret",
    "valid-looking-but-not-a-uuid",
    "7f3a9c22-0000-0000-0000-00000000000",  # one char short
    "7f3a9c22-0000-0000-0000-000000000000/../..",
    "",
    ".",
    "..",
    "con",  # reserved device name on Windows
    "a\x00b",  # NUL byte
]


@pytest.mark.parametrize("hostile", HOSTILE_IDS)
def test_hostile_dataset_ids_are_rejected(data_root, hostile):
    """A UUID cannot contain a separator, a '..', a drive letter or a NUL byte.

    Parsing as a UUID *is* the defence — anything path-shaped fails before it is
    concatenated to anything.
    """
    with pytest.raises(InvalidDatasetIdError):
        storage.dataset_dir(hostile)


def test_valid_uuid_is_accepted(data_root):
    ds = uuid.uuid4()
    assert storage.dataset_dir(ds).is_relative_to(data_root.resolve())


def test_uuid_object_and_string_agree(data_root):
    ds = uuid.uuid4()
    assert storage.dataset_dir(ds) == storage.dataset_dir(str(ds))


# --------------------------------------------------------------------------------
# Version handling
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("bad_version", [0, -1, "1", 1.5, None, True])
def test_invalid_versions_are_rejected(data_root, bad_version):
    """`True` is in this list on purpose: bool is a subclass of int in Python, so a
    naive `isinstance(v, int)` check would accept True and build a directory 'vTrue'.
    """
    with pytest.raises((ValueError, TypeError)):
        storage.version_dir(uuid.uuid4(), bad_version)


def test_versions_start_at_one_and_increment(data_root):
    ds = uuid.uuid4()
    assert storage.existing_versions(ds) == []
    assert storage.next_version(ds) == 1

    v1, dir1 = storage.allocate_version_dir(ds)
    (dir1 / storage.DATA_FILENAME).write_bytes(b"stub")
    assert v1 == 1
    assert storage.next_version(ds) == 2

    v2, dir2 = storage.allocate_version_dir(ds)
    (dir2 / storage.DATA_FILENAME).write_bytes(b"stub")
    assert v2 == 2
    assert storage.existing_versions(ds) == [1, 2]


def test_allocate_is_atomic_against_double_claim(data_root):
    """mkdir(exist_ok=False) means a second claimant fails instead of overwriting."""
    ds = uuid.uuid4()
    _, directory = storage.allocate_version_dir(ds)
    with pytest.raises(FileExistsError):
        directory.mkdir(parents=True, exist_ok=False)


def test_incomplete_version_is_not_counted(data_root):
    """A directory without data.parquet is a half-finished upload, not a version.

    Without this, a crash mid-ingest would leave a phantom version that later reads
    resolve to a missing file.
    """
    ds = uuid.uuid4()
    storage.allocate_version_dir(ds)  # dir created, no parquet written
    assert storage.existing_versions(ds) == []


# --------------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------------


def test_resolve_missing_data_raises(data_root):
    with pytest.raises(DatasetNotFoundError):
        storage.resolve_existing_parquet(uuid.uuid4(), 1)


def test_resolve_existing_returns_path_inside_root(data_root):
    ds = uuid.uuid4()
    _, directory = storage.allocate_version_dir(ds)
    (directory / storage.DATA_FILENAME).write_bytes(b"stub")

    path = storage.resolve_existing_parquet(ds, 1)
    assert path.is_file()
    assert path.is_relative_to(data_root.resolve())
    assert path.name == storage.DATA_FILENAME
