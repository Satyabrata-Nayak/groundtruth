"""Dataset file layout, and the only place an ID becomes a filesystem path.

THIS IS A TRUST BOUNDARY, NOT A PATH UTILITY
--------------------------------------------
In M5 an LLM chooses which dataset to analyse. It supplies a `dataset_id`; it must
never supply, see, or influence a filesystem path. Every path in the system is
derived here from an ID that has been validated as a UUID, and every derived path is
checked to be inside the data root before it is returned.

    LLM  ──"dataset_id=7f3a9c22-..."──►  resolve_version_path()  ──►  trusted path

The bad version of this module accepts a string and joins it to a root. That is how
`../../../../etc/passwd` and `C:/Users/me/.ssh/id_rsa` get read.

IMMUTABILITY
------------
    data/datasets/<dataset_id>/v<n>/data.parquet

A version directory is written once and never modified. Re-uploading the same logical
dataset creates v2 alongside v1. This is what makes a stored analysis reproducible: a
saved run references (dataset_id, version) and can be re-executed to the same numbers
later. With mutable files, every stored result is a claim about data that may no
longer exist.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from app.config import get_settings

DATA_FILENAME = "data.parquet"
_VERSION_DIR_RE = re.compile(r"^v(\d+)$")


class InvalidDatasetIdError(ValueError):
    """The supplied dataset id is not a well-formed UUID."""


class DatasetNotFoundError(FileNotFoundError):
    """No stored data exists for this dataset id / version."""


class UnsafePathError(RuntimeError):
    """A resolved path escaped the data root. Should be unreachable; loud if not."""


def data_root() -> Path:
    """Absolute path of the directory holding all datasets."""
    return get_settings().data_dir.resolve()


def parse_dataset_id(dataset_id: str | uuid.UUID) -> uuid.UUID:
    """Validate an untrusted dataset id.

    Parsing as a UUID is the actual defence: a UUID cannot contain a path separator,
    a `..`, a drive letter, or a NUL byte. Anything path-shaped fails here, before it
    is ever concatenated to anything.
    """
    if isinstance(dataset_id, uuid.UUID):
        return dataset_id
    try:
        return uuid.UUID(str(dataset_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise InvalidDatasetIdError(f"not a valid dataset id: {dataset_id!r}") from exc


def _guard_inside_root(path: Path) -> Path:
    """Belt-and-braces: confirm a resolved path is really under the data root.

    `parse_dataset_id` should make escape impossible already. This catches the case
    where that assumption stops holding — a future caller, a symlink inside the data
    directory, a change to the layout — and fails loudly instead of reading the file.
    """
    root = data_root()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise UnsafePathError(f"path escapes data root: {resolved}")
    return resolved


def dataset_dir(dataset_id: str | uuid.UUID) -> Path:
    """Directory holding every version of one dataset."""
    return _guard_inside_root(data_root() / str(parse_dataset_id(dataset_id)))


def version_dir(dataset_id: str | uuid.UUID, version: int) -> Path:
    """Directory of one immutable dataset version."""
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"version must be a positive integer, got {version!r}")
    return _guard_inside_root(dataset_dir(dataset_id) / f"v{version}")


def parquet_path(dataset_id: str | uuid.UUID, version: int) -> Path:
    """Path to one version's Parquet file. Existence is NOT implied."""
    return _guard_inside_root(version_dir(dataset_id, version) / DATA_FILENAME)


def resolve_existing_parquet(dataset_id: str | uuid.UUID, version: int) -> Path:
    """Path to a version's Parquet file, erroring if it is not there.

    This is the accessor the query layer uses: it never invents a path for data that
    does not exist, so a missing dataset surfaces as a clean error rather than as a
    confusing DuckDB parse failure on a nonexistent file.
    """
    path = parquet_path(dataset_id, version)
    if not path.is_file():
        raise DatasetNotFoundError(f"no data for dataset {dataset_id} version {version}")
    return path


def existing_versions(dataset_id: str | uuid.UUID) -> list[int]:
    """Version numbers that exist on disk, ascending."""
    directory = dataset_dir(dataset_id)
    if not directory.is_dir():
        return []
    found = []
    for child in directory.iterdir():
        match = _VERSION_DIR_RE.match(child.name)
        if child.is_dir() and match and (child / DATA_FILENAME).is_file():
            found.append(int(match.group(1)))
    return sorted(found)


def next_version(dataset_id: str | uuid.UUID) -> int:
    """The version number a new upload should claim."""
    versions = existing_versions(dataset_id)
    return (versions[-1] + 1) if versions else 1


def allocate_version_dir(dataset_id: str | uuid.UUID) -> tuple[int, Path]:
    """Create and return the next version directory.

    `mkdir(exist_ok=False)` is deliberate: it makes claiming a version atomic at the
    filesystem level, so two concurrent uploads cannot both believe they own v2 —
    the loser raises FileExistsError rather than silently overwriting.
    """
    version = next_version(dataset_id)
    directory = version_dir(dataset_id, version)
    directory.mkdir(parents=True, exist_ok=False)
    return version, directory


def sql_path_literal(path: Path) -> str:
    """Quote a filesystem path for inlining into SQL.

    DuckDB does NOT accept prepared parameters in every position — verified: both
    `COPY ... TO ?` and `CREATE VIEW ... read_parquet(?)` fail (the first misbinds
    silently, the second raises "This type of statement can't be prepared"). Those
    places need the path inlined.

    Only ever called with a server-generated path derived from a validated UUID, never
    with user input. The quote-doubling is kept anyway: this function cannot verify its
    caller kept that promise, and correctness here is free.
    """
    return "'" + path.as_posix().replace("'", "''") + "'"
