"""Dataset endpoints: upload, list, inspect, delete.

THE UPLOAD IS STREAMED, AND CAPPED WHILE STREAMING
--------------------------------------------------
`await file.read()` is the one-liner everyone writes and it loads the whole upload
into memory. On a 15.8 GB machine sharing RAM with Postgres and a local model, a
512 MB upload doing that is a real problem, and two at once is an outage.

So the body is copied to a temporary file a megabyte at a time, and the size cap is
checked *during* the copy rather than after it. Checking afterwards means the damage
is already done — the point of a limit is to stop before you are hurt, not to report
that you were.

`Content-Length` is not trusted for this. It is a client-supplied header; a client
that lies about it is precisely the client the cap exists for.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.api.schemas import DatasetOut, ProfileOut
from app.config import get_settings
from app.data import service
from app.data.ingest import IngestError, detect_format
from app.data.storage import InvalidDatasetIdError

# FastAPI's recommended form. Putting Depends() in a default argument works, but the
# default is then a live object evaluated at import time, which ruff flags (B008) and
# which breaks the moment the function is called outside FastAPI -- as a test would.
# Annotated keeps the dependency in the TYPE, where it describes rather than defaults.
SessionDep = Annotated[Session, Depends(get_session)]

router = APIRouter(prefix="/datasets", tags=["datasets"])

_CHUNK_BYTES = 1024 * 1024


@router.post("", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
def upload_dataset(
    session: SessionDep,
    file: Annotated[UploadFile, File(description="A .csv or .parquet file")],
    name: Annotated[str | None, Form()] = None,
    description: Annotated[str | None, Form()] = None,
    dataset_id: Annotated[
        str | None,
        Form(description="Add a new version to this dataset instead of creating one"),
    ] = None,
) -> DatasetOut:
    """Ingest a file and register it as a queryable dataset version."""
    settings = get_settings()
    filename = file.filename or "upload"

    # Reject on the extension before spooling half a gigabyte to disk to discover the
    # same thing. `detect_format` owns the list of what is supported; duplicating it
    # here is how the two drift apart.
    try:
        detect_format(filename)
    except IngestError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc

    if dataset_id is not None:
        try:
            uuid.UUID(dataset_id)
        except (ValueError, AttributeError, TypeError) as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"dataset_id is not a valid UUID: {dataset_id!r}"
            ) from exc

    temp_path = _spool_to_disk(file, max_bytes=settings.max_upload_mb * 1024 * 1024)
    try:
        created = service.create_dataset(
            temp_path,
            name=name or Path(filename).stem,
            description=description,
            dataset_id=dataset_id,
            original_filename=filename,
        )
    except IngestError as exc:
        # The file is malformed, not the request. 422 says "I understood you, and what
        # you sent me cannot be processed", which is the honest description.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except InvalidDatasetIdError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    dataset = service.get_dataset(session, created.dataset_id)
    assert dataset is not None  # written in the transaction that just committed
    return DatasetOut.model_validate(dataset)


def _spool_to_disk(upload: UploadFile, *, max_bytes: int) -> Path:
    """Copy the upload to a temp file, refusing it the moment it exceeds the cap.

    Returns the temp file's path; the caller is responsible for deleting it. Written
    with `delete=False` because on Windows a NamedTemporaryFile cannot be reopened by
    another handle while it is still open, and DuckDB needs to open it by path.
    """
    suffix = Path(upload.filename or "upload").suffix
    handle = NamedTemporaryFile(delete=False, suffix=suffix)
    path = Path(handle.name)
    written = 0

    try:
        with handle:
            while chunk := upload.file.read(_CHUNK_BYTES):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status.HTTP_413_CONTENT_TOO_LARGE,
                        f"upload exceeds the {max_bytes // (1024 * 1024)} MB limit",
                    )
                handle.write(chunk)
    except BaseException:
        # Includes the 413 above. A rejected upload must not leave its partial bytes
        # in the temp directory, where nothing will ever clean them up.
        path.unlink(missing_ok=True)
        raise

    return path


@router.get("", response_model=list[DatasetOut])
def list_datasets(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[DatasetOut]:
    return [DatasetOut.model_validate(d) for d in service.list_datasets(session, limit=limit)]


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, session: SessionDep) -> DatasetOut:
    dataset = _require_dataset(session, dataset_id)
    return DatasetOut.model_validate(dataset)


@router.get("/{dataset_id}/profile", response_model=ProfileOut)
def get_profile(
    dataset_id: str,
    session: SessionDep,
    version: Annotated[int | None, Query(ge=1, description="Defaults to the latest")] = None,
) -> ProfileOut:
    _require_dataset(session, dataset_id)
    stored = service.get_version(session, dataset_id, version)
    if stored is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"dataset {dataset_id} has no version {version}"
            if version
            else f"dataset {dataset_id} has no versions",
        )
    return ProfileOut(
        dataset_id=stored.dataset_id,
        version=stored.version,
        row_count=stored.row_count,
        column_count=stored.column_count,
        duplicate_row_count=stored.duplicate_row_count,
        columns=stored.columns,
    )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: str) -> None:
    """Remove a dataset from Postgres and from disk.

    Takes no session dependency on purpose: `service.delete_dataset` opens its own
    transaction so that the database delete commits *before* the files are removed.
    Sharing the request's transaction would delete the files first and leave rows
    pointing at nothing if the commit then failed.
    """
    try:
        removed = service.delete_dataset(dataset_id)
    except InvalidDatasetIdError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if not removed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no dataset {dataset_id}")


def _require_dataset(session: Session, dataset_id: str):
    try:
        dataset = service.get_dataset(session, dataset_id)
    except InvalidDatasetIdError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no dataset {dataset_id}")
    return dataset
