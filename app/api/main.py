"""The FastAPI application.

    browser ──HTTP──► API ──INSERT──► Postgres ◄──claim── worker ──► tools ──► DuckDB
                       ▲                                     │
                       └──────── polls status and events ────┘

The API never touches DuckDB and never runs an analysis. It reads and writes Postgres
and returns in milliseconds. Everything slow is on the other side of the queue, which
is what lets a single-process dev server stay responsive while a 60-second analysis
runs.

Run it with:

    uv run uvicorn app.api.main:app --reload
    uv run python -m app.worker
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.routes import analyses, datasets, models
from app.api.schemas import HealthOut
from app.config import get_settings
from app.data.ingest import IngestError
from app.data.storage import DatasetNotFoundError, InvalidDatasetIdError
from app.db.session import get_engine

log = logging.getLogger("app.api")

app = FastAPI(
    title="AI Data Analyser",
    version="0.4.0",
    summary="Deterministic tools do the maths; the model only chooses which to call.",
    description=(
        "M4: the full request path with no AI in it. Upload a dataset, ask a "
        "question, and a separate worker process runs a fixed analysis through the "
        "M3 tool registry. M5 replaces the fixed analysis with a model-driven one "
        "and changes none of these endpoints."
    ),
)

app.add_middleware(
    CORSMiddleware,
    # An explicit list, not "*". The wildcard cannot be combined with credentials, and
    # naming the dev server means an unexpected origin shows up as a CORS error in the
    # console rather than working by accident and breaking on deployment.
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(analyses.router)
app.include_router(models.router)


# --------------------------------------------------------------------------------
# Domain errors -> HTTP status codes, in one place
# --------------------------------------------------------------------------------
#
# Without these, every route needs its own try/except around every service call, and
# the one that gets forgotten returns a 500 with a stack trace. Mapping the domain's
# own exception types once means a new route inherits correct behaviour by default.


@app.exception_handler(InvalidDatasetIdError)
def _invalid_id(_request: Request, exc: InvalidDatasetIdError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


@app.exception_handler(DatasetNotFoundError)
def _not_found(_request: Request, exc: DatasetNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})


@app.exception_handler(IngestError)
def _bad_file(_request: Request, exc: IngestError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": str(exc)}
    )


# --------------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------------


@app.get("/healthz", response_model=HealthOut, tags=["meta"])
def healthz() -> HealthOut:
    """Liveness plus one real dependency check.

    A health endpoint that only returns 200 tells you the process is running, which
    you already knew because it answered. Actually issuing `SELECT 1` is what
    distinguishes "up" from "up and able to do its job" — the state where an API
    accepts questions it can never enqueue.
    """
    database_ok = True
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        log.warning("health check: database unreachable", exc_info=True)
        database_ok = False

    return HealthOut(
        status="ok" if database_ok else "degraded", database=database_ok, version=app.version
    )
