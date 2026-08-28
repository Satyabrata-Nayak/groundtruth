"""The API's contract, written down.

WHY RESPONSE MODELS EXIST AT ALL
--------------------------------
FastAPI will happily serialise an ORM object. Doing so publishes whatever happens to
be on the model — including `worker_id`, and in future anything added to it — and the
API's shape then changes silently whenever the database does. Declaring the response
explicitly makes the boundary a decision rather than an accident, and it is what
generates the OpenAPI document the frontend is written against.

WHAT IS DELIBERATELY NOT EXPOSED
--------------------------------
`worker_id` and `heartbeat_at` are internal scheduling state. A client that could see
them would start depending on them, and they are the two fields most likely to change
when the queue is tuned. `attempts` IS exposed: a user watching a job get retried
deserves to know that is what is happening.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ColumnOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    position: int
    duckdb_type: str
    semantic_type: str
    null_count: int
    null_fraction: float
    distinct_count: int | None
    min_value: str | None
    max_value: str | None
    mean_value: float | None
    stddev_value: float | None
    q25_value: float | None
    q50_value: float | None
    q75_value: float | None
    is_constant: bool
    is_high_cardinality: bool


class VersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    version: int
    original_filename: str
    original_format: str
    source_bytes: int
    parquet_bytes: int
    row_count: int
    column_count: int
    duplicate_row_count: int
    ingested_at: datetime


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime
    versions: list[VersionOut] = Field(default_factory=list)

    # `computed_field`, not a bare `@property`. Pydantic serialises only declared
    # fields and computed ones, so as a plain property this was documented in the class,
    # invisible in the JSON and absent from the OpenAPI schema — the frontend read
    # `dataset.latest_version` and silently got undefined. A property that callers are
    # meant to see has to say so.
    @computed_field
    @property
    def latest_version(self) -> int | None:
        return max((v.version for v in self.versions), default=None)


class ProfileOut(BaseModel):
    """A dataset version's shape and per-column statistics.

    Read back from Postgres, never recomputed — profiling a multi-million-row file
    takes seconds and was already paid for at ingest.
    """

    dataset_id: uuid.UUID
    version: int
    row_count: int
    column_count: int
    duplicate_row_count: int
    columns: list[ColumnOut]


class AnalysisCreate(BaseModel):
    dataset_id: uuid.UUID
    question: str = Field(min_length=1, max_length=2000)
    # None means "whatever the latest version is when the request is handled". The
    # resolved number is stored on the row, so the analysis is pinned from that moment
    # on even though the request did not name a version.
    version: int | None = Field(default=None, ge=1)
    # Makes the POST safe to retry. Two requests carrying the same key produce one
    # analysis; the second gets 200 and the first one's id, not a duplicate job.
    idempotency_key: str | None = Field(default=None, max_length=128)

    # WHICH MODEL SHOULD ANSWER, chosen per question rather than per deployment.
    # Validated against a catalogue in the route: this string arrives from a browser,
    # and passing it to Ollama unchecked would let a request pull and run any model on
    # the host. None means "whatever the worker is configured with".
    model: str | None = Field(default=None, max_length=128)
    # Tri-state. None is "the asker expressed no preference", which is not False.
    thinking: bool | None = None


class AnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    dataset_id: uuid.UUID
    dataset_version: int
    question: str
    status: str
    attempts: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    result: dict[str, Any] | None
    error: str | None
    # Echoed back so a client can show what a stored analysis was actually run with,
    # rather than what the picker happens to be set to now.
    llm_model: str | None = None
    llm_thinking: bool | None = None


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    kind: str
    message: str
    payload: dict[str, Any] | None
    created_at: datetime


class EventPage(BaseModel):
    """Events plus the cursor to ask for next time.

    `next_after` rather than a page number: the UI polls a growing list, and an offset
    would re-read everything it already has on every poll. Passing back the last id it
    saw makes each poll cost only what is new.
    """

    events: list[EventOut]
    next_after: int
    status: str


class HealthOut(BaseModel):
    status: str
    database: bool
    version: str


class ModelOut(BaseModel):
    """One selectable model, with the measurements that justify choosing it.

    Every figure here came from this project's own evaluation set, so the UI can put
    real numbers in front of the choice instead of adjectives. `available` is checked
    against Ollama at request time: a model that is catalogued but not pulled is shown
    with the command to get it rather than silently offered and then failing.
    """

    name: str
    label: str
    tagline: str
    good_at: list[str]
    weak_at: list[str]
    speed: str
    accuracy_pct: int | None
    reasons: bool
    size_gb: float
    available: bool
    is_default: bool
