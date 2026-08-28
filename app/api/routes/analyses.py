"""Analysis endpoints: ask a question, watch it run, read the answer.

WHAT POST /analyses DOES NOT DO
-------------------------------
It does not analyse anything. It resolves the dataset version, writes one PENDING row,
and returns 201 with an id — a few milliseconds, entirely predictable. The work
happens in the worker process.

That split is the whole point of M4. Doing the analysis inside the request would mean
the browser holds a connection open for the length of a model call, every reverse
proxy in the path applies its own idea of a timeout, a page refresh abandons the work,
and restarting the API loses every request in flight.

    POST   /analyses           enqueue, return an id            (milliseconds)
    GET    /analyses/{id}      poll for status and result       (milliseconds)
    GET    /analyses/{id}/events   the trail, incrementally     (milliseconds)
    POST   /analyses/{id}/cancel   ask it to stop               (milliseconds)

Polling rather than websockets, deliberately: a websocket would need connection state,
reconnection logic and a way to replay what was missed while disconnected. The event
cursor already gives replay for free, and at one poll per second the cost is a
primary-key lookup.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent.models import BY_NAME, is_selectable
from app.api.deps import get_session
from app.api.schemas import AnalysisCreate, AnalysisOut, EventOut, EventPage
from app.data import service
from app.data.storage import InvalidDatasetIdError
from app.db.models import Analysis, AnalysisEvent, Conversation
from app.jobs import queue

# See the note in routes/datasets.py: the dependency belongs in the type, not in a
# default argument evaluated at import time.
SessionDep = Annotated[Session, Depends(get_session)]

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post("", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
def create_analysis(
    payload: AnalysisCreate,
    response: Response,
    session: SessionDep,
) -> AnalysisOut:
    """Queue a question. Returns immediately; the worker does the work.

    Responds 201 for a new analysis and **200 for one an idempotency key matched** —
    the distinction is the useful part of the pattern. A client retrying after a
    dropped connection can tell "my first attempt already landed" from "this is new",
    without either creating a duplicate job or being told its retry failed.
    """
    try:
        dataset = service.get_dataset(session, payload.dataset_id)
    except InvalidDatasetIdError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no dataset {payload.dataset_id}")

    version = _resolve_version(session, payload)

    # An ALLOWLIST, not a passthrough. `model` is a string from a browser; handing it
    # to Ollama unchecked would let any request pull and load an arbitrary model on the
    # host, which is a resource-exhaustion hole rather than a feature. An operator can
    # still run anything through LLM_MODEL; a request may only name what was measured.
    if payload.model is not None and not is_selectable(payload.model):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown model {payload.model!r}. Choose one of: {', '.join(sorted(BY_NAME))}",
        )

    conversation_id, turn_index = _thread(session, payload, dataset.id, version)

    analysis, created = queue.enqueue(
        session,
        dataset_id=dataset.id,
        dataset_version=version,
        question=payload.question.strip(),
        idempotency_key=payload.idempotency_key,
        llm_model=payload.model,
        llm_thinking=payload.thinking,
        conversation_id=conversation_id,
        turn_index=turn_index,
    )
    if not created:
        response.status_code = status.HTTP_200_OK
    return AnalysisOut.model_validate(analysis)


def _thread(
    session: Session,
    payload: AnalysisCreate,
    dataset_id: uuid.UUID,
    version: int,
) -> tuple[uuid.UUID, int]:
    """Find or start the conversation this question belongs to, and its position.

    Creating one implicitly means a client never needs two round trips to start a
    thread: ask a question, get a `conversation_id` back, send it with the next one.

    A conversation is REFUSED if it belongs to a different dataset or version. Half a
    thread about `retail` and half about `sensors` would let the model carry a fact from
    one into an answer about the other — the exact plausible-but-wrong failure this
    project exists to prevent. Pinning the version matters for the same reason it does
    on the analysis: a follow-up must be about the data the first answer described.
    """
    if payload.conversation_id is None:
        conversation = Conversation(dataset_id=dataset_id, dataset_version=version)
        session.add(conversation)
        session.flush()
        return conversation.id, 0

    conversation = session.get(Conversation, payload.conversation_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no conversation {payload.conversation_id}")
    if conversation.dataset_id != dataset_id or conversation.dataset_version != version:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "that conversation is about a different dataset or version. Start a new one "
            "rather than mixing two datasets into one thread.",
        )

    # COUNT rather than max(turn_index) + 1: the position only has to be a stable total
    # order within the thread, and a count cannot be knocked out of sequence by a row
    # whose index was never written.
    used = session.scalar(
        select(func.count())
        .select_from(Analysis)
        .where(Analysis.conversation_id == conversation.id)
    )
    return conversation.id, int(used or 0)


def _resolve_version(session: Session, payload: AnalysisCreate) -> int:
    """Pin the version now, so the answer cannot silently be about different data.

    Omitting `version` means "the latest at the time of asking", and that number is
    written to the row. If a new version is uploaded while the job waits in the queue,
    the worker still analyses the one the user was looking at.
    """
    stored = service.get_version(session, payload.dataset_id, payload.version)
    if stored is None:
        detail = (
            f"dataset {payload.dataset_id} has no version {payload.version}"
            if payload.version is not None
            else f"dataset {payload.dataset_id} has no versions to analyse"
        )
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail)
    return stored.version


@router.get("/{analysis_id}", response_model=AnalysisOut)
def get_analysis(analysis_id: uuid.UUID, session: SessionDep) -> AnalysisOut:
    return AnalysisOut.model_validate(_require_analysis(session, analysis_id))


@router.get("/{analysis_id}/events", response_model=EventPage)
def get_events(
    analysis_id: uuid.UUID,
    session: SessionDep,
    after: Annotated[int, Query(ge=0, description="Return events with an id above this")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> EventPage:
    """The trail so far, from a cursor.

    `status` is returned alongside so a polling client needs ONE request per tick
    rather than two. Without it the UI would poll this endpoint for the trail and
    `/analyses/{id}` for the status, and would then have to reason about the two
    answers disagreeing because they were read a few milliseconds apart.
    """
    analysis = _require_analysis(session, analysis_id)
    events = list(
        session.scalars(
            select(AnalysisEvent)
            .where(AnalysisEvent.analysis_id == analysis_id, AnalysisEvent.id > after)
            .order_by(AnalysisEvent.id)
            .limit(limit)
        )
    )
    return EventPage(
        events=[EventOut.model_validate(e) for e in events],
        next_after=events[-1].id if events else after,
        status=analysis.status,
    )


@router.post("/{analysis_id}/cancel", response_model=AnalysisOut)
def cancel_analysis(analysis_id: uuid.UUID, session: SessionDep) -> AnalysisOut:
    """Ask for an analysis to stop.

    A queued job is cancelled outright. A running one gets a flag — the worker owns
    the DuckDB connection and (in M5) the model call, and it is the only party that
    can stop cleanly and record where it stopped. It notices within one heartbeat
    interval, so the response says RUNNING and the status endpoint reports CANCELLED
    a moment later. Reporting CANCELLED immediately would be a lie the client would
    then have to un-believe.
    """
    resulting = queue.cancel(session, analysis_id)
    if resulting is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no analysis {analysis_id}")
    session.flush()
    return AnalysisOut.model_validate(_require_analysis(session, analysis_id))


def _require_analysis(session: Session, analysis_id: uuid.UUID) -> Analysis:
    analysis = session.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no analysis {analysis_id}")
    return analysis
