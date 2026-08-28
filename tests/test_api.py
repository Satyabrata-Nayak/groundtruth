"""The HTTP surface, driven the way the browser drives it.

WHY THESE GO THROUGH A REAL CLIENT
----------------------------------
Calling the route functions directly would skip everything that actually goes wrong at
a boundary: status codes, multipart parsing, query-parameter coercion, the session
dependency's transaction, and pydantic's validation of the response. Those are the
parts a frontend collides with, so they are the parts worth exercising.

`TestClient` runs the ASGI app in-process — no server, no port — so this stays fast
while still being a genuine request.
"""

from __future__ import annotations

import io
import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.worker.loop import Worker

pytestmark = pytest.mark.integration

CSV = (
    "region,category,revenue,units\n"
    "West,Books,100.5,3\n"
    "West,Toys,50.25,1\n"
    "East,Books,80.0,2\n"
    "East,Toys,20.0,1\n"
    "North,Books,60.0,2\n"
    "North,Toys,10.0,1\n"
)


@pytest.fixture
def client(db, data_root):
    with TestClient(app) as test_client:
        yield test_client


def upload(client, content=CSV, filename="sales.csv", **form):
    return client.post(
        "/datasets",
        files={"file": (filename, io.BytesIO(content.encode()), "text/csv")},
        data=form,
    )


@pytest.fixture
def dataset(client):
    response = upload(client, name="sales")
    assert response.status_code == 201
    return response.json()


# ----------------------------------------------------------------------- meta


def test_healthz_reports_the_database(client):
    """A health check that only proves the process answered is not a health check."""
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["database"] is True


def test_the_openapi_document_covers_every_endpoint(client):
    """The frontend is written against this document, so an endpoint missing from it
    is an endpoint the frontend cannot be generated or checked against."""
    paths = client.get("/openapi.json").json()["paths"]
    assert {
        "/datasets",
        "/datasets/{dataset_id}",
        "/datasets/{dataset_id}/profile",
        "/analyses",
        "/analyses/{analysis_id}",
        "/analyses/{analysis_id}/events",
        "/analyses/{analysis_id}/cancel",
        "/healthz",
    } <= set(paths)


# ------------------------------------------------------------------- datasets


def test_upload_returns_the_dataset_with_its_first_version(client):
    body = upload(client, name="sales").json()
    assert body["name"] == "sales"
    version = body["versions"][0]
    assert version["version"] == 1
    assert version["row_count"] == 6
    assert version["column_count"] == 4
    assert version["original_format"] == "csv"


def test_uploading_again_with_a_dataset_id_adds_a_version(client, dataset):
    body = upload(client, dataset_id=dataset["id"]).json()
    assert body["id"] == dataset["id"]
    assert [v["version"] for v in body["versions"]] == [1, 2]


def test_an_unsupported_file_type_is_rejected_before_it_is_read(client):
    """415, not 500, and refused on the extension so a 500 MB `.exe` is never spooled
    to disk just to be rejected afterwards."""
    response = upload(client, content="MZ", filename="virus.exe")
    assert response.status_code == 415
    assert ".exe" in response.json()["detail"]


def test_a_file_that_is_not_really_a_csv_is_rejected_as_unprocessable(client):
    """422: the request was understood, the payload cannot be used. Distinct from 415,
    which is about a type we never accept."""
    response = upload(client, content="\x00\x01\x02 not a csv at all", filename="broken.csv")
    assert response.status_code == 422


def test_an_upload_over_the_limit_is_refused(client, monkeypatch):
    """The cap is enforced DURING the copy. Checking afterwards means the disk is
    already full by the time anyone complains."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_mb", 0)
    response = upload(client)
    assert response.status_code == 413
    assert "limit" in response.json()["detail"]


def test_listing_and_fetching(client, dataset):
    assert [d["id"] for d in client.get("/datasets").json()] == [dataset["id"]]
    assert client.get(f"/datasets/{dataset['id']}").json()["name"] == "sales"


def test_a_malformed_id_is_a_bad_request_and_an_absent_one_is_a_not_found(client):
    """Different mistakes, different codes. Collapsing both into 404 tells a client
    that its typo'd URL is a missing resource, and it retries forever."""
    assert client.get("/datasets/not-a-uuid").status_code == 400
    assert client.get(f"/datasets/{uuid.uuid4()}").status_code == 404


def test_the_profile_is_the_one_stored_at_ingest(client, dataset):
    profile = client.get(f"/datasets/{dataset['id']}/profile").json()
    assert profile["row_count"] == 6
    by_name = {c["name"]: c for c in profile["columns"]}
    assert by_name["region"]["semantic_type"] == "categorical"
    assert by_name["region"]["distinct_count"] == 3
    assert by_name["revenue"]["semantic_type"] == "numeric"
    # Exact, not estimated (D-012). Six distinct revenues in six rows.
    assert by_name["revenue"]["distinct_count"] == 6


def test_asking_for_a_version_that_does_not_exist_is_a_not_found(client, dataset):
    assert client.get(f"/datasets/{dataset['id']}/profile?version=99").status_code == 404


def test_delete_removes_it_from_both_stores(client, dataset):
    assert client.delete(f"/datasets/{dataset['id']}").status_code == 204
    assert client.get(f"/datasets/{dataset['id']}").status_code == 404
    assert client.delete(f"/datasets/{dataset['id']}").status_code == 404


# ------------------------------------------------------------------- analyses


def test_creating_an_analysis_returns_immediately_with_a_pending_row(client, dataset):
    """The whole point of the queue: this call does no analysis at all."""
    response = client.post(
        "/analyses", json={"dataset_id": dataset["id"], "question": "which region sells most?"}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "PENDING"
    assert body["result"] is None
    assert body["attempts"] == 0
    assert body["dataset_version"] == 1


def test_the_version_is_pinned_at_the_moment_of_asking(client, dataset):
    """A new upload while the job waits in the queue must not change what is analysed.

    Otherwise a user asks a question about v1, uploads v2 a second later, and gets an
    answer about data they have not seen — which is wrong in the worst way, because it
    is plausible.
    """
    queued = client.post("/analyses", json={"dataset_id": dataset["id"], "question": "q"}).json()
    upload(client, dataset_id=dataset["id"])  # v2 arrives

    assert client.get(f"/analyses/{queued['id']}").json()["dataset_version"] == 1
    # ...and a question asked now gets v2.
    later = client.post("/analyses", json={"dataset_id": dataset["id"], "question": "q"}).json()
    assert later["dataset_version"] == 2


def test_an_explicit_version_is_honoured(client, dataset):
    upload(client, dataset_id=dataset["id"])
    body = client.post(
        "/analyses", json={"dataset_id": dataset["id"], "question": "q", "version": 1}
    ).json()
    assert body["dataset_version"] == 1


def test_an_idempotency_key_turns_a_retry_into_a_lookup(client, dataset):
    """201 for the new one, 200 for the retry. The status code is how a client that
    lost its connection tells "already done" from "do it again"."""
    payload = {
        "dataset_id": dataset["id"],
        "question": "which region sells most?",
        "idempotency_key": "retry-me",
    }
    first = client.post("/analyses", json=payload)
    second = client.post("/analyses", json=payload)

    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]


def test_analysing_a_dataset_that_does_not_exist_is_a_not_found(client):
    response = client.post("/analyses", json={"dataset_id": str(uuid.uuid4()), "question": "q"})
    assert response.status_code == 404


def test_an_empty_question_is_rejected_by_validation(client, dataset):
    assert (
        client.post("/analyses", json={"dataset_id": dataset["id"], "question": ""}).status_code
        == 422
    )


# --------------------------------------------------------------------- events


def test_events_are_returned_from_a_cursor_and_never_repeat(client, dataset):
    """The polling contract. A UI that re-reads the whole list every second would
    re-transfer a hundred-step M5 trace a hundred times."""
    analysis_id = client.post(
        "/analyses", json={"dataset_id": dataset["id"], "question": "q"}
    ).json()["id"]

    first = client.get(f"/analyses/{analysis_id}/events").json()
    assert [e["kind"] for e in first["events"]] == ["QUEUED"]
    assert first["status"] == "PENDING"

    # Nothing new has happened, so a poll from the cursor returns nothing.
    assert (
        client.get(f"/analyses/{analysis_id}/events?after={first['next_after']}").json()["events"]
        == []
    )

    Worker(worker_id="api-test-worker").tick()

    page = client.get(f"/analyses/{analysis_id}/events?after={first['next_after']}").json()
    assert page["status"] == "SUCCEEDED"
    assert [e["kind"] for e in page["events"]] == [
        "CLAIMED",
        # Which engine ran is the first thing recorded, so a trail is self-describing:
        # "why did I get a mechanical answer" is answerable from the events alone.
        "NOTE",
        "TOOL_CALL",
        "TOOL_RESULT",
        "NOTE",
        "TOOL_CALL",
        "TOOL_RESULT",
        "TOOL_CALL",
        "TOOL_RESULT",
        "SUCCEEDED",
    ]
    assert (
        client.get(f"/analyses/{analysis_id}/events?after={page['next_after']}").json()["events"]
        == []
    )


def test_the_status_travels_with_the_events(client, dataset):
    """One request per poll tick, not two. Two would let the trail and the status
    disagree because they were read a few milliseconds apart."""
    analysis_id = client.post(
        "/analyses", json={"dataset_id": dataset["id"], "question": "q"}
    ).json()["id"]
    assert "status" in client.get(f"/analyses/{analysis_id}/events").json()


# ------------------------------------------------------- the full round trip


def test_upload_ask_run_and_read_the_answer(client, dataset):
    """M4's exit criterion, in one test."""
    analysis_id = client.post(
        "/analyses", json={"dataset_id": dataset["id"], "question": "which region sells most?"}
    ).json()["id"]

    assert Worker(worker_id="api-test-worker").tick() is True

    body = client.get(f"/analyses/{analysis_id}").json()
    assert body["status"] == "SUCCEEDED"
    assert body["error"] is None

    result = body["result"]
    assert result["engine"] == "hardcoded-v1"
    assert "West" in result["answer"]
    assert result["table"]["rows"][0][0] == "West"
    assert result["chart"]["chart"]["data"][0]["x"] == "West"
    assert [s["tool"] for s in result["steps"]] == [
        "inspect_schema",
        "compare_groups",
        "create_chart",
    ]


# ---------------------------------------------------------------- cancellation


def test_cancelling_a_queued_analysis_stops_the_worker_taking_it(client, dataset):
    analysis_id = client.post(
        "/analyses", json={"dataset_id": dataset["id"], "question": "q"}
    ).json()["id"]

    assert client.post(f"/analyses/{analysis_id}/cancel").json()["status"] == "CANCELLED"
    assert Worker(worker_id="api-test-worker").tick() is False
    assert client.get(f"/analyses/{analysis_id}").json()["status"] == "CANCELLED"


def test_cancelling_a_finished_analysis_reports_its_real_status(client, dataset):
    """Not an error, and not a lie. The job is over; the answer is what it is."""
    analysis_id = client.post(
        "/analyses", json={"dataset_id": dataset["id"], "question": "q"}
    ).json()["id"]
    Worker(worker_id="api-test-worker").tick()

    assert client.post(f"/analyses/{analysis_id}/cancel").json()["status"] == "SUCCEEDED"


def test_cancelling_something_that_does_not_exist_is_a_not_found(client):
    assert client.post(f"/analyses/{uuid.uuid4()}/cancel").status_code == 404


def test_latest_version_is_actually_in_the_json(client, dataset):
    """It was declared as a bare @property, so pydantic never serialised it: documented
    in the class, absent from the response, and `undefined` in the frontend."""
    body = client.get("/datasets").json()[0]
    assert body["latest_version"] == 1
    assert (
        "latest_version"
        in client.get("/openapi.json").json()["components"]["schemas"]["DatasetOut"]["properties"]
    )
