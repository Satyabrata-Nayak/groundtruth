"""Choosing the model per question, and the allowlist that makes that safe.

The choice is stored on the analysis row rather than read from configuration when the
worker picks the job up, for the same reason `dataset_version` is: two answers to the
same question can legitimately differ because one was asked of a 3B model and one of a
4B, and an answer that cannot say which is one nobody can act on.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.agent.models import BY_NAME, CATALOGUE, is_selectable, profile_for
from app.api.main import app
from app.data.service import create_dataset
from app.db.models import Analysis
from app.db.session import session_scope
from app.jobs import queue

CSV = "region,category,revenue,units\nWest,Books,100.5,3\nEast,Books,80.0,2\nNorth,Toys,10.0,1\n"


@pytest.fixture
def client(db, data_root):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def dataset(client):
    response = client.post(
        "/datasets",
        files={"file": ("sales.csv", io.BytesIO(CSV.encode()), "text/csv")},
        data={"name": "sales"},
    )
    assert response.status_code == 201
    return response.json()


@pytest.fixture
def sales_dataset(db, data_root, tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text(CSV, encoding="utf-8")
    return create_dataset(path, name="sales")


# ============================================================== the catalogue


def test_every_catalogued_model_states_a_weakness():
    """A chooser that only lists strengths is an advert. The user is picking between two
    things that are each bad at something, and the bad half is the deciding half."""
    for profile in CATALOGUE:
        assert profile.good_at, profile.name
        assert profile.weak_at, profile.name


def test_the_catalogue_is_an_allowlist():
    """`model` arrives as a string in a JSON body from a browser. Passing it through to
    Ollama would let a request pull and load any model on the host."""
    assert is_selectable("qwen3:4b")
    assert not is_selectable("llama3:70b")
    assert not is_selectable("../../etc/passwd")


def test_an_unmeasured_model_gets_no_invented_profile():
    """An operator may point LLM_MODEL anywhere. That must produce a working system and
    no fabricated claims about how accurate it is."""
    assert profile_for("something-nobody-benchmarked") is None


# ============================================================== GET /models


def test_the_models_endpoint_reports_what_is_known(client):
    body = client.get("/models").json()
    assert {m["name"] for m in body} == set(BY_NAME)
    assert sum(1 for m in body if m["is_default"]) == 1
    for entry in body:
        assert entry["speed"]
        assert entry["good_at"] and entry["weak_at"]


def test_only_a_reasoning_model_advertises_reasoning(client):
    """The thinking toggle is rendered from this flag. Offering it for a model with no
    reasoning step would be a control that does nothing."""
    by_name = {m["name"]: m for m in client.get("/models").json()}
    assert by_name["qwen3:4b"]["reasons"] is True
    assert by_name["qwen2.5:3b-instruct"]["reasons"] is False


# ============================================================== POST /analyses


def test_the_chosen_model_is_stored_on_the_row(client, dataset):
    body = client.post(
        "/analyses",
        json={
            "dataset_id": dataset["id"],
            "question": "which country earns most?",
            "model": "qwen2.5:3b-instruct",
            "thinking": False,
        },
    ).json()

    with session_scope() as session:
        stored = session.get(Analysis, body["id"])
        assert stored.llm_model == "qwen2.5:3b-instruct"
        assert stored.llm_thinking is False


def test_choosing_nothing_stores_nothing(client, dataset):
    """NULL means "use whatever the worker is configured with", which is not the same as
    choosing the default explicitly — and is what every row written before these columns
    existed means."""
    body = client.post("/analyses", json={"dataset_id": dataset["id"], "question": "q"}).json()
    with session_scope() as session:
        stored = session.get(Analysis, body["id"])
        assert stored.llm_model is None
        assert stored.llm_thinking is None


def test_an_unknown_model_is_refused_with_the_valid_options(client, dataset):
    response = client.post(
        "/analyses",
        json={"dataset_id": dataset["id"], "question": "q", "model": "gpt-4o"},
    )
    assert response.status_code == 400
    assert "qwen3:4b" in response.json()["detail"]


def test_the_stored_choice_comes_back_on_the_analysis(client, dataset):
    """So a client shows what an old analysis was RUN with, not what the picker happens
    to be set to now."""
    created = client.post(
        "/analyses",
        json={"dataset_id": dataset["id"], "question": "q", "model": "qwen3:4b"},
    ).json()
    fetched = client.get(f"/analyses/{created['id']}").json()
    assert fetched["llm_model"] == "qwen3:4b"


# ============================================================== the queue


def test_the_claim_carries_the_choice_to_the_worker(db, sales_dataset):
    """Stored and then ignored would be the worst of both: the row would claim an answer
    came from a model that never saw it."""
    with session_scope() as session:
        queue.enqueue(
            session,
            dataset_id=sales_dataset.dataset_id,
            dataset_version=1,
            question="q",
            llm_model="qwen2.5:3b-instruct",
            llm_thinking=False,
        )

    with session_scope() as session:
        claimed = queue.claim_next(session, "test-worker")

    assert claimed is not None
    assert claimed.llm_model == "qwen2.5:3b-instruct"
    assert claimed.llm_thinking is False
