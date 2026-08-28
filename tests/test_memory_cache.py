"""Conversation memory, the answer cache, and the query rewriter's gate.

These three are one feature seen from three sides: a thread makes a follow-up mean
something, the rewriter turns it into a question that stands alone, and a standalone
question is the only kind that can be cached.
"""

from __future__ import annotations

import io
import uuid

import pytest
from fastapi.testclient import TestClient

from app.agent import memory, rewrite
from app.api.main import app
from app.data.service import create_dataset
from app.db.models import Analysis, AnalysisStatus, Conversation
from app.db.session import session_scope
from app.jobs import cache

CSV = "region,category,revenue\nWest,Books,100.5\nEast,Books,80.0\n"


@pytest.fixture
def client(db, data_root):
    with TestClient(app) as test_client:
        yield test_client


def _upload(client, name):
    response = client.post(
        "/datasets",
        files={"file": (f"{name}.csv", io.BytesIO(CSV.encode()), "text/csv")},
        data={"name": name},
    )
    assert response.status_code == 201
    return response.json()


@pytest.fixture
def dataset(client):
    return _upload(client, "sales")


@pytest.fixture
def second_dataset(client):
    return _upload(client, "other")


@pytest.fixture
def stored(db, data_root, tmp_path):
    path = tmp_path / "s.csv"
    path.write_text(CSV, encoding="utf-8")
    return create_dataset(path, name="sales")


RESULT = {
    "answer": "East leads at 20.",
    "table": {"columns": ["region", "revenue"], "rows": [["East", 20.0]]},
    "steps": [{"tool": "execute_sql", "ok": True, "arguments": {"sql": "SELECT 1"}}],
    "warnings": [],
}


# ============================================================== the cache key


def test_the_same_question_written_differently_is_one_entry():
    assert cache.question_hash("Which Country Earns Most?  ") == cache.question_hash(
        "which country earns most"
    )


def test_different_questions_are_different_entries():
    assert cache.question_hash("which country earns most") != cache.question_hash(
        "which product earns most"
    )


def test_an_answer_with_a_warning_is_never_cached():
    """The one that matters. A warning means the agent ran out of budget, answered
    without querying, or wrote an untraceable figure — and replaying THAT instantly is
    worse than recomputing it slowly, because speed reads as confidence."""
    flagged = {**RESULT, "warnings": ["a figure could not be traced to a tool result"]}
    assert cache.is_cacheable(RESULT)
    assert not cache.is_cacheable(flagged)


def test_an_answer_with_nothing_computed_is_not_cached():
    assert not cache.is_cacheable({"answer": "I think so.", "warnings": []})
    assert not cache.is_cacheable({"answer": "", "table": {"rows": [[1]]}})


# ============================================================== the cache, live


def test_a_stored_answer_comes_back_marked_as_replayed(stored):
    key = {
        "dataset_id": stored.dataset_id,
        "dataset_version": 1,
        "question": "which region earns most?",
        "llm_model": "qwen3:4b",
    }
    with session_scope() as session:
        assert cache.lookup(session, **key) is None
        cache.store(session, **key, result=RESULT)

    with session_scope() as session:
        hit = cache.lookup(session, **{**key, "question": "Which Region Earns Most"})

    assert hit is not None
    assert hit["answer"] == RESULT["answer"]
    # An answer arriving in five milliseconds has to say why.
    assert hit["cached"] is True


def test_a_different_model_does_not_share_a_cached_answer(stored):
    """The two models score 60% and 29%. Serving one's answer to somebody who asked for
    the other would be answering a question nobody asked."""
    key = {
        "dataset_id": stored.dataset_id,
        "dataset_version": 1,
        "question": "which region earns most?",
    }
    with session_scope() as session:
        cache.store(session, **key, llm_model="qwen3:4b", result=RESULT)
    with session_scope() as session:
        assert cache.lookup(session, **key, llm_model="qwen2.5:3b-instruct") is None


def test_a_new_dataset_version_does_not_serve_the_old_answer(stored):
    key = {
        "dataset_id": stored.dataset_id,
        "question": "which region earns most?",
        "llm_model": "qwen3:4b",
    }
    with session_scope() as session:
        cache.store(session, **key, dataset_version=1, result=RESULT)
    with session_scope() as session:
        assert cache.lookup(session, **key, dataset_version=2) is None


def test_storing_the_same_answer_twice_does_not_raise(stored):
    """Two workers can finish the same question at the same moment. The loser must not
    turn a completed analysis into a unique-violation traceback."""
    key = {
        "dataset_id": stored.dataset_id,
        "dataset_version": 1,
        "question": "q",
        "llm_model": "m",
    }
    for _ in range(2):
        with session_scope() as session:
            cache.store(session, **key, result=RESULT)


# ============================================================== memory


def _thread(session, dataset_id, exchanges):
    conversation = Conversation(dataset_id=dataset_id, dataset_version=1)
    session.add(conversation)
    session.flush()
    for index, (question, answer, sql) in enumerate(exchanges):
        session.add(
            Analysis(
                dataset_id=dataset_id,
                dataset_version=1,
                question=question,
                status=AnalysisStatus.SUCCEEDED,
                conversation_id=conversation.id,
                turn_index=index,
                result={
                    "answer": answer,
                    "steps": [{"tool": "execute_sql", "ok": True, "arguments": {"sql": sql}}],
                },
            )
        )
    session.flush()
    return conversation.id


def test_a_thread_carries_the_question_the_answer_and_the_sql(stored):
    """The SQL is the part people leave out and the most valuable of the three: a model
    that can see the previous query writes the follow-up by editing one clause."""
    with session_scope() as session:
        conversation_id = _thread(
            session,
            stored.dataset_id,
            [("which region earns most?", "East, at 20.", "SELECT region FROM dataset")],
        )
        turns = memory.recent_turns(session, conversation_id)

    assert len(turns) == 1
    assert turns[0].question == "which region earns most?"
    assert turns[0].answer == "East, at 20."
    assert "SELECT region" in turns[0].sql


def test_only_the_last_three_exchanges_are_carried(stored):
    """The window is bounded by the context, not by taste: at ~120 tokens a turn, the
    schema and samples already take ~1,200 of 8,192."""
    with session_scope() as session:
        conversation_id = _thread(
            session, stored.dataset_id, [(f"q{i}", f"a{i}", f"SELECT {i}") for i in range(6)]
        )
        turns = memory.recent_turns(session, conversation_id)

    assert [t.question for t in turns] == ["q3", "q4", "q5"]


def test_a_failed_analysis_is_not_carried_as_context(stored):
    """Replaying it would tell the model that something happened which did not."""
    with session_scope() as session:
        conversation_id = _thread(session, stored.dataset_id, [("q1", "a1", "SELECT 1")])
        session.add(
            Analysis(
                dataset_id=stored.dataset_id,
                dataset_version=1,
                question="q2",
                status=AnalysisStatus.FAILED,
                conversation_id=conversation_id,
                turn_index=1,
                error="boom",
            )
        )
        session.flush()
        turns = memory.recent_turns(session, conversation_id)

    assert [t.question for t in turns] == ["q1"]


def test_no_conversation_means_no_history(stored):
    with session_scope() as session:
        assert memory.recent_turns(session, None) == []


def test_the_history_block_forbids_reusing_an_old_number():
    rendered = memory.render([memory.Turn("which region?", "East, at 20.", "SELECT 1")])
    assert "EARLIER IN THIS CONVERSATION" in rendered
    assert "Do NOT reuse an" in rendered
    assert memory.render([]) == ""


# ============================================================== the rewriter's gate


def test_a_standalone_question_never_pays_for_a_rewrite():
    """The gate exists so the common case does not spend a second learning nothing."""
    turns = [memory.Turn("q", "a", None)]
    assert not rewrite.needs_rewriting("which country generated the most revenue?", turns)
    assert not rewrite.needs_rewriting("which country earns most", turns)


def test_a_follow_up_is_recognised():
    turns = [memory.Turn("q", "a", None)]
    assert rewrite.needs_rewriting("what about France?", turns)
    assert rewrite.needs_rewriting("and why is that", turns)
    assert rewrite.needs_rewriting("by month", turns)


def test_nothing_is_rewritten_without_a_thread():
    assert not rewrite.needs_rewriting("what about France?", [])


def test_a_rewrite_that_is_not_a_question_is_refused():
    """A small model asked for one sentence sometimes returns a paragraph explaining
    what it changed. Sending that on would replace the user's words with commentary."""
    assert rewrite._accept("x" * 400, "original") == "original"
    assert rewrite._accept("", "original") == "original"
    assert rewrite._accept("Which country earns most?", "orig") == "Which country earns most?"


def test_an_unreachable_rewriter_changes_nothing():
    """An optimisation that can change the answer is a bug."""
    from app.agent.llm import ModelError

    class Broken:
        model = "x"
        think = None

        def chat(self, *args, **kwargs):
            raise ModelError("down")

        def close(self):
            pass

    turns = [memory.Turn("which country earns most?", "the UK", None)]
    assert rewrite.rewrite("what about France?", turns, client=Broken()) == "what about France?"


# ============================================================== the thread, over HTTP


def test_asking_without_a_conversation_starts_one(client, dataset):
    body = client.post(
        "/analyses", json={"dataset_id": dataset["id"], "question": "which region earns most?"}
    ).json()
    assert body["conversation_id"] is not None
    assert body["turn_index"] == 0


def test_a_follow_up_continues_the_same_thread(client, dataset):
    first = client.post(
        "/analyses", json={"dataset_id": dataset["id"], "question": "which region earns most?"}
    ).json()
    second = client.post(
        "/analyses",
        json={
            "dataset_id": dataset["id"],
            "question": "what about the lowest?",
            "conversation_id": first["conversation_id"],
        },
    ).json()

    assert second["conversation_id"] == first["conversation_id"]
    assert second["turn_index"] == 1


def test_a_thread_cannot_be_moved_to_another_dataset(client, dataset, second_dataset):
    """Half a thread about one dataset and half about another would let the model carry
    a fact from one into an answer about the other."""
    first = client.post("/analyses", json={"dataset_id": dataset["id"], "question": "q"}).json()
    response = client.post(
        "/analyses",
        json={
            "dataset_id": second_dataset["id"],
            "question": "q",
            "conversation_id": first["conversation_id"],
        },
    )
    assert response.status_code == 400
    assert "different dataset" in response.json()["detail"]


def test_an_unknown_conversation_is_refused(client, dataset):
    response = client.post(
        "/analyses",
        json={"dataset_id": dataset["id"], "question": "q", "conversation_id": str(uuid.uuid4())},
    )
    assert response.status_code == 404
