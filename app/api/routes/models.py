"""What the user may choose to answer with, and what is known about each option.

WHY THE API OWNS THIS AND NOT THE FRONTEND
------------------------------------------
Two of the three facts a chooser needs can only be known here. Whether a model is
actually pulled is a question for Ollama; which one is the default is a question for
this deployment's configuration. Only the third — the prose — could live in the
frontend, and splitting one card's contents across two repositories to save one
endpoint is not a trade worth making.

Availability is checked live rather than cached. A model the catalogue knows about but
that nobody has pulled must be shown as unavailable, with the command to get it, not
silently offered and then failing two minutes into a question.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter

from app.agent.models import CATALOGUE
from app.api.schemas import ModelOut
from app.config import get_settings

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelOut])
def list_models() -> list[ModelOut]:
    """The selectable models, in catalogue order, each marked available or not."""
    settings = get_settings()
    pulled = _pulled_models(settings.ollama_base_url)

    return [
        ModelOut(
            name=profile.name,
            label=profile.label,
            tagline=profile.tagline,
            good_at=profile.good_at,
            weak_at=profile.weak_at,
            speed=profile.speed_label,
            accuracy_pct=profile.accuracy_pct,
            reasons=profile.reasons,
            size_gb=profile.size_gb,
            # An empty set means Ollama is unreachable, not that nothing is installed.
            # Reporting everything as unavailable then would be a worse lie than
            # reporting it available and failing with the message that says why.
            available=not pulled or profile.name in pulled,
            is_default=profile.name == settings.llm_model,
        )
        for profile in CATALOGUE
    ]


def _pulled_models(base_url: str) -> set[str]:
    """Model names Ollama has on disk. Empty when the server cannot be reached.

    Short timeout and a swallowed error, deliberately: this endpoint decorates a menu.
    A dead Ollama must not make the model picker hang or 500 — the health dot already
    reports that, and the analysis itself fails with an instruction if it matters.
    """
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3.0)
        response.raise_for_status()
        return {model.get("name", "") for model in response.json().get("models", [])}
    except httpx.HTTPError:
        return set()
