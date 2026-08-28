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
    has_groq_key = bool(settings.groq_api_key)

    return [
        ModelOut(
            name=profile.name,
            label=profile.label,
            tagline=profile.tagline,
            provider=profile.provider,
            good_at=profile.good_at,
            weak_at=profile.weak_at,
            speed=profile.speed_label,
            cost=profile.cost_label,
            preview=profile.preview,
            accuracy_pct=profile.accuracy_pct,
            reasons=profile.reasons,
            size_gb=profile.size_gb,
            available=_available(profile, pulled, has_groq_key),
            is_default=profile.name == settings.llm_model,
        )
        for profile in CATALOGUE
    ]


def _available(profile, pulled: set[str], has_groq_key: bool) -> bool:
    """Can this model actually be used right now?

    Availability means something different per provider, and conflating them would make
    the picker lie. A local model is available when it is PULLED; a hosted one is
    available when there is a KEY. Showing a Groq model as ready with no key would send
    a user into a two-minute wait that ends in a 401.

    An empty `pulled` set means Ollama is unreachable, not that nothing is installed —
    so local models stay available and fail with the message that says why.
    """
    if profile.provider == "groq":
        return has_groq_key
    return not pulled or profile.name in pulled


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
