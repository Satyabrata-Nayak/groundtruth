"""The reasoning layer: a language model constrained by deterministic tools.

    contract.py   what every analysis engine must satisfy
    llm.py        the only code that talks to Ollama
    prompt.py     what the model is told before it is asked anything
    analyst.py    the loop: model proposes, tools compute, evidence is collected
    evidence.py   the table and chart shown underneath the answer

The division that matters: the model chooses WHAT to compute, and never computes
anything. Every number that reaches a user has been through DuckDB.
"""

from __future__ import annotations

from app.agent.analyst import AGENT_TOOLS, ENGINE, run_agent_analysis
from app.agent.contract import AnalysisFailed, Checkpoint, Emit
from app.agent.llm import LlmClient, ModelError, ModelTurn, ModelUnavailable, ToolCall

__all__ = [
    "AGENT_TOOLS",
    "ENGINE",
    "AnalysisFailed",
    "Checkpoint",
    "Emit",
    "LlmClient",
    "ModelError",
    "ModelTurn",
    "ModelUnavailable",
    "ToolCall",
    "run_agent_analysis",
]
