"""
schemas.py – Pydantic / dataclass schemas for memory layer objects.

These mirror the SQLite table columns so that GlobalMemory can hydrate
rows directly into typed objects.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Strategy record  (maps to  strategies  table)
# ---------------------------------------------------------------------------

@dataclass
class StrategyRecord:
    """
    A stored problem→solution strategy with an observed success rate.

    Attributes
    ----------
    problem:
        Natural-language description of the problem.
    solution:
        Natural-language description of the recommended solution.
    success_rate:
        Float in [0, 1].  Updated incrementally as the strategy is tried.
    tags:
        Optional list of keyword tags for similarity search.
    """
    problem: str
    solution: str
    success_rate: float = 0.0
    entry_id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Failure log  (maps to  failure_logs  table)
# ---------------------------------------------------------------------------

@dataclass
class FailureLog:
    """
    A record of a single failed task execution attempt.

    Attributes
    ----------
    problem:
        Short description of what went wrong.
    context:
        Free-form dict with surrounding context (task_id, manager_id, …).
    error_details:
        Free-form dict with raw error info (type, message, stack trace, …).
    """
    problem: str
    context: Dict[str, Any] = field(default_factory=dict)
    error_details: Dict[str, Any] = field(default_factory=dict)
    log_id: str = field(default_factory=_new_id)
    logged_at: datetime = field(default_factory=_utcnow)
    resolved: bool = False
    resolution_notes: Optional[str] = None
