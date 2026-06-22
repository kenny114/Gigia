"""
message_schemas.py – Pydantic v2 models for all inter-component messages.

All models are immutable by default (``model_config = ConfigDict(frozen=True)``).
Use ``.model_copy(update={...})`` to produce modified copies.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubBotType(str, Enum):
    SCRAPER = "scraper"
    SELENIUM = "selenium"
    BROWSER = "browser"
    GENERIC = "generic"
    SKILL = "skill"     # calls back to the almcp gateway /api/brain/execute
    CODE = "code"       # sandboxed Python execution
    FILE = "file"       # read/write files in the workspace directory
    SHELL = "shell"     # run whitelisted shell commands on the VPS


class ErrorType(str, Enum):
    CAPTCHA_DETECTED = "CaptchaDetected"
    TIMEOUT = "Timeout"
    HTTP_404 = "HTTP404"
    HTTP_403 = "HTTP403"
    HTTP_5XX = "HTTP5xx"
    PARSE_ERROR = "ParseError"
    BROWSER_CRASH = "BrowserCrash"
    EXECUTION_ERROR = "ExecutionError"   # code/shell returned non-zero exit
    PERMISSION_DENIED = "PermissionDenied"  # path traversal or blocked command
    UNKNOWN = "Unknown"


class ManagerStatusEnum(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    CRASHED = "crashed"


# ---------------------------------------------------------------------------
# Core message envelope
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """Generic event/message envelope passed through the event bus."""

    model_config = ConfigDict(frozen=True)

    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)
    correlation_id: str = Field(default_factory=_new_id)


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class Goal(BaseModel):
    """A high-level user goal submitted to the PerceptionBrain."""

    model_config = ConfigDict(frozen=True)

    goal_id: str = Field(default_factory=_new_id)
    description: str
    submitted_at: datetime = Field(default_factory=_utcnow)
    correlation_id: str = Field(default_factory=_new_id)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """An atomic unit of work produced by the PlanningBrain."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(default_factory=_new_id)
    goal_id: str
    title: str
    description: str
    sub_bot_type: SubBotType = SubBotType.SCRAPER
    # Set when sub_bot_type == SKILL — the almcp catalog slug to execute.
    skill_slug: Optional[str] = None
    priority: int = 0                          # lower = higher priority
    dependencies: List[str] = Field(default_factory=list)   # list of task_ids
    status: TaskStatus = TaskStatus.PENDING
    correlation_id: str = Field(default_factory=_new_id)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubBotInstruction(BaseModel):
    """
    Instruction sent from a ManagerBot to a SubBot.

    ``parameters`` is a free-form dict whose keys depend on the sub-bot type:
      - scraper: url, css_selectors, headers, proxy
      - selenium: url, actions, wait_for, screenshot_on_error
    """

    model_config = ConfigDict(frozen=True)

    instruction_id: str = Field(default_factory=_new_id)
    task_id: str
    sub_bot_type: SubBotType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: str = Field(default_factory=_new_id)
    timeout_seconds: int = 120


class Result(BaseModel):
    """Successful output from a SubBot."""

    model_config = ConfigDict(frozen=True)

    result_id: str = Field(default_factory=_new_id)
    instruction_id: str
    task_id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    produced_at: datetime = Field(default_factory=_utcnow)
    correlation_id: str = Field(default_factory=_new_id)


class ErrorReport(BaseModel):
    """Structured error returned by a SubBot when execution fails."""

    model_config = ConfigDict(frozen=True)

    error_id: str = Field(default_factory=_new_id)
    instruction_id: str
    task_id: str
    error_type: ErrorType = ErrorType.UNKNOWN
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)
    correlation_id: str = Field(default_factory=_new_id)
    retryable: bool = True


class SubBotError(BaseModel):
    """Internal model used inside ManagerBot to represent a sub-bot failure."""

    model_config = ConfigDict(frozen=True)

    error_report: ErrorReport
    attempt_number: int = 1
    instruction: SubBotInstruction


class EscalationReport(BaseModel):
    """Sent by a ManagerBot to the LearningBrain when a problem is unsolvable."""

    model_config = ConfigDict(frozen=True)

    escalation_id: str = Field(default_factory=_new_id)
    manager_id: str
    task_id: str
    problem: str
    context: Dict[str, Any] = Field(default_factory=dict)
    error_history: List[ErrorReport] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utcnow)
    correlation_id: str = Field(default_factory=_new_id)


class MemoryEntry(BaseModel):
    """A stored strategy / best-practice record in GlobalMemory."""

    model_config = ConfigDict(frozen=True)

    entry_id: str = Field(default_factory=_new_id)
    problem: str
    solution: str
    success_rate: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ManagerStatus(BaseModel):
    """Snapshot of a single ManagerBot's current state."""

    model_config = ConfigDict(frozen=True)

    manager_id: str
    task_id: Optional[str] = None
    status: ManagerStatusEnum = ManagerStatusEnum.IDLE
    active_sub_bots: int = 0
    last_heartbeat: datetime = Field(default_factory=_utcnow)
    error_count: int = 0


class SystemHealth(BaseModel):
    """Aggregate health snapshot produced by PerceptionBrain."""

    model_config = ConfigDict(frozen=True)

    total_managers: int = 0
    running_managers: int = 0
    crashed_managers: int = 0
    completed_managers: int = 0
    manager_statuses: List[ManagerStatus] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Gateway (almcp) orchestration contract
# ---------------------------------------------------------------------------

class GatewayCallback(BaseModel):
    """How Gigia calls back to the gateway to execute a skill or deliver results."""

    model_config = ConfigDict(frozen=True)

    execute_url: str          # e.g. https://almcp.vercel.app/api/brain/execute
    token: str                # Bearer API key scoped to this owner
    result_url: str = ""      # e.g. https://almcp.vercel.app/api/brain/result — POST synthesized answer here


class OrchestrateCandidate(BaseModel):
    """A skill the gateway retrieved and is making available to Gigia's planner."""

    model_config = ConfigDict(frozen=True)

    slug: str
    name: str
    description: str
    tags: List[str] = Field(default_factory=list)
    credits: int = 1
    best_used_when: Optional[str] = None
    avoid_when: Optional[str] = None
    example_call: Optional[Dict[str, Any]] = None


class OrchestrateRequest(BaseModel):
    """
    Payload the gateway POSTs to /orchestrate when delegating a complex goal.
    Gigia decomposes it into a task DAG and executes each step via callback.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str                                    # gateway brain_requests id
    task: str                                      # the human goal
    input: Dict[str, Any] = Field(default_factory=dict)
    max_credits: Optional[int] = None
    candidates: List[OrchestrateCandidate] = Field(default_factory=list)
    callback: GatewayCallback


class OrchestrateResponse(BaseModel):
    """Immediate response from /orchestrate (Gigia accepted the run)."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str = "accepted"   # accepted | rejected
    message: Optional[str] = None
