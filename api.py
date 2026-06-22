"""
api.py – FastAPI wrapper around the Giga AI bot system.

Always-on Railway web service. Starts all 4 brains on startup and
runs forever. Results are captured from the event bus and persisted
to SQLite, then returned via REST endpoints.

Endpoints
---------
  POST /goal       – Submit a new goal
  GET  /status     – Active manager statuses
  GET  /results    – Scraped results stored in DB
  GET  /health     – Railway health check
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiosqlite
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from giga_ai.messaging.message_schemas import (
    GatewayCallback,
    OrchestrateCandidate,
    OrchestrateRequest,
    OrchestrateResponse,
)

from main_bot import MainBot
from giga_ai.config import load_config
from giga_ai.messaging.event_bus import EventType
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Results DB (separate table in the same SQLite file)
# ---------------------------------------------------------------------------

_DB_PATH: str = ""  # set during startup from config


async def _ensure_results_table(db: aiosqlite.Connection) -> None:
    await db.execute("""
        CREATE TABLE IF NOT EXISTS scraped_results (
            result_id    TEXT PRIMARY KEY,
            task_id      TEXT NOT NULL,
            goal_id      TEXT,
            data         TEXT NOT NULL,   -- JSON
            produced_at  TEXT NOT NULL
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_results_task ON scraped_results(task_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_results_goal ON scraped_results(goal_id)"
    )
    await db.commit()


async def _save_result(result_id: str, task_id: str, goal_id: str,
                       data: dict, produced_at: str) -> None:
    async with aiosqlite.connect(_DB_PATH) as db:
        await _ensure_results_table(db)
        await db.execute(
            """INSERT OR REPLACE INTO scraped_results
               (result_id, task_id, goal_id, data, produced_at)
               VALUES (?, ?, ?, ?, ?)""",
            (result_id, task_id, goal_id, json.dumps(data), produced_at),
        )
        await db.commit()


async def _load_results(limit: int = 100) -> List[dict]:
    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await _ensure_results_table(db)
        async with db.execute(
            "SELECT * FROM scraped_results ORDER BY produced_at DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [
        {
            "result_id": r["result_id"],
            "task_id": r["task_id"],
            "goal_id": r["goal_id"],
            "data": json.loads(r["data"]),
            "produced_at": r["produced_at"],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Global bot instance + goal tracking
# ---------------------------------------------------------------------------

_bot: Optional[MainBot] = None
# goal_id -> {"description": str, "submitted_at": str, "status": str}
_goals: Dict[str, dict] = {}


async def _result_listener(bot: MainBot) -> None:
    """Capture SUB_BOT_RESULT events and persist them to SQLite."""
    async for msg in bot.bus.subscribe(EventType.SUB_BOT_RESULT):
        try:
            p = msg.payload
            result_id = p.get("result_id") or str(uuid.uuid4())
            task_id = p.get("task_id", "")
            # Resolve goal_id from any in-flight goal (best-effort)
            goal_id = p.get("goal_id", "")
            data = p.get("data", p)
            produced_at = p.get("produced_at") or datetime.now(timezone.utc).isoformat()
            asyncio.create_task(
                _save_result(result_id, task_id, goal_id, data, produced_at)
            )
            log.info("api: captured SUB_BOT_RESULT", extra={"result_id": result_id})
        except Exception as exc:
            log.error("api: error persisting result", extra={"error": str(exc)})


# ---------------------------------------------------------------------------
# FastAPI lifespan – start/stop bot
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bot, _DB_PATH

    config = load_config()
    _DB_PATH = config.database.sqlite_path

    _bot = MainBot(config=config)
    await _bot.start()

    # Start result capture listener
    asyncio.create_task(_result_listener(_bot), name="api_result_listener")

    log.info("api: Giga AI bot started — ready for goals")
    yield

    log.info("api: shutting down Giga AI bot")
    await _bot.stop()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Giga AI",
    description="Always-on REST interface to the Giga AI bot system.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GoalRequest(BaseModel):
    description: str
    metadata: Optional[Dict[str, Any]] = None


class GoalResponse(BaseModel):
    goal_id: str
    description: str
    submitted_at: str
    status: str


class ManagerStatusResponse(BaseModel):
    manager_id: str
    task_id: Optional[str]
    status: str
    active_sub_bots: int
    error_count: int
    last_heartbeat: str


class StatusResponse(BaseModel):
    active_managers: int
    managers: List[ManagerStatusResponse]
    pending_goals: int
    pending_tasks: int


class ResultItem(BaseModel):
    result_id: str
    task_id: str
    goal_id: str
    data: Dict[str, Any]
    produced_at: str


class ResultsResponse(BaseModel):
    count: int
    results: List[ResultItem]


class HealthResponse(BaseModel):
    status: str
    bot_running: bool
    uptime_check: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health():
    """Railway health check — always returns 200 while the process is alive."""
    return HealthResponse(
        status="ok",
        bot_running=_bot is not None and _bot._running,
        uptime_check=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/goal", response_model=GoalResponse, status_code=202, tags=["bot"])
async def submit_goal(req: GoalRequest):
    """Submit a high-level goal to the bot. Returns immediately; processing is async."""
    if not _bot or not _bot._running:
        raise HTTPException(status_code=503, detail="Bot is not running")

    goal = await _bot.submit_goal(req.description, metadata=req.metadata)
    submitted_at = goal.submitted_at.isoformat()

    _goals[goal.goal_id] = {
        "description": req.description,
        "submitted_at": submitted_at,
        "status": "processing",
    }

    return GoalResponse(
        goal_id=goal.goal_id,
        description=req.description,
        submitted_at=submitted_at,
        status="processing",
    )


@app.get("/status", response_model=StatusResponse, tags=["bot"])
async def get_status():
    """Return current active manager statuses."""
    if not _bot:
        raise HTTPException(status_code=503, detail="Bot is not running")

    managers = []
    for ms in _bot.execution.get_manager_statuses():
        managers.append(ManagerStatusResponse(
            manager_id=ms.manager_id,
            task_id=ms.task_id,
            status=ms.status.value,
            active_sub_bots=ms.active_sub_bots,
            error_count=ms.error_count,
            last_heartbeat=ms.last_heartbeat.isoformat(),
        ))

    return StatusResponse(
        active_managers=len(managers),
        managers=managers,
        pending_goals=_bot._pending_goals,
        pending_tasks=_bot._pending_tasks,
    )


@app.post("/orchestrate", response_model=OrchestrateResponse, status_code=202, tags=["gateway"])
async def orchestrate(
    req: OrchestrateRequest,
    x_giga_secret: Optional[str] = Header(None, alias="X-Giga-Secret"),
):
    """
    Receive a complex goal from the almcp gateway and orchestrate it.

    The gateway POSTs here when combo.brain.run detects a multi-step goal
    that benefits from Gigia's DAG execution, retries, and replan-on-failure.
    Gigia calls back to req.callback.execute_url to run each skill, metered
    by the gateway. Returns immediately; execution runs asynchronously.
    """
    if not _bot or not _bot._running:
        raise HTTPException(status_code=503, detail="Orchestrator is not running")

    # Verify shared secret if configured
    config = _bot.config
    expected_secret = getattr(getattr(config, "gateway", None), "shared_secret", "")
    if expected_secret and x_giga_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Invalid X-Giga-Secret")

    # Build the goal metadata so MainBot._on_goal_received_loop passes
    # candidates + callback through to the PlanningBrain.
    goal_metadata = {
        "run_id": req.run_id,
        "candidates": [c.model_dump() for c in req.candidates],
        "callback": req.callback.model_dump(),
        "input": req.input,
        "max_credits": req.max_credits,
        "_skill_mode": True,
    }

    goal = await _bot.submit_goal(req.task, metadata=goal_metadata)

    _goals[goal.goal_id] = {
        "description": req.task,
        "submitted_at": goal.submitted_at.isoformat(),
        "status": "orchestrating",
        "run_id": req.run_id,
    }

    log.info(
        "api: orchestrate accepted",
        extra={"run_id": req.run_id, "goal_id": goal.goal_id, "task": req.task[:80]},
    )

    return OrchestrateResponse(run_id=req.run_id, status="accepted")


@app.get("/results", response_model=ResultsResponse, tags=["bot"])
async def get_results(limit: int = 100):
    """Return scraped results stored in the database (most recent first)."""
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")

    rows = await _load_results(limit=limit)
    return ResultsResponse(
        count=len(rows),
        results=[ResultItem(**r) for r in rows],
    )


@app.get("/skills", tags=["brain"])
async def get_skills():
    """Return all skill profiles SkillBrain has learned from real executions."""
    profiles = await _bot.skill_memory.get_all_profiles()
    return {"count": len(profiles), "skills": profiles}
