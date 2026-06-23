"""
synthesis_brain.py – The sixth brain: synthesizes task results into a final answer
and delivers it back to the caller.

Responsibilities
----------------
- Listen for GOAL_COMPLETED events.
- Load all raw task results for that goal from the results DB.
- Call gpt-4o to synthesize a coherent, human-readable answer.
- If the goal was skill-mode (came via /orchestrate), POST the synthesized
  answer back to the gateway result_url so the MCP client can retrieve it.

This closes the loop: goals go in via almcp → Gigia executes → answer comes back.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
import aiosqlite

from giga_ai.messaging.event_bus import EventBus, EventType
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)

_SYNTHESIS_SYSTEM_PROMPT = """You are a synthesis engine for an AI agent system.
Multiple tasks ran in parallel to accomplish a goal. Your job is to combine their
raw results into a single, clear, useful answer for the person who asked.

Rules:
- Be direct and specific — use the actual data from the results
- Organise naturally (bullet lists, short sections if needed)
- Do NOT mention the tasks, the system, or how the answer was produced
- If results are empty or all failed, say honestly what you could not find
- Keep it concise but complete"""

_SYNTHESIS_USER_TEMPLATE = """GOAL: {goal}

RAW TASK RESULTS:
{results}

Synthesize these into a clear, direct answer to the goal."""


class SynthesisBrain:
    """
    Sixth brain: turns raw task results into a delivered answer.

    Parameters
    ----------
    event_bus:
        Shared EventBus.
    llm_client:
        LLM client for synthesis calls.
    db_path:
        Path to the SQLite results database.
    """

    def __init__(self, event_bus: EventBus, llm_client, db_path: str) -> None:
        self._bus = event_bus
        self._llm = llm_client
        self._db_path = db_path
        self._running = False
        self._listener_task: Optional[asyncio.Task] = None

        # goal_id → {result_url, token, goal_description} for skill-mode goals
        self._pending: Dict[str, dict] = {}
        # goal_id → list of skill result dicts (from SUB_BOT_RESULT events)
        self._skill_results: Dict[str, List[dict]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Subscribe to GOAL_RECEIVED to capture result_url before execution starts
        self._goal_recv_task = asyncio.create_task(
            self._goal_received_listener(), name="synthesis_goal_recv"
        )
        self._listener_task = asyncio.create_task(
            self._goal_completed_listener(), name="synthesis_goal_completed"
        )
        self._skill_result_task = asyncio.create_task(
            self._skill_result_listener(), name="synthesis_skill_results"
        )
        log.info("SynthesisBrain started")

    async def stop(self) -> None:
        self._running = False
        for t in [self._goal_recv_task, self._listener_task, self._skill_result_task]:
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        log.info("SynthesisBrain stopped")

    # ------------------------------------------------------------------
    # Event listeners
    # ------------------------------------------------------------------

    async def _goal_received_listener(self) -> None:
        """Capture result_url and token for skill-mode goals before they run."""
        async for msg in self._bus.subscribe(EventType.GOAL_RECEIVED):
            if not self._running:
                break
            try:
                payload = msg.payload
                goal_id = payload.get("goal_id", "")
                metadata = payload.get("metadata") or {}
                if metadata.get("_skill_mode") and "callback" in metadata:
                    cb = metadata["callback"]
                    result_url = cb.get("result_url", "")
                    token = cb.get("token", "")
                    # run_id lives at metadata level, not inside callback
                    run_id = metadata.get("run_id") or goal_id
                    if result_url:
                        self._pending[goal_id] = {
                            "result_url": result_url,
                            "token": token,
                            "run_id": run_id,
                            "goal_description": payload.get("description", ""),
                        }
                        log.debug(
                            "SynthesisBrain: registered skill-mode goal",
                            extra={"goal_id": goal_id, "result_url": result_url},
                        )
            except Exception as exc:
                log.error("SynthesisBrain: error on GOAL_RECEIVED", extra={"error": str(exc)})

    async def _goal_completed_listener(self) -> None:
        """Synthesize and deliver when a goal finishes."""
        async for msg in self._bus.subscribe(EventType.GOAL_COMPLETED):
            if not self._running:
                break
            try:
                goal_id = msg.payload.get("goal_id", "")
                goal_desc = msg.payload.get("goal_description", "")
                asyncio.create_task(
                    self._synthesize_and_deliver(goal_id, goal_desc),
                    name=f"synthesis_{goal_id[:8]}",
                )
            except Exception as exc:
                log.error("SynthesisBrain: error on GOAL_COMPLETED", extra={"error": str(exc)})

    # ------------------------------------------------------------------
    # Core synthesis + delivery
    # ------------------------------------------------------------------

    async def _synthesize_and_deliver(self, goal_id: str, goal_desc: str) -> None:
        try:
            results = await self._load_results(goal_id)

            if not results:
                synthesized = f"No results were produced for this goal: {goal_desc}"
            else:
                synthesized = await self._synthesize(goal_desc, results)

            log.info(
                "SynthesisBrain: synthesized answer",
                extra={"goal_id": goal_id, "answer_len": len(synthesized)},
            )

            # Store locally so /results can show synthesized answers too
            await self._save_synthesis(goal_id, goal_desc, synthesized)

            # Deliver to almcp if this was a skill-mode goal
            cb = self._pending.pop(goal_id, None)
            if cb and cb.get("result_url"):
                await self._deliver(
                    result_url=cb["result_url"],
                    token=cb["token"],
                    run_id=cb["run_id"],
                    goal_description=goal_desc,
                    synthesized_answer=synthesized,
                    raw_results=results,
                )
        except Exception as exc:
            log.error(
                "SynthesisBrain: synthesis/delivery failed",
                extra={"goal_id": goal_id, "error": str(exc)},
            )

    async def _synthesize(self, goal: str, results: List[dict]) -> str:
        """Call the LLM to produce a coherent answer from raw task results."""
        results_str = "\n\n".join(
            f"[Task: {r.get('task_id', '?')}]\n{_format_result(r.get('data', {}))}"
            for r in results
        )
        prompt = _SYNTHESIS_USER_TEMPLATE.format(
            goal=goal,
            results=results_str[:12_000],  # cap to avoid token overflow
        )
        return await self._llm.complete(prompt, system_prompt=_SYNTHESIS_SYSTEM_PROMPT)

    async def _deliver(
        self,
        result_url: str,
        token: str,
        run_id: str,
        goal_description: str,
        synthesized_answer: str,
        raw_results: List[dict],
    ) -> None:
        """POST the synthesized answer back to almcp."""
        payload = {
            "run_id": run_id,
            "status": "completed",
            "answer": synthesized_answer,
            "goal": goal_description,
            "result_count": len(raw_results),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    result_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status in (200, 201):
                        log.info(
                            "SynthesisBrain: result delivered",
                            extra={"run_id": run_id, "result_url": result_url},
                        )
                    else:
                        body = await resp.text()
                        log.warning(
                            "SynthesisBrain: delivery returned non-200",
                            extra={"status": resp.status, "body": body[:200]},
                        )
        except Exception as exc:
            log.error(
                "SynthesisBrain: delivery failed",
                extra={"run_id": run_id, "error": str(exc)},
            )

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    async def _skill_result_listener(self) -> None:
        """Cache skill execution results by goal_id for synthesis."""
        async for msg in self._bus.subscribe(EventType.SUB_BOT_RESULT):
            if not self._running:
                break
            try:
                p = msg.payload
                goal_id = p.get("goal_id", "")
                if not goal_id:
                    continue
                data = p.get("data") or p
                task_id = p.get("task_id", "")
                self._skill_results.setdefault(goal_id, []).append(
                    {"task_id": task_id, "data": data}
                )
            except Exception as exc:
                log.warning("SynthesisBrain: error on SUB_BOT_RESULT", extra={"error": str(exc)})

    async def _load_results(self, goal_id: str) -> List[dict]:
        # Skill-mode results arrive via SUB_BOT_RESULT events and are cached in memory.
        skill_results = self._skill_results.pop(goal_id, [])
        if skill_results:
            return skill_results

        # Native-mode: fall back to scraped_results SQLite table.
        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                rows = await (await db.execute(
                    "SELECT task_id, data FROM scraped_results WHERE goal_id = ? ORDER BY produced_at",
                    (goal_id,),
                )).fetchall()
            return [
                {"task_id": r["task_id"], "data": json.loads(r["data"])}
                for r in rows
            ]
        except Exception as exc:
            log.warning("SynthesisBrain: could not load results", extra={"error": str(exc)})
            return []

    async def _save_synthesis(self, goal_id: str, goal_desc: str, answer: str) -> None:
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS goal_syntheses (
                        goal_id     TEXT PRIMARY KEY,
                        description TEXT,
                        answer      TEXT,
                        created_at  TEXT
                    )
                """)
                await db.execute(
                    "INSERT OR REPLACE INTO goal_syntheses VALUES (?,?,?,?)",
                    (goal_id, goal_desc, answer, datetime.now(timezone.utc).isoformat()),
                )
                await db.commit()
        except Exception as exc:
            log.warning("SynthesisBrain: could not save synthesis", extra={"error": str(exc)})


def _format_result(data: dict) -> str:
    """Convert a result dict to a compact readable string."""
    # For skill results, the useful data is usually under 'result' key
    if "result" in data and data["result"] is not None:
        return json.dumps(data["result"], indent=2)[:3000]
    # Otherwise show the top-level keys with truncated values
    return json.dumps(
        {k: (str(v)[:200] if isinstance(v, str) else v) for k, v in list(data.items())[:10]},
        indent=2,
    )[:3000]
