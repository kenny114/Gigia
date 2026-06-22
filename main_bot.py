"""
main_bot.py – The Main Bot head. Wire up the four brains here.

This is YOUR file to customise. The brains are fully built — you just
need to initialise them, connect them together, and decide what goals
to feed in.

Quick-start
-----------
1.  Set your OpenAI key:
        export OPENAI_API_KEY=sk-...
    or edit config.yaml → llm.api_key

2.  Run:
        python main_bot.py

3.  Pass any high-level goal string to `main_bot.submit_goal(...)`.

Architecture reminder
---------------------
                        ┌─────────────────────┐
                        │      MAIN BOT        │
                        │   (this file)        │
    ┌───────────────┐   │  ┌───────────────┐  │
    │  EventBus     │◄──┼──│ PerceptionBrain│  │  ← receives goals + health
    │  (shared bus) │   │  └───────┬───────┘  │
    └───────┬───────┘   │          │ GOAL_RECEIVED event
            │           │  ┌───────▼───────┐  │
            │           │  │ PlanningBrain │  │  ← LLM decomposes goal → tasks
            │           │  └───────┬───────┘  │
            │           │          │ TASK_CREATED events
            │           │  ┌───────▼───────┐  │
            │           │  │ExecutionBrain │  │  ← spawns + monitors ManagerBots
            │           │  └───────┬───────┘  │
            │           │          │ MANAGER_SPAWNED / ESCALATION events
            │           │  ┌───────▼───────┐  │
            │           │  │ LearningBrain │  │  ← learns from failures, updates memory
            │           │  └───────────────┘  │
            │           └─────────────────────┘
            │
     ManagerBot (per task)
            │
       SubBots (workers)
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Optional

from giga_ai.brains.execution_brain import ExecutionBrain
from giga_ai.brains.learning_brain import LearningBrain
from giga_ai.brains.perception_brain import PerceptionBrain
from giga_ai.brains.planning_brain import PlanningBrain
from giga_ai.brains.skill_brain import SkillBrain
from giga_ai.config import load_config
from giga_ai.memory.global_memory import GlobalMemory
from giga_ai.memory.skill_memory import SkillMemory
from giga_ai.messaging.event_bus import EventBus, EventType
from giga_ai.messaging.message_schemas import (
    GatewayCallback,
    Goal,
    OrchestrateCandidate,
    Task,
)
from giga_ai.utils.llm_client import get_llm_client
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# MainBot
# ---------------------------------------------------------------------------

class MainBot:
    """
    Top-level controller. Holds the four brains and routes events between them.

    Usage
    -----
        bot = MainBot()
        await bot.start()
        await bot.submit_goal("Find high-quality leads in the restaurant industry")
        await bot.run_until_done()   # or await bot.stop() whenever you like
    """

    def __init__(self, config=None) -> None:
        # ── Config ──────────────────────────────────────────────────────────
        self.config = config or load_config()

        # ── Shared infrastructure ───────────────────────────────────────────
        self.bus = EventBus()
        self.memory = GlobalMemory(self.config.database.sqlite_path)
        self.skill_memory = SkillMemory(self.config.database.sqlite_path)
        self.llm = get_llm_client(self.config)

        # ── Five brains ─────────────────────────────────────────────────────
        self.perception = PerceptionBrain(
            event_bus=self.bus,
            health_poll_interval=10.0,
        )
        self.skill_brain = SkillBrain(
            event_bus=self.bus,
            skill_memory=self.skill_memory,
        )
        self.planning = PlanningBrain(
            event_bus=self.bus,
            llm_client=self.llm,
            skill_brain=self.skill_brain,
        )
        self.execution = ExecutionBrain(
            event_bus=self.bus,
            memory=self.memory,
            config=self.config,
            monitor_interval=5.0,
        )
        self.learning = LearningBrain(
            event_bus=self.bus,
            memory=self.memory,
            config=self.config,
        )

        # Wire PerceptionBrain → ExecutionBrain (for health monitoring)
        self.perception.set_execution_brain(self.execution)

        # Background listener tasks
        self._goal_listener_task: Optional[asyncio.Task] = None
        self._task_listener_task: Optional[asyncio.Task] = None
        self._running = False

        # Tracks goals/tasks currently in-flight (submitted but not yet spawned as managers)
        self._pending_goals: int = 0
        self._pending_tasks: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise memory, start all brains, and start event listeners."""
        log.info("MainBot: starting up")

        # Init DBs
        await self.memory.init()
        await self.skill_memory.init()

        # Start brains
        await self.perception.start()
        await self.skill_brain.start()
        await self.execution.start()
        await self.learning.start()
        # PlanningBrain has no background loop — it's called on demand

        # Start internal event listeners
        self._goal_listener_task = asyncio.create_task(
            self._on_goal_received_loop(), name="main_bot_goal_listener"
        )
        self._task_listener_task = asyncio.create_task(
            self._on_task_created_loop(), name="main_bot_task_listener"
        )

        self._running = True
        # Yield to the event loop so all listener tasks start iterating before
        # any goal is submitted — prevents events firing before subscribers are ready
        await asyncio.sleep(0.1)
        log.info("MainBot: all systems running")

    async def stop(self) -> None:
        """Gracefully shut down all brains and listeners."""
        log.info("MainBot: shutting down")
        self._running = False

        for t in [self._goal_listener_task, self._task_listener_task]:
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        await self.perception.stop()
        await self.skill_brain.stop()
        await self.execution.stop()
        await self.learning.stop()
        await self.bus.shutdown()
        log.info("MainBot: shutdown complete")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit_goal(self, goal_description: str, metadata: Optional[dict] = None) -> Goal:
        """
        Feed a high-level goal into the system.

        The PerceptionBrain wraps it in a Goal, publishes GOAL_RECEIVED,
        which this bot picks up to run the PlanningBrain, which in turn
        emits TASK_CREATED events for the ExecutionBrain to act on.

        Parameters
        ----------
        goal_description:
            Plain-English goal, e.g. "Find restaurant leads in NYC".
        metadata:
            Optional extra context forwarded to the planning prompt.
        """
        self._pending_goals += 1
        return await self.perception.submit_goal(goal_description, metadata=metadata)

    async def run_until_done(self, timeout: float = 300.0) -> None:
        """
        Block until all spawned managers have finished AND the pipeline is idle.

        The pipeline goes through three stages:
          1. Goal submitted  → pending_goals > 0 (LLM is decomposing)
          2. Tasks created   → pending_tasks > 0 (managers being spawned)
          3. Managers active → execution.get_manager_statuses() has entries

        We only declare "done" when ALL THREE are empty simultaneously.
        """
        log.info("MainBot: waiting for all managers to complete", extra={"timeout": timeout})
        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            if asyncio.get_event_loop().time() > deadline:
                raise asyncio.TimeoutError("MainBot.run_until_done timed out")

            pipeline_busy = (
                self._pending_goals > 0
                or self._pending_tasks > 0
                or len(list(self.execution.get_manager_statuses())) > 0
            )

            if not pipeline_busy:
                # Double-check after a brief yield to catch last-moment spawns
                await asyncio.sleep(2)
                pipeline_busy = (
                    self._pending_goals > 0
                    or self._pending_tasks > 0
                    or len(list(self.execution.get_manager_statuses())) > 0
                )
                if not pipeline_busy:
                    break

            await asyncio.sleep(2)

        log.info("MainBot: all managers done")

    # ------------------------------------------------------------------
    # Internal event handlers
    # ------------------------------------------------------------------

    async def _on_goal_received_loop(self) -> None:
        """
        Listen for GOAL_RECEIVED events, pull best-practice context from
        memory, and ask PlanningBrain to decompose the goal into tasks.
        """
        async for message in self.bus.subscribe(EventType.GOAL_RECEIVED):
            if not self._running:
                break
            try:
                goal = Goal(**message.payload)

                # Enrich context with memory best practices
                best_practices = await self.learning.query_best_practices(goal.description)
                context = {
                    "best_practices": [
                        {"problem": e.problem, "solution": e.solution, "success_rate": e.success_rate}
                        for e in best_practices
                    ]
                }

                log.info(
                    "MainBot: planning for goal",
                    extra={"goal_id": goal.goal_id, "best_practice_count": len(best_practices)},
                )

                # Skill-mode: gateway sent candidates + callback in metadata
                skill_mode = goal.metadata.get("_skill_mode", False)
                candidates = None
                gateway_callback = None
                if skill_mode:
                    try:
                        candidates = [
                            OrchestrateCandidate(**c)
                            for c in goal.metadata.get("candidates", [])
                        ]
                        gateway_callback = GatewayCallback(**goal.metadata["callback"])
                    except Exception as exc:
                        log.warning(
                            "MainBot: failed to parse skill-mode metadata — falling back",
                            extra={"error": str(exc)},
                        )
                        candidates = None
                        gateway_callback = None

                async def _plan_and_decrement(g, ctx, cands, gw_cb):
                    try:
                        await self.planning.decompose_goal(
                            g,
                            extra_context=ctx,
                            candidates=cands,
                            gateway_callback=gw_cb,
                        )
                    except Exception as exc:
                        log.error("MainBot: decompose_goal failed", extra={"error": type(exc).__name__, "detail": str(exc)})
                    finally:
                        self._pending_goals = max(0, self._pending_goals - 1)

                asyncio.create_task(
                    _plan_and_decrement(goal, context, candidates, gateway_callback),
                    name=f"plan_{goal.goal_id[:8]}",
                )
            except Exception as exc:
                self._pending_goals = max(0, self._pending_goals - 1)
                log.error("MainBot: error handling GOAL_RECEIVED", extra={"error": str(exc)})

    async def _on_task_created_loop(self) -> None:
        """
        Listen for TASK_CREATED events and spawn a ManagerBot for each task.
        """
        async for message in self.bus.subscribe(EventType.TASK_CREATED):
            if not self._running:
                break
            try:
                task = Task(**message.payload)
                self._pending_tasks += 1

                # Fetch memory context relevant to this task
                memory_context_entries = await self.learning.query_best_practices(task.description)
                memory_context = {
                    e.problem: e.solution for e in memory_context_entries
                }

                log.info(
                    "MainBot: spawning manager for task",
                    extra={"task_id": task.task_id, "title": task.title},
                )

                async def _spawn_and_decrement(t, ctx):
                    try:
                        await self.execution.spawn_manager(t, memory_context=ctx)
                    finally:
                        self._pending_tasks = max(0, self._pending_tasks - 1)

                asyncio.create_task(
                    _spawn_and_decrement(task, memory_context),
                    name=f"spawn_{task.task_id[:8]}",
                )
            except Exception as exc:
                self._pending_tasks = max(0, self._pending_tasks - 1)
                log.error("MainBot: error handling TASK_CREATED", extra={"error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point — customise the goal below and run: python main_bot.py
# ---------------------------------------------------------------------------

async def main() -> None:
    bot = MainBot()
    await bot.start()

    # ── CHANGE THIS GOAL TO WHATEVER YOU WANT ──────────────────────────────
    goal_description = "Find high-quality leads in the restaurant industry"
    # ───────────────────────────────────────────────────────────────────────

    await bot.submit_goal(goal_description)

    try:
        await bot.run_until_done(timeout=300.0)
    except asyncio.TimeoutError:
        log.warning("MainBot: run_until_done timed out — stopping anyway")
    finally:
        await bot.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
