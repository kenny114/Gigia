"""
learning_brain.py – Global memory management, failure analysis, and strategy learning.

Responsibilities
----------------
- Receive ``EscalationReport`` objects from ManagerBots that hit unsolvable problems.
- Spawn a specialised ManagerBot to empirically test solutions.
- Update GlobalMemory with problem→solution mappings and success rates.
- Answer queries for best practices given a context string.
- Emit ``STRATEGY_LEARNED`` events when a new strategy is confirmed.

Usage
-----
    brain = LearningBrain(event_bus=bus, memory=global_memory, config=cfg)
    await brain.start()

    # Called by ExecutionBrain / ManagerBot when escalation arrives
    await brain.handle_escalation(report)

    entries = await brain.query_best_practices("proxy blocked")
    await brain.update_memory("proxy blocked", "rotate proxy", 0.9)

    await brain.stop()
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from giga_ai.memory.global_memory import GlobalMemory
from giga_ai.messaging.event_bus import EventBus, EventType
from giga_ai.messaging.message_schemas import EscalationReport, MemoryEntry, Task, SubBotType
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)


class LearningBrain:
    """
    Layer-1 brain: global memory and failure analysis.

    Parameters
    ----------
    event_bus:
        Shared EventBus instance.
    memory:
        GlobalMemory instance (must be initialised before use).
    config:
        Config override; loaded from singleton if not supplied.
    """

    def __init__(
        self,
        event_bus: EventBus,
        memory: GlobalMemory,
        config=None,
    ) -> None:
        self._bus = event_bus
        self._memory = memory
        self._escalation_listener_task: Optional[asyncio.Task] = None
        self._running = False

        if config is None:
            from giga_ai.config import get_config
            config = get_config()
        self._config = config

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to ESCALATION events and start processing them."""
        if self._running:
            return
        self._running = True
        self._escalation_listener_task = asyncio.create_task(
            self._escalation_listener(), name="learning_brain_escalation_listener"
        )
        log.info("LearningBrain started")

    async def stop(self) -> None:
        """Stop the escalation listener."""
        self._running = False
        if self._escalation_listener_task and not self._escalation_listener_task.done():
            self._escalation_listener_task.cancel()
            try:
                await self._escalation_listener_task
            except asyncio.CancelledError:
                pass
        log.info("LearningBrain stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_escalation(self, report: EscalationReport) -> None:
        """
        Process an escalation report from a ManagerBot.

        Steps:
        1. Log the failure in GlobalMemory.
        2. Check if a known strategy exists for the problem.
        3. If yes, update it and notify the original manager.
        4. If no, spawn a problem-solver manager to discover a fix.

        Parameters
        ----------
        report:
            The escalation report emitted by the failing ManagerBot.
        """
        log.info(
            "LearningBrain: received escalation",
            extra={
                "escalation_id": report.escalation_id,
                "manager_id": report.manager_id,
                "problem": report.problem[:120],
            },
        )

        # Always log the failure
        await self._memory.log_failure(
            problem=report.problem,
            context={
                "manager_id": report.manager_id,
                "task_id": report.task_id,
                **report.context,
            },
            error_details={
                "error_count": len(report.error_history),
                "last_error": (
                    report.error_history[-1].model_dump(mode="json")
                    if report.error_history
                    else None
                ),
            },
        )

        # Check for an existing strategy
        existing = await self._memory.get_strategy(report.problem)
        if existing and existing.success_rate > 0.5:
            log.info(
                "LearningBrain: found existing high-confidence strategy",
                extra={
                    "problem": report.problem,
                    "solution": existing.solution,
                    "success_rate": existing.success_rate,
                },
            )
            # Publish so the requesting manager can act on it
            await self._bus.publish(
                EventType.STRATEGY_LEARNED,
                payload={
                    "escalation_id": report.escalation_id,
                    "problem": report.problem,
                    "solution": existing.solution,
                    "success_rate": existing.success_rate,
                    "source": "existing_memory",
                },
                correlation_id=report.correlation_id,
            )
            return

        # No known strategy – spawn a problem-solver
        await self.spawn_problem_solver(
            problem=report.problem,
            context=report.context,
            correlation_id=report.correlation_id,
            escalation_id=report.escalation_id,
        )

    async def spawn_problem_solver(
        self,
        problem: str,
        context: Optional[dict] = None,
        correlation_id: Optional[str] = None,
        escalation_id: Optional[str] = None,
    ) -> None:
        """
        Create a specialised ManagerBot whose sole job is to test solutions
        for *problem* and report back.

        The problem-solver manager runs as a background asyncio task.
        On success it calls ``update_memory`` with the working solution.

        Parameters
        ----------
        problem:
            Natural-language description of the problem to solve.
        context:
            Additional context dict forwarded to the manager.
        correlation_id:
            Optional trace ID.
        escalation_id:
            ID of the originating escalation (used for correlation).
        """
        from giga_ai.manager_bot.manager_bot import ManagerBot

        log.info(
            "LearningBrain: spawning problem-solver manager",
            extra={"problem": problem[:120]},
        )

        # Build a synthetic Task for the problem-solver
        solver_task = Task(
            goal_id=f"solver_{escalation_id or 'anon'}",
            title=f"Solve: {problem[:60]}",
            description=(
                f"Empirically test solutions for the following problem and report "
                f"the most effective one back:\n\n{problem}"
            ),
            sub_bot_type=SubBotType.GENERIC,
            priority=0,
            correlation_id=correlation_id or "",
            metadata={"is_problem_solver": True, "original_problem": problem, **(context or {})},
        )

        manager = ManagerBot(
            task=solver_task,
            event_bus=self._bus,
            memory=self._memory,
            memory_context=context or {},
            config=self._config,
            on_complete_callback=self._on_problem_solver_complete,
        )

        asyncio.create_task(
            manager.run(),
            name=f"problem_solver_{manager.manager_id[:8]}",
        )

    async def update_memory(
        self,
        problem: str,
        solution: str,
        success_rate: float,
    ) -> MemoryEntry:
        """
        Store or update a problem→solution strategy in GlobalMemory and
        emit a STRATEGY_LEARNED event.

        Parameters
        ----------
        problem:
            Natural-language problem description.
        solution:
            Natural-language solution description.
        success_rate:
            Observed success rate in [0, 1].

        Returns
        -------
        MemoryEntry
            The Pydantic model corresponding to the persisted record.
        """
        record = await self._memory.store_strategy(
            problem=problem,
            solution=solution,
            success_rate=success_rate,
        )

        entry = MemoryEntry(
            entry_id=record.entry_id,
            problem=record.problem,
            solution=record.solution,
            success_rate=record.success_rate,
            created_at=record.created_at,
            updated_at=record.updated_at,
            tags=record.tags,
            metadata=record.metadata,
        )

        log.info(
            "LearningBrain: memory updated",
            extra={
                "problem": problem[:80],
                "solution": solution[:80],
                "success_rate": success_rate,
            },
        )

        await self._bus.publish(
            EventType.STRATEGY_LEARNED,
            payload=entry.model_dump(mode="json"),
        )

        return entry

    async def query_best_practices(self, context: str) -> List[MemoryEntry]:
        """
        Return the most relevant MemoryEntry records for the given context.

        Uses keyword matching against the stored problem/solution fields.
        Returns all entries sorted by success_rate descending if no
        keyword matches are found.

        Parameters
        ----------
        context:
            A string describing the current situation or problem.

        Returns
        -------
        List[MemoryEntry]
            Matching memory entries, best first.
        """
        # Try keyword search first
        results = await self._memory.query_strategies_by_keyword(context)
        if results:
            log.debug(
                "LearningBrain: best-practices query returned results",
                extra={"context": context[:80], "count": len(results)},
            )
            return results

        # Fallback: return top-N by success rate
        all_entries = await self._memory.get_all_strategies()
        top = all_entries[:10]
        log.debug(
            "LearningBrain: best-practices fallback to top-N",
            extra={"context": context[:80], "count": len(top)},
        )
        return top

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _escalation_listener(self) -> None:
        """Listen for ESCALATION events on the bus and dispatch them."""
        async for message in self._bus.subscribe(EventType.ESCALATION):
            if not self._running:
                break
            try:
                report = EscalationReport(**message.payload)
                asyncio.create_task(
                    self.handle_escalation(report),
                    name=f"escalation_{report.escalation_id[:8]}",
                )
            except Exception as exc:
                log.error(
                    "LearningBrain: failed to parse escalation message",
                    extra={"error": str(exc), "payload": str(message.payload)[:300]},
                )

    async def _on_problem_solver_complete(
        self,
        manager_id: str,
        problem: str,
        solution: str,
        success_rate: float,
    ) -> None:
        """Callback invoked by a problem-solver ManagerBot when it finishes."""
        log.info(
            "LearningBrain: problem-solver completed",
            extra={"manager_id": manager_id, "problem": problem[:80]},
        )
        if solution and success_rate > 0.0:
            await self.update_memory(problem, solution, success_rate)
