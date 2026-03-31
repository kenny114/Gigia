"""
perception_brain.py – Goal intake and system health monitoring.

Responsibilities
----------------
- Accept high-level goal strings from the Main Bot and normalise them
  into ``Goal`` domain objects.
- Emit ``GOAL_RECEIVED`` events on the EventBus so the PlanningBrain
  (and any other subscriber) can react.
- Poll all active ManagerBots via ExecutionBrain to produce a
  ``SystemHealth`` snapshot.

Usage
-----
    bus = EventBus()
    brain = PerceptionBrain(event_bus=bus, execution_brain=exec_brain)
    await brain.start()

    goal = await brain.submit_goal("Scrape top-10 product prices from example.com")
    health = await brain.monitor_health()

    await brain.stop()
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from giga_ai.messaging.event_bus import EventBus, EventType
from giga_ai.messaging.message_schemas import Goal, ManagerStatusEnum, SystemHealth
from giga_ai.utils.logger import get_logger


log = get_logger(__name__)


class PerceptionBrain:
    """
    Layer-1 brain: goal intake and health monitoring.

    Parameters
    ----------
    event_bus:
        Shared EventBus instance.
    execution_brain:
        Reference to the ExecutionBrain (used for health polling).
        May be ``None`` during construction; set via ``set_execution_brain``.
    health_poll_interval:
        How often (in seconds) ``_health_loop`` polls manager statuses.
    """

    def __init__(
        self,
        event_bus: EventBus,
        execution_brain=None,
        health_poll_interval: float = 10.0,
    ) -> None:
        self._bus = event_bus
        self._execution_brain = execution_brain
        self._health_poll_interval = health_poll_interval
        self._health_task: Optional[asyncio.Task] = None
        self._running = False
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background health-monitoring loop."""
        if self._running:
            return
        self._running = True
        self._health_task = asyncio.create_task(self._health_loop(), name="perception_health_loop")
        self._logger.info("PerceptionBrain started")

    async def stop(self) -> None:
        """Stop the background health-monitoring loop gracefully."""
        self._running = False
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        self._logger.info("PerceptionBrain stopped")

    def set_execution_brain(self, execution_brain) -> None:
        """Inject the ExecutionBrain after construction (avoids circular deps)."""
        self._execution_brain = execution_brain

    # ------------------------------------------------------------------
    # Goal submission
    # ------------------------------------------------------------------

    async def submit_goal(self, goal_description: str, metadata: Optional[dict] = None) -> Goal:
        """
        Accept a high-level goal string, wrap it in a ``Goal`` object, and
        emit a ``GOAL_RECEIVED`` event on the bus.

        Parameters
        ----------
        goal_description:
            Natural-language goal text.
        metadata:
            Optional free-form metadata attached to the goal.

        Returns
        -------
        Goal
            The normalised, timestamped Goal object.
        """
        goal = Goal(
            description=goal_description.strip(),
            metadata=metadata or {},
        )

        self._logger.info(
            "PerceptionBrain: goal received",
            extra={"goal_id": goal.goal_id, "description": goal.description[:120]},
        )

        await self._bus.publish(
            EventType.GOAL_RECEIVED,
            payload={
                "goal_id": goal.goal_id,
                "description": goal.description,
                "submitted_at": goal.submitted_at.isoformat(),
                "metadata": goal.metadata,
            },
            correlation_id=goal.correlation_id,
        )

        return goal

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    async def monitor_health(self) -> SystemHealth:
        """
        Poll all active ManagerBots and return an aggregate SystemHealth snapshot.

        If no ExecutionBrain is wired up, an empty snapshot is returned.
        """
        if self._execution_brain is None:
            return SystemHealth()

        statuses = list(self._execution_brain.get_manager_statuses())

        total = len(statuses)
        running = sum(1 for s in statuses if s.status == ManagerStatusEnum.RUNNING)
        crashed = sum(1 for s in statuses if s.status == ManagerStatusEnum.CRASHED)
        completed = sum(1 for s in statuses if s.status == ManagerStatusEnum.COMPLETED)

        health = SystemHealth(
            total_managers=total,
            running_managers=running,
            crashed_managers=crashed,
            completed_managers=completed,
            manager_statuses=statuses,
            checked_at=datetime.now(timezone.utc),
        )

        self._logger.debug(
            "PerceptionBrain: health snapshot",
            extra={
                "total": total,
                "running": running,
                "crashed": crashed,
                "completed": completed,
            },
        )

        await self._bus.publish(
            EventType.HEALTH_UPDATE,
            payload=health.model_dump(mode="json"),
        )

        return health

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _health_loop(self) -> None:
        """Continuously poll health at the configured interval."""
        while self._running:
            try:
                await self.monitor_health()
            except Exception as exc:
                self._logger.error(
                    "PerceptionBrain: error during health poll",
                    extra={"error": str(exc)},
                )
            await asyncio.sleep(self._health_poll_interval)
