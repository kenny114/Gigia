"""
execution_brain.py – ManagerBot lifecycle management.

Responsibilities
----------------
- Spawn ManagerBot instances for given Tasks.
- Track all active managers in a dict keyed by manager_id.
- Periodically poll managers to detect crashes or completions.
- Stop individual managers on demand.
- Emit MANAGER_SPAWNED / MANAGER_CRASHED / MANAGER_COMPLETED events.

Usage
-----
    bus = EventBus()
    brain = ExecutionBrain(event_bus=bus, memory=global_memory)
    await brain.start()

    manager = await brain.spawn_manager(task, memory_context={})
    await brain.stop_manager(manager.manager_id)

    await brain.stop()
"""

from __future__ import annotations

import asyncio
from typing import Dict, Iterable, List, Optional

from giga_ai.messaging.event_bus import EventBus, EventType
from giga_ai.messaging.message_schemas import ManagerStatus, ManagerStatusEnum, Task
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)


class ExecutionBrain:
    """
    Layer-1 brain: ManagerBot lifecycle management.

    Parameters
    ----------
    event_bus:
        Shared EventBus instance.
    memory:
        GlobalMemory instance passed down to each spawned ManagerBot.
    config:
        Optional Config override; loaded from singleton if not supplied.
    monitor_interval:
        Seconds between manager health checks.
    """

    def __init__(
        self,
        event_bus: EventBus,
        memory=None,
        config=None,
        monitor_interval: float = 5.0,
    ) -> None:
        self._bus = event_bus
        self._memory = memory
        self._monitor_interval = monitor_interval
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

        # manager_id → ManagerBot instance
        self._managers: Dict[str, "ManagerBot"] = {}  # type: ignore[name-defined]
        # manager_id → asyncio.Task running manager.run()
        self._manager_tasks: Dict[str, asyncio.Task] = {}
        # goal_id → set of manager_ids still running (for GOAL_COMPLETED tracking)
        self._goal_managers: Dict[str, set] = {}
        # goal_id → goal description (for SkillBrain context)
        self._goal_descriptions: Dict[str, str] = {}

        if config is None:
            from giga_ai.config import get_config
            config = get_config()
        self._config = config

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background manager-monitoring loop."""
        if self._running:
            return
        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="execution_brain_monitor"
        )
        log.info("ExecutionBrain started")

    async def stop(self) -> None:
        """Stop monitoring and gracefully shut down all active managers."""
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Stop all managers
        manager_ids = list(self._managers.keys())
        for mid in manager_ids:
            await self.stop_manager(mid)

        log.info("ExecutionBrain stopped")

    # ------------------------------------------------------------------
    # Manager management
    # ------------------------------------------------------------------

    async def spawn_manager(
        self,
        task: Task,
        memory_context: Optional[dict] = None,
    ) -> "ManagerBot":  # type: ignore[name-defined]
        """
        Create and start a ManagerBot for *task*.

        Parameters
        ----------
        task:
            The Task this manager is responsible for.
        memory_context:
            Optional context dict from GlobalMemory (best practices, etc.).

        Returns
        -------
        ManagerBot
            The newly spawned (already running) manager instance.
        """
        from giga_ai.manager_bot.manager_bot import ManagerBot

        manager = ManagerBot(
            task=task,
            event_bus=self._bus,
            memory=self._memory,
            memory_context=memory_context or {},
            config=self._config,
        )

        self._managers[manager.manager_id] = manager

        # Track which goal this manager belongs to
        goal_id = task.goal_id
        if goal_id not in self._goal_managers:
            self._goal_managers[goal_id] = set()
            self._goal_descriptions[goal_id] = task.metadata.get("goal_description", task.title)
        self._goal_managers[goal_id].add(manager.manager_id)

        # Run the manager in a background asyncio task
        t = asyncio.create_task(
            manager.run(),
            name=f"manager_{manager.manager_id[:8]}",
        )
        self._manager_tasks[manager.manager_id] = t

        log.info(
            "ExecutionBrain: manager spawned",
            extra={"manager_id": manager.manager_id, "task_id": task.task_id},
        )

        await self._bus.publish(
            EventType.MANAGER_SPAWNED,
            payload={
                "manager_id": manager.manager_id,
                "task_id": task.task_id,
                "task_title": task.title,
            },
            correlation_id=task.correlation_id,
        )

        return manager

    async def stop_manager(self, manager_id: str) -> None:
        """
        Cancel and remove a manager by ID.

        Parameters
        ----------
        manager_id:
            ID of the manager to stop.
        """
        task_handle = self._manager_tasks.pop(manager_id, None)
        if task_handle and not task_handle.done():
            task_handle.cancel()
            try:
                await task_handle
            except (asyncio.CancelledError, Exception):
                pass

        manager = self._managers.pop(manager_id, None)
        if manager:
            log.info("ExecutionBrain: manager stopped", extra={"manager_id": manager_id})

    async def monitor_managers(self) -> List[ManagerStatus]:
        """
        Check all active managers for crashes or completions.

        Returns
        -------
        List[ManagerStatus]
            Current status snapshot for every tracked manager.
        """
        statuses: List[ManagerStatus] = []
        crashed_ids: List[str] = []
        completed_ids: List[str] = []

        for mid, manager in list(self._managers.items()):
            task_handle = self._manager_tasks.get(mid)
            status = manager.get_status()
            statuses.append(status)

            if task_handle and task_handle.done():
                exc = task_handle.exception() if not task_handle.cancelled() else None
                if exc is not None:
                    crashed_ids.append(mid)
                    log.error(
                        "ExecutionBrain: manager task raised exception",
                        extra={"manager_id": mid, "error": str(exc)},
                    )
                    await self._bus.publish(
                        EventType.MANAGER_CRASHED,
                        payload={"manager_id": mid, "error": str(exc)},
                    )
                elif status.status == ManagerStatusEnum.COMPLETED:
                    completed_ids.append(mid)
                    await self._bus.publish(
                        EventType.MANAGER_COMPLETED,
                        payload={"manager_id": mid},
                    )

        # Retire finished / crashed managers and check for goal completion
        for mid in crashed_ids + completed_ids:
            manager = self._managers.pop(mid, None)
            self._manager_tasks.pop(mid, None)

            # Check if all managers for a goal are now done
            if manager:
                goal_id = manager.task.goal_id
                if goal_id in self._goal_managers:
                    self._goal_managers[goal_id].discard(mid)
                    if not self._goal_managers[goal_id]:
                        # All tasks for this goal have finished
                        self._goal_managers.pop(goal_id, None)
                        goal_desc = self._goal_descriptions.pop(goal_id, "")
                        success = goal_id not in [m.task.goal_id for m in self._managers.values()]
                        await self._bus.publish(
                            EventType.GOAL_COMPLETED,
                            payload={
                                "goal_id": goal_id,
                                "goal_description": goal_desc,
                                "success": True,
                            },
                        )
                        log.info(
                            "ExecutionBrain: goal completed",
                            extra={"goal_id": goal_id},
                        )

        return statuses

    def get_manager_statuses(self) -> Iterable[ManagerStatus]:
        """Yield the current ManagerStatus for each tracked manager."""
        for manager in self._managers.values():
            yield manager.get_status()

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                await self.monitor_managers()
            except Exception as exc:
                log.error(
                    "ExecutionBrain: error in monitor loop",
                    extra={"error": str(exc)},
                )
            await asyncio.sleep(self._monitor_interval)
