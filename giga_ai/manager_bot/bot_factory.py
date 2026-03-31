"""
bot_factory.py – Factory for creating ManagerBot instances.

The ``BotFactory`` provides a clean entry-point for the Main Bot (or any
other orchestrator) to obtain pre-configured ManagerBot instances without
having to wire all dependencies manually.

Usage
-----
    factory = BotFactory(event_bus=bus, memory=mem, config=cfg)
    manager = factory.create_manager(task)
    await manager.run()
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from giga_ai.messaging.event_bus import EventBus
from giga_ai.messaging.message_schemas import Task
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)


class BotFactory:
    """
    Factory for ManagerBot instances.

    Parameters
    ----------
    event_bus:
        Shared EventBus passed to every created manager.
    memory:
        GlobalMemory instance passed to every created manager.
    config:
        Config override; loaded from singleton if not supplied.
    """

    def __init__(
        self,
        event_bus: EventBus,
        memory=None,
        config=None,
    ) -> None:
        self._bus = event_bus
        self._memory = memory

        if config is None:
            from giga_ai.config import get_config
            config = get_config()
        self._config = config

    def create_manager(
        self,
        task: Task,
        memory_context: Optional[Dict[str, Any]] = None,
        on_complete_callback=None,
    ) -> "ManagerBot":  # type: ignore[name-defined]
        """
        Instantiate a ManagerBot for *task*.

        Parameters
        ----------
        task:
            The task the manager will execute.
        memory_context:
            Optional pre-fetched context from GlobalMemory.
        on_complete_callback:
            Optional async callable invoked on completion.

        Returns
        -------
        ManagerBot
            A fully configured (but not yet started) ManagerBot.
        """
        from giga_ai.manager_bot.manager_bot import ManagerBot

        manager = ManagerBot(
            task=task,
            event_bus=self._bus,
            memory=self._memory,
            memory_context=memory_context or {},
            config=self._config,
            on_complete_callback=on_complete_callback,
        )

        log.debug(
            "BotFactory: created manager",
            extra={"manager_id": manager.manager_id, "task_id": task.task_id},
        )

        return manager

    def create_managers_for_tasks(
        self,
        tasks: list,
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> list:
        """
        Convenience helper: create one ManagerBot per task.

        Parameters
        ----------
        tasks:
            List of Task objects.
        memory_context:
            Shared memory context applied to all managers.

        Returns
        -------
        list[ManagerBot]
        """
        return [
            self.create_manager(task, memory_context=memory_context)
            for task in tasks
        ]
