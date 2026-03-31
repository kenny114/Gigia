"""
event_bus.py – Internal async pub/sub event bus.

Design
------
- Each event type has its own asyncio.Queue per subscriber.
- ``publish`` fans out to all registered subscriber queues.
- ``subscribe`` returns an AsyncGenerator that yields Messages.
- Subscribers can listen to a specific event type or ALL events (wildcard).

Event types
-----------
GOAL_RECEIVED       PerceptionBrain → rest of system
TASK_CREATED        PlanningBrain   → ExecutionBrain
MANAGER_SPAWNED     ExecutionBrain  → PerceptionBrain / LearningBrain
SUB_BOT_RESULT      SubBot          → ManagerBot
ESCALATION          ManagerBot      → LearningBrain
STRATEGY_LEARNED    LearningBrain   → any interested party
HEALTH_UPDATE       PerceptionBrain → any interested party

Usage
-----
    bus = EventBus()

    # Publisher
    await bus.publish(EventType.GOAL_RECEIVED, {"description": "..."}, cid)

    # Subscriber (async generator)
    async for message in bus.subscribe(EventType.GOAL_RECEIVED):
        print(message)

    # Unsubscribe by calling .aclose() on the generator or using the
    # context-manager helper:
    async with bus.subscription(EventType.GOAL_RECEIVED) as sub:
        async for msg in sub:
            ...
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Set
from datetime import datetime, timezone

from giga_ai.messaging.message_schemas import Message
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Event type catalogue
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    GOAL_RECEIVED = "GOAL_RECEIVED"
    TASK_CREATED = "TASK_CREATED"
    MANAGER_SPAWNED = "MANAGER_SPAWNED"
    SUB_BOT_RESULT = "SUB_BOT_RESULT"
    ESCALATION = "ESCALATION"
    STRATEGY_LEARNED = "STRATEGY_LEARNED"
    HEALTH_UPDATE = "HEALTH_UPDATE"
    MANAGER_CRASHED = "MANAGER_CRASHED"
    MANAGER_COMPLETED = "MANAGER_COMPLETED"
    ALL = "*"               # wildcard – receives every event


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

_SENTINEL = object()        # used to signal generator shutdown


class EventBus:
    """
    Async pub/sub event bus backed by per-subscriber asyncio Queues.

    Thread-safety
    -------------
    All operations are coroutine-based.  The bus must be used from within
    a single asyncio event loop.
    """

    def __init__(self, queue_max_size: int = 1000) -> None:
        self._queue_max_size = queue_max_size
        # {event_type_value: [queue, ...]}
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def publish(
        self,
        event_type: EventType | str,
        payload: dict,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Fan a message out to all subscribers of *event_type* and wildcard
        (``EventType.ALL``) subscribers.

        Parameters
        ----------
        event_type:
            The event category.
        payload:
            Arbitrary dict payload.
        correlation_id:
            Trace identifier.  A new UUID is generated if not supplied.
        """
        import uuid
        cid = correlation_id or str(uuid.uuid4())
        message = Message(
            type=str(event_type),
            payload=payload,
            correlation_id=cid,
        )

        targets: Set[str] = {str(event_type), EventType.ALL}
        async with self._lock:
            queues_snapshot = []
            for key in targets:
                queues_snapshot.extend(self._subscribers.get(key, []))

        dropped = 0
        for q in queues_snapshot:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                dropped += 1

        if dropped:
            log.warning(
                "EventBus: dropped messages due to full subscriber queues",
                extra={"event_type": str(event_type), "dropped": dropped},
            )
        else:
            log.debug(
                "EventBus: published event",
                extra={
                    "event_type": str(event_type),
                    "correlation_id": cid,
                    "subscriber_count": len(queues_snapshot),
                },
            )

    async def subscribe(
        self,
        event_type: EventType | str,
    ) -> AsyncGenerator[Message, None]:
        """
        Async generator that yields Messages for *event_type*.

        The generator runs indefinitely until the caller closes it
        (``aclose()``) or the bus is shut down.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_max_size)
        key = str(event_type)

        async with self._lock:
            self._subscribers.setdefault(key, []).append(queue)

        log.debug("EventBus: new subscriber", extra={"event_type": key})

        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            async with self._lock:
                try:
                    self._subscribers[key].remove(queue)
                    if not self._subscribers[key]:
                        del self._subscribers[key]
                except (KeyError, ValueError):
                    pass
            log.debug("EventBus: subscriber unregistered", extra={"event_type": key})

    @asynccontextmanager
    async def subscription(
        self,
        event_type: EventType | str,
    ):
        """
        Async context manager wrapper around ``subscribe``.

        Usage::

            async with bus.subscription(EventType.GOAL_RECEIVED) as gen:
                async for msg in gen:
                    ...
        """
        gen = self.subscribe(event_type)
        try:
            yield gen
        finally:
            await gen.aclose()

    async def shutdown(self) -> None:
        """
        Signal all active subscriber generators to stop by enqueuing
        a sentinel value into each queue.
        """
        async with self._lock:
            for queues in self._subscribers.values():
                for q in queues:
                    try:
                        q.put_nowait(_SENTINEL)
                    except asyncio.QueueFull:
                        pass
        log.info("EventBus: shutdown signal sent to all subscribers")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    async def subscriber_count(self, event_type: Optional[EventType | str] = None) -> int:
        """Return the number of active subscribers (optionally filtered by type)."""
        async with self._lock:
            if event_type is not None:
                return len(self._subscribers.get(str(event_type), []))
            return sum(len(v) for v in self._subscribers.values())
