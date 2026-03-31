"""
planning_brain.py – LLM-powered goal decomposition and replanning.

Responsibilities
----------------
- Receive a ``Goal`` and call the LLM to decompose it into a list of
  ``Task`` objects (each with a sub_bot_type, priority, and dependencies).
- Re-decompose a failed task with updated context to produce an
  alternative execution plan.
- Emit ``TASK_CREATED`` events for each produced task.

Prompt template
---------------
The planning prompt instructs the model to respond with a valid JSON array
of task objects.  ``MockLLMClient`` obeys this contract deterministically.

Usage
-----
    bus = EventBus()
    llm = get_llm_client()
    brain = PlanningBrain(event_bus=bus, llm_client=llm)

    tasks = await brain.decompose_goal(goal)
    alt_tasks = await brain.replan(failed_task, {"error": "HTTP 404"})
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from giga_ai.messaging.event_bus import EventBus, EventType
from giga_ai.messaging.message_schemas import Goal, SubBotType, Task, TaskStatus
from giga_ai.utils.llm_client import LLMClient
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM_PROMPT = """You are a planning engine for an autonomous bot system.
Your job is to break down a high-level goal into a list of discrete, executable tasks.

Each task must have the following fields:
  task_id      (string) – a unique identifier like "task-1", "task-2", etc.
  title        (string) – short task title
  description  (string) – detailed description of what to do
  sub_bot_type (string) – one of: "scraper", "selenium", "generic"
  priority     (integer) – execution order, lower = higher priority (start at 1)
  dependencies (array of task_id strings) – tasks that must complete before this one
  metadata     (object) – REQUIRED for scraper/selenium tasks; must include:
                  "url": the full URL to scrape (e.g. "https://www.google.com/search?q=...")
                  "css_selectors": object mapping field names to CSS selectors (optional)
                  For search-based goals use Google: "https://www.google.com/search?q=ENCODED+QUERY"

IMPORTANT: Every scraper or selenium task MUST include a "url" inside metadata.
Without a url the task will fail immediately. Use real, publicly accessible URLs.

Respond with ONLY a valid JSON array of task objects. No prose, no markdown fences."""

_DECOMPOSE_USER_TEMPLATE = """Goal: {goal}

Additional context:
{context}

Decompose this goal into 2–6 concrete tasks."""


_REPLAN_SYSTEM_PROMPT = """You are a replanning engine for an autonomous bot system.
A task has failed. Your job is to produce an alternative set of tasks that achieve
the same objective using a different approach.

Respond with ONLY a valid JSON array of task objects (same schema as before)."""

_REPLAN_USER_TEMPLATE = """Original failed task:
  Title: {task_title}
  Description: {task_description}
  Sub-bot type: {sub_bot_type}

Failure context:
{failure_context}

Goal context:
{goal_context}

Produce 1–4 alternative tasks that accomplish the same goal differently."""


# ---------------------------------------------------------------------------
# PlanningBrain
# ---------------------------------------------------------------------------

class PlanningBrain:
    """
    Layer-1 brain: LLM-powered goal decomposition.

    Parameters
    ----------
    event_bus:
        Shared EventBus instance for emitting TASK_CREATED events.
    llm_client:
        Any ``LLMClient`` implementation (OpenAI or Mock).
    """

    def __init__(self, event_bus: EventBus, llm_client: LLMClient) -> None:
        self._bus = event_bus
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def decompose_goal(
        self,
        goal: Goal,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[Task]:
        """
        Decompose *goal* into a list of ``Task`` objects via the LLM.

        Parameters
        ----------
        goal:
            The Goal to decompose.
        extra_context:
            Optional additional context forwarded to the prompt.

        Returns
        -------
        List[Task]
            Ordered list of tasks (sorted by priority).
        """
        context_str = json.dumps(extra_context or {}, indent=2)
        prompt = _DECOMPOSE_USER_TEMPLATE.format(
            goal=goal.description,
            context=context_str,
        )

        log.info(
            "PlanningBrain: decomposing goal",
            extra={"goal_id": goal.goal_id, "description": goal.description[:80]},
        )

        try:
            raw_response = await self._llm.complete(prompt, system_prompt=_DECOMPOSE_SYSTEM_PROMPT)
        except Exception as exc:
            log.error(
                "PlanningBrain: LLM call failed",
                extra={"goal_id": goal.goal_id, "error": type(exc).__name__, "detail": str(exc)},
            )
            raise
        tasks = self._parse_tasks(raw_response, goal.goal_id, goal.correlation_id)

        log.info(
            "PlanningBrain: goal decomposed",
            extra={"goal_id": goal.goal_id, "task_count": len(tasks)},
        )

        # Emit TASK_CREATED for each task
        for task in tasks:
            await self._emit_task_created(task)

        return tasks

    async def replan(
        self,
        failed_task: Task,
        context: Dict[str, Any],
    ) -> List[Task]:
        """
        Produce alternative tasks after *failed_task* failed.

        Parameters
        ----------
        failed_task:
            The task that could not be completed.
        context:
            Failure context (error type, error message, attempt history, …).

        Returns
        -------
        List[Task]
            Alternative tasks to attempt instead.
        """
        failure_str = json.dumps(context, indent=2)
        goal_ctx = context.get("goal_description", "")

        prompt = _REPLAN_USER_TEMPLATE.format(
            task_title=failed_task.title,
            task_description=failed_task.description,
            sub_bot_type=failed_task.sub_bot_type,
            failure_context=failure_str,
            goal_context=goal_ctx,
        )

        log.info(
            "PlanningBrain: replanning after failure",
            extra={"task_id": failed_task.task_id, "context": str(context)[:200]},
        )

        raw_response = await self._llm.complete(prompt, system_prompt=_REPLAN_SYSTEM_PROMPT)
        new_tasks = self._parse_tasks(raw_response, failed_task.goal_id, failed_task.correlation_id)

        log.info(
            "PlanningBrain: replan complete",
            extra={"original_task_id": failed_task.task_id, "new_task_count": len(new_tasks)},
        )

        for task in new_tasks:
            await self._emit_task_created(task)

        return new_tasks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_tasks(self, raw: str, goal_id: str, correlation_id: str) -> List[Task]:
        """
        Parse an LLM response into a list of Task objects.

        Falls back to a single generic task if parsing fails.
        """
        raw = raw.strip()

        # Strip markdown fences if the model wrapped the JSON
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )
            raw = raw.strip()

        # Extract the JSON array if the model wrapped it in prose
        if not raw.startswith("["):
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1:
                raw = raw[start:end + 1]

        # Strip JavaScript-style // comments (GPT sometimes adds them)
        import re
        raw = re.sub(r"//[^\n]*", "", raw)
        # Strip trailing commas before ] or } (common GPT mistake)
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        # Replace literal control characters inside strings (newlines, tabs, etc.)
        raw = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f]', ' ', raw)
        # Replace unescaped literal newlines inside JSON strings
        raw = re.sub(r'(?<!\\)\n', ' ', raw)

        try:
            data = json.loads(raw)
            # json_object mode returns {"tasks": [...]} — unwrap it
            if isinstance(data, dict):
                data = next(
                    (v for v in data.values() if isinstance(v, list)),
                    [data],
                )
        except json.JSONDecodeError as exc:
            log.warning(
                "PlanningBrain: failed to parse LLM response as JSON – using fallback",
                extra={"error": str(exc), "raw": raw[:300]},
            )
            data = [
                {
                    "task_id": "fallback-1",
                    "title": "Fallback task",
                    "description": "Could not parse LLM decomposition; manual intervention required.",
                    "sub_bot_type": "generic",
                    "priority": 1,
                    "dependencies": [],
                }
            ]

        tasks: List[Task] = []
        for item in data:
            try:
                sub_bot_type_raw = item.get("sub_bot_type", "scraper")
                try:
                    sub_bot_type = SubBotType(sub_bot_type_raw)
                except ValueError:
                    sub_bot_type = SubBotType.GENERIC

                task = Task(
                    task_id=item.get("task_id", f"task-{len(tasks)+1}"),
                    goal_id=goal_id,
                    title=item.get("title", "Untitled task"),
                    description=item.get("description", ""),
                    sub_bot_type=sub_bot_type,
                    priority=int(item.get("priority", len(tasks) + 1)),
                    dependencies=item.get("dependencies", []),
                    status=TaskStatus.PENDING,
                    correlation_id=correlation_id,
                    metadata=item.get("metadata", {}),
                )
                tasks.append(task)
            except Exception as exc:
                log.warning(
                    "PlanningBrain: skipped malformed task entry",
                    extra={"error": str(exc), "item": str(item)[:200]},
                )

        # Sort by priority
        tasks.sort(key=lambda t: t.priority)
        return tasks

    async def _emit_task_created(self, task: Task) -> None:
        await self._bus.publish(
            EventType.TASK_CREATED,
            payload=task.model_dump(mode="json"),
            correlation_id=task.correlation_id,
        )
