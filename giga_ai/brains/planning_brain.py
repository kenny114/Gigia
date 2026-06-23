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
from giga_ai.messaging.message_schemas import (
    GatewayCallback,
    Goal,
    OrchestrateCandidate,
    SubBotType,
    Task,
    TaskStatus,
)
from giga_ai.utils.llm_client import LLMClient
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM_PROMPT = """You are a planning engine for an autonomous bot system.
Your job is to break down a high-level goal into a list of discrete, executable tasks.

You must respond with a JSON object in this exact format:
{"tasks": [ ... array of task objects ... ]}

Each task object must have these fields:
  task_id      (string)  – unique id like "task-1", "task-2", etc.
  title        (string)  – short task title
  description  (string)  – what to do, single line, no newlines
  sub_bot_type (string)  – one of: "scraper", "browser", "code", "file", "shell"
  priority     (integer) – execution order, 1 = highest priority
  dependencies (array)   – list of task_id strings that must finish first
  metadata     (object)  – REQUIRED; see rules below

SUB_BOT_TYPE SELECTION RULES (follow exactly):
- "browser": Google Maps, Google Search, Yelp, TripAdvisor, any JavaScript-heavy site or SPA.
- "scraper": Wikipedia, plain HTML pages, static sites, REST APIs returning HTML.
- "code": run Python — data analysis, math, CSV/JSON processing, transformations, charting. metadata must include "code" (Python source string). Use input_data to pass values from prior tasks.
- "file": read or write a file in the workspace. metadata must include "operation" ("read"/"write"/"append"/"list"/"delete") and "path" (relative path). For writes also include "content".
- "shell": run a shell command — ffmpeg, git, curl, imagemagick, zip, etc. metadata must include "command". Only use for tools that have no Python equivalent.
- When in doubt between browser/scraper, use "browser".
- Prefer "code" over "shell" for data processing tasks.

METADATA RULES FOR "browser" AND "scraper":
- metadata MUST include "url": a fully-qualified public URL.
- For maps/business searches: "url": "https://www.google.com/maps/search/your+encoded+query"
- For web searches: "url": "https://www.google.com/search?q=your+encoded+query"

ADDITIONAL METADATA FOR "browser":
- "wait_for" (string): CSS selector that appears when results load.
- "scroll_feed" (boolean): true if scrolling is needed.
- "css_selectors" (object): field_name → CSS selector for extraction.

METADATA FOR "code":
- "code" (string): Python source. Print structured output with "RESULT: <json>" on its own line. Access prior task results via input_data dict.
- "input_data" (object): key/value pairs injected as the input_data variable.
- "timeout" (int): seconds, default 30.

METADATA FOR "file":
- "operation": "read" | "write" | "append" | "list" | "delete"
- "path": relative path within workspace (e.g. "output/results.csv")
- "content": string to write (required for write/append)

METADATA FOR "shell":
- "command": the shell command to run (e.g. "ffmpeg -i input.mp4 output.mp3")
- "working_dir": relative path within workspace (optional)
- "timeout": seconds, default 30.

ABSOLUTE RULES:
- browser/scraper tasks MUST have metadata.url set to a real public URL.
- description must be a single line string with no newline characters.
- Always produce 2-6 tasks."""

_DECOMPOSE_USER_TEMPLATE = """Goal: {goal}

Additional context:
{context}

Decompose this goal into 2–6 concrete tasks."""


# ---------------------------------------------------------------------------
# Skill-mode prompts (used when candidates are provided by the gateway)
# ---------------------------------------------------------------------------

_SKILL_DECOMPOSE_SYSTEM_PROMPT = """You are a planning engine for an AI skill-execution system.
Your job is to break a goal into discrete tasks, each executed by ONE skill from the provided catalog.

You must respond with a JSON object in this exact format:
{"tasks": [ ... array of task objects ... ]}

Each task object must have these fields:
  task_id      (string)  – unique id like "task-1", "task-2", etc.
  title        (string)  – short task title
  description  (string)  – what the skill will do, single line, no newlines
  sub_bot_type (string)  – always "skill"
  skill_slug   (string)  – MUST be one of the provided candidate slugs
  priority     (integer) – execution order, 1 = highest priority
  dependencies (array)   – list of task_id strings that must complete first (enables parallel execution)
  metadata     (object)  – skill arguments; use "args" key for the skill's input parameters

RULES:
- skill_slug MUST be chosen from the CANDIDATE SKILLS list. Never invent a slug.
- metadata.args should contain the skill's input parameters (e.g. {"url": "...", "query": "..."}).
- Use dependencies to model data flow: if task-2 needs task-1's output, add "task-1" to task-2's dependencies.
- Parallel tasks (no shared dependencies) run concurrently — use this to speed up multi-entity goals.
- Prefer 2–5 tasks. Only add more if the goal genuinely requires it.
- description must be a single line string with no newline characters."""

_SKILL_DECOMPOSE_USER_TEMPLATE = """Goal: {goal}

CANDIDATE SKILLS (choose skill_slug only from these):
{candidates}

Additional context:
{context}

Decompose this goal into tasks. Each task must use exactly one skill from the list above."""

_SKILL_REPLAN_SYSTEM_PROMPT = """You are a replanning engine for an AI skill-execution system.
A skill task has failed. Produce alternative tasks using different skills from the provided catalog.

Respond with a JSON object: {"tasks": [ ... ]}
Use the same task schema as the decompose prompt (sub_bot_type: "skill", skill_slug from candidates).
If the failure was a credit/rate error, try a lighter skill.
If the failure was unknown_tool or invalid_input, pick a different skill for the same intent."""

_SKILL_REPLAN_USER_TEMPLATE = """Original failed task:
  Title: {task_title}
  Description: {task_description}
  Skill slug: {skill_slug}

Failure context:
{failure_context}

CANDIDATE SKILLS:
{candidates}

Produce 1–3 alternative tasks using different skills from the list."""

_REPLAN_SYSTEM_PROMPT = """You are a replanning engine for an autonomous bot system.
A task has failed. Your job is to produce an alternative set of tasks that achieve
the same objective using a different approach.

Respond with a JSON object in this exact format:
{"tasks": [ ... array of task objects ... ]}

Use the same task schema and sub_bot_type rules as the decompose prompt.
If the failure was a bot-detection/captcha, try a different URL or site.
If the failure was a parse error, adjust the css_selectors or use a different approach."""

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

    def __init__(self, event_bus: EventBus, llm_client: LLMClient, skill_brain=None) -> None:
        self._bus = event_bus
        self._llm = llm_client
        self._skill_brain = skill_brain  # Optional[SkillBrain] — injected by MainBot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def decompose_goal(
        self,
        goal: Goal,
        extra_context: Optional[Dict[str, Any]] = None,
        candidates: Optional[List[OrchestrateCandidate]] = None,
        gateway_callback: Optional[GatewayCallback] = None,
    ) -> List[Task]:
        """
        Decompose *goal* into a list of Task objects via the LLM.

        When *candidates* and *gateway_callback* are provided the planner
        operates in skill-mode: tasks use sub_bot_type="skill" and each maps
        to one almcp catalog skill. Otherwise it falls back to the original
        scraper/browser decomposition.
        """
        skill_mode = bool(candidates and gateway_callback)
        context_str = json.dumps(extra_context or {}, indent=2)

        if skill_mode:
            # Build base candidate list
            candidate_dicts = [
                {
                    "slug": c.slug,
                    "name": c.name,
                    "description": c.description,
                    "credit_cost": c.credits,
                    "best_used_when": c.best_used_when or "",
                }
                for c in (candidates or [])
            ]

            # Enrich with SkillBrain intelligence if available
            if self._skill_brain is not None:
                try:
                    candidates_str = await self._skill_brain.get_briefing(candidate_dicts)
                except Exception as exc:
                    log.warning("PlanningBrain: SkillBrain briefing failed — using raw candidates", extra={"error": str(exc)})
                    import json as _json
                    def _fmt_fallback(c: dict) -> str:
                        line = f"  {c['slug']}: {c['name']} — {c['description']}"
                        if c.get("best_used_when"):
                            line += f" (best when: {c['best_used_when']})"
                        if c.get("example_call"):
                            line += f" [example args: {_json.dumps(c['example_call'])}]"
                        return line
                    candidates_str = "\n".join(_fmt_fallback(c) for c in candidate_dicts)
            else:
                def _fmt_candidate(c: dict) -> str:
                    line = f"  {c['slug']}: {c['name']} — {c['description']}"
                    if c.get("best_used_when"):
                        line += f" (best when: {c['best_used_when']})"
                    if c.get("example_call"):
                        import json as _json
                        line += f" [example args: {_json.dumps(c['example_call'])}]"
                    return line
                candidates_str = "\n".join(_fmt_candidate(c) for c in candidate_dicts)

            prompt = _SKILL_DECOMPOSE_USER_TEMPLATE.format(
                goal=goal.description,
                candidates=candidates_str,
                context=context_str,
            )
            system_prompt = _SKILL_DECOMPOSE_SYSTEM_PROMPT
        else:
            prompt = _DECOMPOSE_USER_TEMPLATE.format(
                goal=goal.description,
                context=context_str,
            )
            system_prompt = _DECOMPOSE_SYSTEM_PROMPT

        log.info(
            "PlanningBrain: decomposing goal",
            extra={
                "goal_id": goal.goal_id,
                "description": goal.description[:80],
                "skill_mode": skill_mode,
            },
        )

        try:
            raw_response = await self._llm.complete(prompt, system_prompt=system_prompt)
        except Exception as exc:
            log.error(
                "PlanningBrain: LLM call failed",
                extra={"goal_id": goal.goal_id, "error": type(exc).__name__, "detail": str(exc)},
            )
            raise

        tasks = self._parse_tasks(
            raw_response,
            goal.goal_id,
            goal.correlation_id,
            gateway_callback=gateway_callback,
        )

        log.info(
            "PlanningBrain: goal decomposed",
            extra={"goal_id": goal.goal_id, "task_count": len(tasks)},
        )

        if not tasks:
            if skill_mode:
                # No tasks planned despite being in skill mode — either the
                # candidate list was empty or none of the available skills
                # matched the goal. Signal SkillFactoryBrain so it can generate
                # a new skill for this capability gap.
                await self._bus.publish(
                    EventType.SKILL_GAP_DETECTED,
                    payload={
                        "goal_id":          goal.goal_id,
                        "goal_description": goal.description,
                        "candidate_count":  len(candidates or []),
                    },
                )

            # Fire GOAL_COMPLETED so SynthesisBrain delivers a result.
            await self._bus.publish(
                EventType.GOAL_COMPLETED,
                payload={
                    "goal_id":          goal.goal_id,
                    "goal_description": goal.description,
                    "success":          False,
                },
            )
            return tasks

        for task in tasks:
            await self._emit_task_created(task)

        return tasks

    async def replan(
        self,
        failed_task: Task,
        context: Dict[str, Any],
        candidates: Optional[List[OrchestrateCandidate]] = None,
        gateway_callback: Optional[GatewayCallback] = None,
    ) -> List[Task]:
        """
        Produce alternative tasks after *failed_task* failed.
        Supports both skill-mode (candidates provided) and scraper/browser mode.
        """
        failure_str = json.dumps(context, indent=2)
        skill_mode = bool(candidates and gateway_callback)

        if skill_mode:
            candidates_str = "\n".join(
                f"  {c.slug}: {c.name} — {c.description}" for c in (candidates or [])
            )
            prompt = _SKILL_REPLAN_USER_TEMPLATE.format(
                task_title=failed_task.title,
                task_description=failed_task.description,
                skill_slug=failed_task.skill_slug or "unknown",
                failure_context=failure_str,
                candidates=candidates_str,
            )
            system_prompt = _SKILL_REPLAN_SYSTEM_PROMPT
        else:
            goal_ctx = context.get("goal_description", "")
            prompt = _REPLAN_USER_TEMPLATE.format(
                task_title=failed_task.title,
                task_description=failed_task.description,
                sub_bot_type=failed_task.sub_bot_type,
                failure_context=failure_str,
                goal_context=goal_ctx,
            )
            system_prompt = _REPLAN_SYSTEM_PROMPT

        log.info(
            "PlanningBrain: replanning after failure",
            extra={"task_id": failed_task.task_id, "skill_mode": skill_mode},
        )

        raw_response = await self._llm.complete(prompt, system_prompt=system_prompt)
        new_tasks = self._parse_tasks(
            raw_response,
            failed_task.goal_id,
            failed_task.correlation_id,
            gateway_callback=gateway_callback,
        )

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

    def _parse_tasks(
        self,
        raw: str,
        goal_id: str,
        correlation_id: str,
        gateway_callback: Optional[GatewayCallback] = None,
    ) -> List[Task]:
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

        # Unwrap {"tasks": [...]} or any top-level object wrapping an array
        if raw.startswith("{"):
            try:
                obj = json.loads(raw)
                arr = next((v for v in obj.values() if isinstance(v, list)), None)
                if arr is not None:
                    raw = json.dumps(arr)
            except json.JSONDecodeError:
                pass  # fall through to array-extraction below

        # Extract the JSON array if the model wrapped it in prose
        if not raw.startswith("["):
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1:
                raw = raw[start:end + 1]

        # NOTE: do NOT strip // here — it also strips URLs (https://...).
        # response_format=json_object means the model won't emit JS comments.
        import re
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
                    "sub_bot_type": "browser",
                    "priority": 1,
                    "dependencies": [],
                    "metadata": {},
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

                skill_slug = item.get("skill_slug") or None
                metadata = dict(item.get("metadata") or {})

                # Skill tasks: stamp gateway callback into metadata so
                # ManagerBot._build_instructions can route to the right endpoint.
                if sub_bot_type == SubBotType.SKILL and gateway_callback:
                    metadata.setdefault("execute_url", gateway_callback.execute_url)
                    metadata.setdefault("token", gateway_callback.token)
                    metadata.setdefault("run_id", goal_id)
                    # args may be nested under "args" key or flat in metadata
                    if "args" not in metadata and skill_slug:
                        metadata["args"] = {
                            k: v for k, v in metadata.items()
                            if k not in {"execute_url", "token", "run_id"}
                        }

                task = Task(
                    task_id=item.get("task_id", f"task-{len(tasks)+1}"),
                    goal_id=goal_id,
                    title=item.get("title", "Untitled task"),
                    description=item.get("description", ""),
                    sub_bot_type=sub_bot_type,
                    skill_slug=skill_slug,
                    priority=int(item.get("priority", len(tasks) + 1)),
                    dependencies=item.get("dependencies", []),
                    status=TaskStatus.PENDING,
                    correlation_id=correlation_id,
                    metadata=metadata,
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
