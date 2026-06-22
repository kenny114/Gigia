"""
skill_brain.py – The fifth brain: skill intelligence and knowledge growth.

Responsibilities
----------------
- Listen for SKILL_EXECUTED events and update SkillMemory profiles.
- Listen for GOAL_COMPLETED events and record skill interaction patterns.
- Answer briefing requests from PlanningBrain: given a list of candidate
  slugs, return enriched profiles the LLM can use to plan smarter.
- Over time, builds a growing knowledge graph of what every skill does,
  how reliable it is, and which skills work well together.

The briefing format is designed to be injected directly into the planning
prompt — compact enough not to overwhelm gpt-4o, rich enough to matter.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from giga_ai.memory.skill_memory import SkillMemory
from giga_ai.messaging.event_bus import EventBus, EventType
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)


class SkillBrain:
    """
    Fifth brain: grows skill knowledge from every execution.

    Parameters
    ----------
    event_bus:
        Shared EventBus.
    skill_memory:
        SkillMemory instance (must be initialised before use).
    """

    def __init__(self, event_bus: EventBus, skill_memory: SkillMemory) -> None:
        self._bus = event_bus
        self._memory = skill_memory
        self._running = False
        self._skill_exec_task: Optional[asyncio.Task] = None
        self._goal_complete_task: Optional[asyncio.Task] = None

        # In-flight goal tracking: goal_id → list of slugs that ran (in order)
        self._goal_skills: Dict[str, List[str]] = {}
        # goal_id → goal description (for recording interactions)
        self._goal_descriptions: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._skill_exec_task = asyncio.create_task(
            self._skill_exec_listener(), name="skill_brain_exec_listener"
        )
        self._goal_complete_task = asyncio.create_task(
            self._goal_complete_listener(), name="skill_brain_goal_listener"
        )
        log.info("SkillBrain started")

    async def stop(self) -> None:
        self._running = False
        for t in [self._skill_exec_task, self._goal_complete_task]:
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        log.info("SkillBrain stopped")

    # ------------------------------------------------------------------
    # Public API — used by PlanningBrain
    # ------------------------------------------------------------------

    async def get_briefing(self, candidates: List[dict]) -> str:
        """
        Return a compact skill briefing string for injection into the
        planning prompt.

        Each candidate gets one line of base info (from the caller),
        followed by up to three lines of learned intelligence:
          - reliability + usage stats (if seen before)
          - output keys (what the skill returns)
          - good pairings (skills that work well with this one)

        Parameters
        ----------
        candidates:
            List of dicts with at least {"slug": str, "description": str}.

        Returns
        -------
        str
            Multi-line briefing ready to embed in a prompt.
        """
        slugs = [c["slug"] for c in candidates]
        profiles = await self._memory.get_profiles(slugs)

        lines = []
        for c in candidates:
            slug = c["slug"]
            desc = c.get("description", "")
            credits = c.get("credit_cost", "?")
            profile = profiles.get(slug)

            line = f"• {slug} ({credits} cr) — {desc}"

            if profile:
                total = profile["success_count"] + profile["fail_count"]
                if total >= 3:
                    pct = int(profile["reliability"] * 100)
                    line += f"\n    reliability: {pct}% ({total} uses)"

                keys = profile["output_keys"]
                if keys:
                    line += f"\n    returns: {{{', '.join(keys[:6])}}}"

                cases = profile["use_cases"]
                if cases:
                    line += f"\n    seen for: {' | '.join(cases[:2])}"

                # Pair suggestions only if seen at least twice
                pairs = await self._memory.get_interactions(slug, min_count=2, limit=3)
                if pairs:
                    partners = [p["partner"] for p in pairs]
                    line += f"\n    pairs well with: {', '.join(partners)}"

            lines.append(line)

        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Event listeners
    # ------------------------------------------------------------------

    async def _skill_exec_listener(self) -> None:
        """Update skill profiles after every skill execution."""
        async for msg in self._bus.subscribe(EventType.SKILL_EXECUTED):
            if not self._running:
                break
            try:
                p = msg.payload
                slug = p.get("slug", "")
                goal_id = p.get("goal_id", "")
                goal_desc = p.get("goal_description", "")
                success = bool(p.get("success", False))
                credits = float(p.get("credits", 0))
                result_keys = p.get("result_keys") or []
                description = p.get("description", "")

                # Track skill order within this goal
                if goal_id:
                    if goal_id not in self._goal_skills:
                        self._goal_skills[goal_id] = []
                        self._goal_descriptions[goal_id] = goal_desc
                    self._goal_skills[goal_id].append(slug)

                await self._memory.record_execution(
                    slug=slug,
                    description=description,
                    goal_description=goal_desc,
                    result_keys=result_keys,
                    success=success,
                    credits=credits,
                )

                log.debug(
                    "SkillBrain: recorded execution",
                    extra={"slug": slug, "success": success, "credits": credits},
                )
            except Exception as exc:
                log.error("SkillBrain: error processing SKILL_EXECUTED", extra={"error": str(exc)})

    async def _goal_complete_listener(self) -> None:
        """Record skill interaction patterns when a goal finishes."""
        async for msg in self._bus.subscribe(EventType.GOAL_COMPLETED):
            if not self._running:
                break
            try:
                goal_id = msg.payload.get("goal_id", "")
                success = bool(msg.payload.get("success", False))

                sequence = self._goal_skills.pop(goal_id, [])
                goal_desc = self._goal_descriptions.pop(goal_id, "")

                if len(sequence) >= 2:
                    await self._memory.record_goal_completion(
                        skill_sequence=sequence,
                        goal_description=goal_desc,
                        success=success,
                    )
                    log.info(
                        "SkillBrain: recorded goal completion",
                        extra={
                            "goal_id": goal_id,
                            "sequence": sequence,
                            "success": success,
                        },
                    )
            except Exception as exc:
                log.error("SkillBrain: error processing GOAL_COMPLETED", extra={"error": str(exc)})
