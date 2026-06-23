"""
goal_generator_brain.py – The seventh brain. Gigia's self-directed improvement engine.

Every 10 minutes this brain looks inward:
  - Which skills are failing too often?
  - Which skills have never actually been used?
  - What goal categories keep going wrong?

It then generates 1-2 concrete, executable goals and submits them back into
Gigia's pipeline so the system learns from real executions — not hypotheticals.

Design
------
Self-improvement goals are tagged with _self_improvement=True in metadata so:
  • SynthesisBrain skips delivery callbacks (no result_url to POST to)
  • They can be filtered out of external-facing analytics
  • GoalGeneratorBrain doesn't re-analyse their syntheses to avoid loops

Observation categories
----------------------
1. Weak skills      — reliability < WEAK_THRESHOLD, >= MIN_SAMPLES uses
2. Untested skills  — seen in skill-mode candidate lists, never executed
3. Failure patterns — GlobalMemory strategies with success_rate < 0.4
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from giga_ai.messaging.event_bus import EventBus, EventType
from giga_ai.memory.skill_memory import SkillMemory
from giga_ai.memory.global_memory import GlobalMemory
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)

# ── Tunables ────────────────────────────────────────────────────────────────
_LOOP_INTERVAL_S   = 600     # analyse every 10 minutes
_WEAK_THRESHOLD    = 0.60    # skills below this are "weak"
_MIN_SAMPLES       = 3       # ignore skills with fewer than this many uses
_MAX_GOALS_PER_RUN = 2       # don't spam the pipeline
_RECENT_GOAL_CAP   = 20      # remember last N generated goal strings (dedup)
_FAILURE_RATE_CAP  = 0.40    # flag strategies below this success rate


# ── Prompt ───────────────────────────────────────────────────────────────────
_SYSTEM = """\
You are Gigia's self-improvement engine. Gigia is an autonomous skill-based agent \
that executes goals by chaining skills together (web scraping, YouTube transcription, \
PDF reading, code execution, etc.).

Your job: given raw observations about Gigia's performance, write 1-2 SHORT, \
CONCRETE, EXECUTABLE goals that Gigia can run RIGHT NOW to learn something useful.

Rules:
- Each goal must be something a real user might ask — Gigia should be able to plan \
  and execute it without special meta-knowledge.
- Prefer goals that naturally exercise weak or untested skills.
- Keep goals simple enough that a single skill call could satisfy them. \
  "Read https://en.wikipedia.org/wiki/Python_(programming_language) and summarise" \
  is perfect for testing web_reader.
- Do NOT write goals about "testing yourself" or "checking reliability". \
  Write the actual task.
- If there are no meaningful issues, return an empty goals list.

Return ONLY valid JSON: {"goals": ["goal 1", "goal 2"]}
"""


class GoalGeneratorBrain:
    """
    Seventh brain — wakes periodically, analyses skill/memory state,
    generates self-improvement goals, and submits them back to the pipeline.

    Parameters
    ----------
    event_bus:
        Shared EventBus instance.
    skill_memory:
        SkillMemory instance (already init'd).
    global_memory:
        GlobalMemory instance (already init'd).
    llm_client:
        Any client with an async ``chat(system, user) -> str`` method
        (same interface used by PlanningBrain / SynthesisBrain).
    submit_goal_fn:
        ``MainBot.submit_goal`` — async callable(description, metadata) -> Goal.
    interval_seconds:
        How often to run the analysis loop (default 600s).
    """

    def __init__(
        self,
        event_bus: EventBus,
        skill_memory: SkillMemory,
        global_memory: GlobalMemory,
        llm_client: Any,
        submit_goal_fn: Callable[..., Coroutine],
        interval_seconds: float = _LOOP_INTERVAL_S,
    ) -> None:
        self._bus            = event_bus
        self._skill_memory   = skill_memory
        self._global_memory  = global_memory
        self._llm            = llm_client
        self._submit         = submit_goal_fn
        self._interval       = interval_seconds

        # Skills Gigia has *seen* in candidate lists (may never have executed)
        self._seen_skills: Set[str] = set()
        # Recent goal strings — avoid re-submitting the same test goal
        self._recent_goals: List[str] = []

        # Total goals generated this session (for logging)
        self._total_generated: int = 0

        self._loop_task: Optional[asyncio.Task] = None
        self._listener_task: Optional[asyncio.Task] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._loop_task     = asyncio.create_task(self._analysis_loop(), name="goal_gen_loop")
        self._listener_task = asyncio.create_task(self._candidate_listener(), name="goal_gen_candidates")
        log.info("GoalGeneratorBrain: started", extra={"interval_s": self._interval})

    async def stop(self) -> None:
        for t in [self._loop_task, self._listener_task]:
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        log.info("GoalGeneratorBrain: stopped", extra={"total_generated": self._total_generated})

    # ── Main loop ────────────────────────────────────────────────────────────

    async def _analysis_loop(self) -> None:
        """Sleep, analyse, generate goals — forever."""
        # Wait a full interval before first run so the system warms up
        await asyncio.sleep(self._interval)
        while True:
            try:
                await self._run_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("GoalGeneratorBrain: cycle error", extra={"error": str(exc)})
            await asyncio.sleep(self._interval)

    async def _run_cycle(self) -> None:
        observations = await self._collect_observations()
        if not observations["has_anything"]:
            log.info("GoalGeneratorBrain: nothing worth improving this cycle")
            return

        goals = await self._generate_goals(observations)
        if not goals:
            return

        for goal_text in goals[:_MAX_GOALS_PER_RUN]:
            if goal_text in self._recent_goals:
                log.info("GoalGeneratorBrain: skipping duplicate goal", extra={"goal": goal_text[:60]})
                continue
            await self._submit(
                goal_text,
                metadata={
                    "_self_improvement": True,
                    "_observation_summary": observations["summary"],
                },
            )
            self._total_generated += 1
            # Track recent goals (cap list size)
            self._recent_goals.append(goal_text)
            if len(self._recent_goals) > _RECENT_GOAL_CAP:
                self._recent_goals.pop(0)
            log.info(
                "GoalGeneratorBrain: submitted self-improvement goal",
                extra={"goal": goal_text[:80], "total": self._total_generated},
            )

    # ── Observation collection ───────────────────────────────────────────────

    async def _collect_observations(self) -> Dict[str, Any]:
        weak_skills:     List[dict] = []
        untested_skills: List[str]  = []
        failure_patterns: List[dict] = []

        # 1. Weak skills
        try:
            profiles = await self._skill_memory.get_all_profiles()
            for p in profiles:
                total = p.get("success_count", 0) + p.get("fail_count", 0)
                if total >= _MIN_SAMPLES and p.get("reliability", 1.0) < _WEAK_THRESHOLD:
                    weak_skills.append({
                        "slug":        p["slug"],
                        "reliability": round(p.get("reliability", 0), 2),
                        "total_uses":  total,
                        "description": p.get("description", ""),
                    })
        except Exception as exc:
            log.warning("GoalGeneratorBrain: could not read skill profiles", extra={"error": str(exc)})

        # 2. Untested skills — in candidate lists but never executed
        try:
            profiles = await self._skill_memory.get_all_profiles()
            used_slugs = {p["slug"] for p in profiles}
            untested_skills = sorted(self._seen_skills - used_slugs)[:10]
        except Exception as exc:
            log.warning("GoalGeneratorBrain: could not compute untested skills", extra={"error": str(exc)})

        # 3. Failure patterns from GlobalMemory
        try:
            strategies = await self._global_memory.get_all_strategies()
            for s in strategies:
                if s.success_rate < _FAILURE_RATE_CAP:
                    failure_patterns.append({
                        "problem":      s.problem,
                        "solution":     s.solution,
                        "success_rate": round(s.success_rate, 2),
                    })
        except Exception as exc:
            log.warning("GoalGeneratorBrain: could not read failure patterns", extra={"error": str(exc)})

        has_anything = bool(weak_skills or untested_skills or failure_patterns)

        # Build a compact summary for the LLM
        parts = []
        if weak_skills:
            slugs = ", ".join(f"{s['slug']} ({int(s['reliability']*100)}% success)" for s in weak_skills[:5])
            parts.append(f"Weak skills: {slugs}")
        if untested_skills:
            parts.append(f"Never-used skills seen in plans: {', '.join(untested_skills[:5])}")
        if failure_patterns:
            probs = " | ".join(p["problem"][:60] for p in failure_patterns[:3])
            parts.append(f"Recurring failure patterns: {probs}")

        summary = "; ".join(parts) if parts else "No issues found"

        log.info(
            "GoalGeneratorBrain: observations collected",
            extra={
                "weak": len(weak_skills),
                "untested": len(untested_skills),
                "failures": len(failure_patterns),
                "summary": summary[:120],
            },
        )

        return {
            "has_anything":    has_anything,
            "weak_skills":     weak_skills,
            "untested_skills": untested_skills,
            "failure_patterns": failure_patterns,
            "summary":         summary,
        }

    # ── LLM goal generation ──────────────────────────────────────────────────

    async def _generate_goals(self, observations: Dict[str, Any]) -> List[str]:
        user_msg = (
            f"Today is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.\n\n"
            f"Observations about Gigia's skill performance:\n"
            f"{observations['summary']}\n\n"
        )
        if observations["weak_skills"]:
            user_msg += "Weak skill details:\n"
            for s in observations["weak_skills"][:5]:
                user_msg += f"  • {s['slug']}: {s['reliability']*100:.0f}% reliability over {s['total_uses']} uses"
                if s.get("description"):
                    user_msg += f" — {s['description'][:80]}"
                user_msg += "\n"
        if observations["untested_skills"]:
            user_msg += f"\nUntested skills (seen in plans, never executed): {', '.join(observations['untested_skills'])}\n"
        if observations["failure_patterns"]:
            user_msg += "\nFailure patterns:\n"
            for fp in observations["failure_patterns"][:3]:
                user_msg += f"  • {fp['problem'][:80]} (success rate: {fp['success_rate']*100:.0f}%)\n"
        user_msg += "\nGenerate 1-2 concrete goals. Return JSON only."

        try:
            raw = await self._llm.chat(_SYSTEM, user_msg)
            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            goals = data.get("goals", [])
            if not isinstance(goals, list):
                return []
            return [g.strip() for g in goals if isinstance(g, str) and g.strip()]
        except Exception as exc:
            log.error("GoalGeneratorBrain: LLM call failed", extra={"error": str(exc)})
            return []

    # ── Candidate listener ───────────────────────────────────────────────────

    async def _candidate_listener(self) -> None:
        """
        Track which skills Gigia has been offered (in skill-mode plans)
        so we know which ones have never been executed.
        """
        async for msg in self._bus.subscribe(EventType.GOAL_RECEIVED):
            try:
                payload = msg.payload
                if not payload.get("metadata", {}).get("_skill_mode"):
                    continue
                for c in payload.get("metadata", {}).get("candidates", []):
                    slug = c.get("slug") or c.get("name")
                    if slug:
                        self._seen_skills.add(slug)
            except Exception:
                pass
