"""
introspection_brain.py – The eighth brain. Gigia's self-awareness engine.

While all the other brains are outward-facing (process goals, execute skills,
learn from failures), IntrospectionBrain looks inward. It watches the whole
system run, measures it, scores the quality of what it produces, identifies
patterns in what it's being used for, and generates an intelligence briefing
that tells the rest of the system what to prioritise.

Responsibilities
----------------
1. RUNTIME TRACKING   — every goal lifecycle: submitted → planned → executed →
                        completed. Captures latency at each stage, credits
                        spent, task count, success/failure.

2. QUALITY SCORING    — after a goal completes, scores the synthesized answer
                        on specificity, completeness, and usefulness (0–10 each).
                        "Did it produce a real answer or just words?"

3. PURPOSE PATTERNS   — across all goals, builds a picture of what Gigia is
                        actually being used for. Categories, success rates per
                        category, credit spend per category.

4. CAPABILITY GAP     — when a category fails repeatedly or quality is
                        consistently low, flags a genuine capability gap and
                        writes a SubBot/skill proposal.

5. INTELLIGENCE BRIEF — every 30 minutes, generates a full briefing that
                        GoalGeneratorBrain reads to decide what to improve next.
                        Stored in SQLite + served via /introspection.

Schema
------
  goal_metrics
    goal_id          TEXT PRIMARY KEY
    goal_text        TEXT
    category         TEXT
    is_self_improve  INTEGER (0/1)
    submitted_at     TEXT
    planned_at       TEXT
    first_task_at    TEXT
    completed_at     TEXT
    status           TEXT   -- completed | failed | timeout
    task_count       INTEGER
    skill_slugs      TEXT   -- JSON list
    credits_spent    REAL
    latency_plan_s   REAL   -- seconds from submit → first task
    latency_exec_s   REAL   -- seconds from first task → completed
    quality_score    REAL   -- 0–1 aggregate
    quality_notes    TEXT

  intelligence_briefings
    id               INTEGER PRIMARY KEY AUTOINCREMENT
    created_at       TEXT
    health_score     INTEGER  (0–100)
    primary_use_case TEXT
    top_bottleneck   TEXT
    top_improvement  TEXT
    capability_gaps  TEXT     -- JSON list
    raw              TEXT     -- full JSON from LLM
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiosqlite

from giga_ai.messaging.event_bus import EventBus, EventType
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)

# ── Tunables ────────────────────────────────────────────────────────────────
_ANALYSIS_INTERVAL_S = 1800   # full briefing every 30 minutes
_QUALITY_MIN_ANSWER  = 40     # skip quality scoring if answer is shorter than this
_GAP_FAIL_THRESHOLD  = 3      # flag capability gap after N consecutive category failures
_HISTORY_WINDOW      = 50     # look at last N goals for patterns


# ── Category classifier (no LLM needed) ─────────────────────────────────────
_CATEGORY_RULES = [
    (["youtube", "video", "transcript", "channel", "watch"],           "YouTube"),
    (["http://", "https://", "website", "article", "web", "url",
      "page", "read", "scrape", "site"],                               "Web Research"),
    (["pdf", "document", "report", "paper"],                           "Document"),
    (["code", "python", "script", "calculate", "compute", "function",
      "algorithm", "fibonacci", "sort", "generate code"],              "Code Execution"),
    (["file", "save", "write to", "read file", "workspace"],           "File Operations"),
    (["search", "find", "discover", "lookup", "list", "top"],          "Search & Discovery"),
    (["email", "message", "send", "notify"],                           "Messaging"),
    (["image", "screenshot", "photo", "picture"],                      "Visual"),
]

def _classify(goal_text: str) -> str:
    low = goal_text.lower()
    for keywords, category in _CATEGORY_RULES:
        if any(kw in low for kw in keywords):
            return category
    return "General"


# ── Prompts ──────────────────────────────────────────────────────────────────

_QUALITY_SYSTEM = """\
You evaluate the quality of an AI agent's answer. Score briefly and honestly.
Return ONLY valid JSON: {"specificity": N, "completeness": N, "usefulness": N, "notes": "one sentence"}
All scores 0–10. No markdown, no extra text."""

_BRIEFING_SYSTEM = """\
You are Gigia's intelligence officer. You have operational data from a running AI agent system.
Your job is to generate an honest, actionable briefing about system health and what to improve.
Return ONLY valid JSON — no markdown, no explanation outside the JSON."""

_BRIEFING_USER_TEMPLATE = """\
Operational data (last {window} goals, as of {date}):

CATEGORY BREAKDOWN:
{category_stats}

RECENT FAILURES:
{recent_failures}

QUALITY SCORES (0-10 avg per category):
{quality_stats}

LATENCY (avg seconds from submit to complete):
{latency_stats}

CAPABILITY GAPS DETECTED:
{gap_signals}

Generate a briefing:
{{
  "health_score": 0-100,
  "primary_use_case": "one sentence about what Gigia is mainly used for",
  "top_bottleneck": "one sentence about the biggest problem right now",
  "top_improvement": "one concrete thing that would help most",
  "capability_gaps": ["thing it cannot do", "thing it cannot do"],
  "goal_generator_focus": "one sentence telling GoalGeneratorBrain what to prioritise"
}}"""

_CAPABILITY_GAP_SYSTEM = """\
You are designing a new capability for an AI agent. Based on the failure pattern described,
write a short technical proposal for what new SubBot or skill integration would fix it.
Return ONLY valid JSON: {"title": "...", "description": "...", "implementation_hint": "..."}"""


class IntrospectionBrain:
    """
    Eighth brain — watches Gigia run, measures everything, generates intelligence briefings.

    Parameters
    ----------
    event_bus   : shared EventBus
    llm_client  : LLMClient instance (complete / system_prompt interface)
    db_path     : SQLite file path (same as everything else)
    """

    def __init__(self, event_bus: EventBus, llm_client: Any, db_path: str) -> None:
        self._bus    = event_bus
        self._llm    = llm_client
        self._db     = db_path

        # In-memory tracking for goals currently in flight
        # goal_id → partial metric dict
        self._in_flight: Dict[str, Dict[str, Any]] = {}

        # Latest briefing — GoalGeneratorBrain reads this
        self._latest_briefing: Dict[str, Any] = {}

        # Category fail streaks for gap detection
        # category → consecutive fail count
        self._fail_streaks: Dict[str, int] = {}

        self._loop_task:       Optional[asyncio.Task] = None
        self._goal_task:       Optional[asyncio.Task] = None
        self._task_task:       Optional[asyncio.Task] = None
        self._complete_task:   Optional[asyncio.Task] = None
        self._skill_task:      Optional[asyncio.Task] = None
        self._escalation_task: Optional[asyncio.Task] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await self._init_db()
        self._goal_task       = asyncio.create_task(self._goal_listener(),       name="intro_goal")
        self._task_task       = asyncio.create_task(self._task_listener(),       name="intro_task")
        self._complete_task   = asyncio.create_task(self._complete_listener(),   name="intro_complete")
        self._skill_task      = asyncio.create_task(self._skill_listener(),      name="intro_skill")
        self._escalation_task = asyncio.create_task(self._escalation_listener(), name="intro_escalation")
        self._loop_task       = asyncio.create_task(self._briefing_loop(),       name="intro_loop")
        log.info("IntrospectionBrain: started")

    async def stop(self) -> None:
        for t in [self._loop_task, self._goal_task, self._task_task,
                  self._complete_task, self._skill_task, self._escalation_task]:
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        log.info("IntrospectionBrain: stopped")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_briefing(self) -> Dict[str, Any]:
        """Return the most recent intelligence briefing (empty dict if not yet generated)."""
        return self._latest_briefing

    async def get_dashboard(self) -> Dict[str, Any]:
        """Full dashboard data for /introspection endpoint."""
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row

            # Recent goal metrics
            async with db.execute(
                "SELECT * FROM goal_metrics ORDER BY submitted_at DESC LIMIT 100"
            ) as c:
                goals = [dict(r) for r in await c.fetchall()]

            # Latest briefing
            async with db.execute(
                "SELECT * FROM intelligence_briefings ORDER BY created_at DESC LIMIT 1"
            ) as c:
                row = await c.fetchone()
                briefing = dict(row) if row else {}

        # Compute live stats
        total = len(goals)
        completed = sum(1 for g in goals if g["status"] == "completed")
        failed    = sum(1 for g in goals if g["status"] == "failed")

        cats: Dict[str, Dict[str, Any]] = {}
        for g in goals:
            cat = g["category"] or "General"
            if cat not in cats:
                cats[cat] = {"count": 0, "success": 0, "total_quality": 0.0, "quality_count": 0}
            cats[cat]["count"] += 1
            if g["status"] == "completed":
                cats[cat]["success"] += 1
            if g.get("quality_score") is not None:
                cats[cat]["total_quality"] += g["quality_score"]
                cats[cat]["quality_count"] += 1

        category_stats = {
            cat: {
                "count":      v["count"],
                "success_rate": round(v["success"] / v["count"], 2) if v["count"] else 0,
                "avg_quality":  round(v["total_quality"] / v["quality_count"], 2) if v["quality_count"] else None,
            }
            for cat, v in cats.items()
        }

        return {
            "totals": {"total": total, "completed": completed, "failed": failed,
                       "success_rate": round(completed / total, 2) if total else 0},
            "category_stats":  category_stats,
            "latest_briefing": briefing,
            "in_flight":       len(self._in_flight),
            "fail_streaks":    self._fail_streaks,
            "recent_goals":    goals[:20],
        }

    # ── Event listeners ───────────────────────────────────────────────────────

    async def _goal_listener(self) -> None:
        async for msg in self._bus.subscribe(EventType.GOAL_RECEIVED):
            try:
                p = msg.payload
                goal_id   = p.get("goal_id", "")
                goal_text = p.get("description", "")
                meta      = p.get("metadata") or {}
                self._in_flight[goal_id] = {
                    "goal_id":       goal_id,
                    "goal_text":     goal_text,
                    "category":      _classify(goal_text),
                    "is_self_improve": int(bool(meta.get("_self_improvement"))),
                    "submitted_at":  _now(),
                    "skill_slugs":   [],
                    "credits_spent": 0.0,
                    "task_count":    0,
                }
            except Exception as exc:
                log.warning("IntrospectionBrain: goal_listener error", extra={"e": str(exc)})

    async def _task_listener(self) -> None:
        async for msg in self._bus.subscribe(EventType.TASK_CREATED):
            try:
                p = msg.payload
                goal_id = p.get("goal_id", "")
                if goal_id not in self._in_flight:
                    continue
                m = self._in_flight[goal_id]
                m["task_count"] = m.get("task_count", 0) + 1
                if "first_task_at" not in m:
                    m["first_task_at"] = _now()
            except Exception:
                pass

    async def _skill_listener(self) -> None:
        async for msg in self._bus.subscribe(EventType.SKILL_EXECUTED):
            try:
                p = msg.payload
                goal_id = p.get("goal_id", "")
                if goal_id not in self._in_flight:
                    continue
                m = self._in_flight[goal_id]
                slug = p.get("slug", "")
                if slug and slug not in m["skill_slugs"]:
                    m["skill_slugs"].append(slug)
                m["credits_spent"] = m.get("credits_spent", 0.0) + float(p.get("credits", 0))
            except Exception:
                pass

    async def _escalation_listener(self) -> None:
        async for msg in self._bus.subscribe(EventType.ESCALATION):
            try:
                p = msg.payload
                goal_id = p.get("goal_id", "")
                if goal_id in self._in_flight:
                    self._in_flight[goal_id]["had_escalation"] = True
            except Exception:
                pass

    async def _complete_listener(self) -> None:
        async for msg in self._bus.subscribe(EventType.GOAL_COMPLETED):
            try:
                p = msg.payload
                goal_id  = p.get("goal_id", "")
                success  = p.get("success", True)
                m = self._in_flight.pop(goal_id, {})
                if not m:
                    continue

                m["completed_at"] = _now()
                m["status"]       = "completed" if success else "failed"

                # Compute latencies
                try:
                    submitted  = _parse(m.get("submitted_at", ""))
                    first_task = _parse(m.get("first_task_at", ""))
                    completed  = _parse(m["completed_at"])
                    if submitted and first_task:
                        m["latency_plan_s"] = (first_task - submitted).total_seconds()
                    if first_task and completed:
                        m["latency_exec_s"] = (completed - first_task).total_seconds()
                except Exception:
                    pass

                # Update fail streak tracking
                cat = m.get("category", "General")
                if success:
                    self._fail_streaks[cat] = 0
                else:
                    streak = self._fail_streaks.get(cat, 0) + 1
                    self._fail_streaks[cat] = streak
                    if streak >= _GAP_FAIL_THRESHOLD:
                        asyncio.create_task(self._flag_capability_gap(cat, m))

                await self._save_metric(m)

                # Quality scoring — fire and forget, non-blocking
                if success and not m.get("is_self_improve"):
                    asyncio.create_task(self._score_quality(m))

            except Exception as exc:
                log.error("IntrospectionBrain: complete_listener error", extra={"e": str(exc)})

    # ── Quality scoring ───────────────────────────────────────────────────────

    async def _score_quality(self, metric: Dict[str, Any]) -> None:
        """Ask gpt-4o to score the synthesis quality for a completed goal."""
        goal_id = metric.get("goal_id", "")
        try:
            # Load the synthesized answer from goal_syntheses
            async with aiosqlite.connect(self._db) as db:
                db.row_factory = aiosqlite.Row
                try:
                    async with db.execute(
                        "SELECT answer FROM goal_syntheses WHERE goal_id = ? LIMIT 1", (goal_id,)
                    ) as c:
                        row = await c.fetchone()
                except Exception:
                    return
            if not row or not row["answer"] or len(row["answer"]) < _QUALITY_MIN_ANSWER:
                return

            answer = row["answer"][:2000]
            prompt = (
                f"Goal: {metric.get('goal_text', '')[:200]}\n\n"
                f"Answer produced:\n{answer}\n\n"
                f"Skills used: {', '.join(metric.get('skill_slugs', []))}\n"
                f"Tasks completed: {metric.get('task_count', 0)}\n\n"
                f"Score this answer (specificity, completeness, usefulness — each 0-10) "
                f"and give one sentence of notes."
            )
            raw = await self._llm.complete(prompt, system_prompt=_QUALITY_SYSTEM)
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(raw)
            scores = [data.get("specificity", 5), data.get("completeness", 5), data.get("usefulness", 5)]
            quality = round(sum(scores) / (len(scores) * 10), 2)  # normalise to 0-1
            notes   = data.get("notes", "")

            async with aiosqlite.connect(self._db) as db:
                await db.execute(
                    "UPDATE goal_metrics SET quality_score=?, quality_notes=? WHERE goal_id=?",
                    (quality, notes, goal_id),
                )
                await db.commit()

            log.info("IntrospectionBrain: quality scored", extra={
                "goal_id": goal_id[:8], "quality": quality, "notes": notes[:60]
            })
        except Exception as exc:
            log.warning("IntrospectionBrain: quality scoring failed", extra={"e": str(exc)})

    # ── Capability gap flagging ───────────────────────────────────────────────

    async def _flag_capability_gap(self, category: str, last_metric: Dict[str, Any]) -> None:
        """When a category hits the fail streak threshold, generate a capability proposal."""
        try:
            prompt = (
                f"Gigia (an AI agent) has failed {_GAP_FAIL_THRESHOLD} consecutive "
                f"'{category}' goals. Last failed goal: '{last_metric.get('goal_text', '')[:200]}'. "
                f"What new SubBot or skill integration would fix this?"
            )
            raw = await self._llm.complete(prompt, system_prompt=_CAPABILITY_GAP_SYSTEM)
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            proposal = json.loads(raw)
            async with aiosqlite.connect(self._db) as db:
                await db.execute(
                    """INSERT INTO capability_gaps (category, proposal, created_at)
                       VALUES (?, ?, ?)""",
                    (category, json.dumps(proposal), _now()),
                )
                await db.commit()
            log.info("IntrospectionBrain: capability gap flagged", extra={
                "category": category, "title": proposal.get("title", "")[:60]
            })
        except Exception as exc:
            log.warning("IntrospectionBrain: gap flagging failed", extra={"e": str(exc)})

    # ── Intelligence briefing ─────────────────────────────────────────────────

    async def _briefing_loop(self) -> None:
        await asyncio.sleep(300)  # first briefing after 5 min warmup
        while True:
            try:
                await self._generate_briefing()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("IntrospectionBrain: briefing error", extra={"e": str(exc)})
            await asyncio.sleep(_ANALYSIS_INTERVAL_S)

    async def _generate_briefing(self) -> None:
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM goal_metrics ORDER BY submitted_at DESC LIMIT ?",
                (_HISTORY_WINDOW,)
            ) as c:
                goals = [dict(r) for r in await c.fetchall()]

        if len(goals) < 3:
            log.info("IntrospectionBrain: not enough data for briefing yet")
            return

        # Aggregate stats for prompt
        cats: Dict[str, Dict] = {}
        for g in goals:
            cat = g.get("category") or "General"
            if cat not in cats:
                cats[cat] = {"count": 0, "success": 0, "quality_sum": 0.0, "q_count": 0,
                             "latency_sum": 0.0, "lat_count": 0}
            cats[cat]["count"] += 1
            if g["status"] == "completed":
                cats[cat]["success"] += 1
            if g.get("quality_score") is not None:
                cats[cat]["quality_sum"] += g["quality_score"]
                cats[cat]["q_count"]     += 1
            total_lat = (g.get("latency_plan_s") or 0) + (g.get("latency_exec_s") or 0)
            if total_lat > 0:
                cats[cat]["latency_sum"] += total_lat
                cats[cat]["lat_count"]   += 1

        cat_lines = []
        for cat, v in sorted(cats.items(), key=lambda x: -x[1]["count"]):
            sr   = f"{v['success']}/{v['count']}"
            qual = f"avg quality {v['quality_sum']/v['q_count']*10:.1f}/10" if v["q_count"] else "no quality data"
            cat_lines.append(f"  {cat}: {sr} success, {qual}")

        failures = [g for g in goals if g["status"] == "failed"]
        fail_lines = [f"  • {g.get('goal_text','')[:80]} [{g.get('category','')}]"
                      for g in failures[:5]] or ["  None"]

        quality_by_cat = {
            cat: f"{v['quality_sum']/v['q_count']*10:.1f}/10"
            for cat, v in cats.items() if v["q_count"] > 0
        } or {"all": "no data yet"}
        quality_lines = [f"  {k}: {v}" for k, v in quality_by_cat.items()]

        lat_lines = []
        for cat, v in cats.items():
            if v["lat_count"] > 0:
                lat_lines.append(f"  {cat}: {v['latency_sum']/v['lat_count']:.1f}s avg")
        if not lat_lines:
            lat_lines = ["  No latency data yet"]

        gap_lines = [f"  {cat}: {n} consecutive failures"
                     for cat, n in self._fail_streaks.items() if n >= 2] or ["  None detected"]

        user_prompt = _BRIEFING_USER_TEMPLATE.format(
            window=len(goals),
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            category_stats="\n".join(cat_lines),
            recent_failures="\n".join(fail_lines),
            quality_stats="\n".join(quality_lines),
            latency_stats="\n".join(lat_lines),
            gap_signals="\n".join(gap_lines),
        )

        raw = await self._llm.complete(user_prompt, system_prompt=_BRIEFING_SYSTEM)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            briefing = json.loads(raw)
        except json.JSONDecodeError:
            # Extract first JSON object as fallback
            m = re.search(r'\{[\s\S]+\}', raw)
            briefing = json.loads(m.group()) if m else {}

        self._latest_briefing = briefing

        async with aiosqlite.connect(self._db) as db:
            await db.execute(
                """INSERT INTO intelligence_briefings
                   (created_at, health_score, primary_use_case, top_bottleneck,
                    top_improvement, capability_gaps, raw)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    _now(),
                    briefing.get("health_score", 0),
                    briefing.get("primary_use_case", ""),
                    briefing.get("top_bottleneck", ""),
                    briefing.get("top_improvement", ""),
                    json.dumps(briefing.get("capability_gaps", [])),
                    raw,
                ),
            )
            await db.commit()

        log.info("IntrospectionBrain: briefing generated", extra={
            "health": briefing.get("health_score"),
            "use_case": str(briefing.get("primary_use_case", ""))[:60],
            "bottleneck": str(briefing.get("top_bottleneck", ""))[:60],
        })

    # ── DB helpers ────────────────────────────────────────────────────────────

    async def _init_db(self) -> None:
        async with aiosqlite.connect(self._db) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS goal_metrics (
                    goal_id         TEXT PRIMARY KEY,
                    goal_text       TEXT,
                    category        TEXT,
                    is_self_improve INTEGER DEFAULT 0,
                    submitted_at    TEXT,
                    planned_at      TEXT,
                    first_task_at   TEXT,
                    completed_at    TEXT,
                    status          TEXT,
                    task_count      INTEGER DEFAULT 0,
                    skill_slugs     TEXT DEFAULT '[]',
                    credits_spent   REAL DEFAULT 0,
                    latency_plan_s  REAL,
                    latency_exec_s  REAL,
                    quality_score   REAL,
                    quality_notes   TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS intelligence_briefings (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at       TEXT,
                    health_score     INTEGER,
                    primary_use_case TEXT,
                    top_bottleneck   TEXT,
                    top_improvement  TEXT,
                    capability_gaps  TEXT,
                    raw              TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS capability_gaps (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    category    TEXT,
                    proposal    TEXT,
                    created_at  TEXT
                )
            """)
            await db.commit()

    async def _save_metric(self, m: Dict[str, Any]) -> None:
        async with aiosqlite.connect(self._db) as db:
            await db.execute(
                """INSERT OR REPLACE INTO goal_metrics
                   (goal_id, goal_text, category, is_self_improve, submitted_at,
                    first_task_at, completed_at, status, task_count,
                    skill_slugs, credits_spent, latency_plan_s, latency_exec_s)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    m.get("goal_id", ""),
                    m.get("goal_text", "")[:500],
                    m.get("category", "General"),
                    m.get("is_self_improve", 0),
                    m.get("submitted_at", ""),
                    m.get("first_task_at", ""),
                    m.get("completed_at", ""),
                    m.get("status", "unknown"),
                    m.get("task_count", 0),
                    json.dumps(m.get("skill_slugs", [])),
                    m.get("credits_spent", 0.0),
                    m.get("latency_plan_s"),
                    m.get("latency_exec_s"),
                ),
            )
            await db.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _parse(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None
