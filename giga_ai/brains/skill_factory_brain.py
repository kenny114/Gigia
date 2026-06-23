"""
skill_factory_brain.py – The ninth brain. Gigia's skill creation engine.

Watches every goal execution and does two things:

1. PATTERN DISCOVERY → COMPOSITION
   Tracks which skill sequences succeed repeatedly. When the same sequence
   wins N times, it composes them into a new single skill and registers it
   with almcp. That skill immediately appears in the catalog — the next
   similar goal uses it as one step instead of four.

2. METADATA IMPROVEMENT
   Reads the full catalog from almcp and finds skills with empty tags,
   vague descriptions, or missing best_used_when. Uses gpt-4o + real
   execution data from SkillBrain to write better metadata, then patches
   it back to almcp.

How it talks to almcp
---------------------
  GET  /api/skills/catalog  — read current global catalog
  PATCH /api/skills/improve — update metadata on existing skill
  POST  /api/skills/compose — register a new composed skill

All calls use X-Giga-Secret header.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

import aiohttp

from giga_ai.messaging.event_bus import EventBus, EventType
from giga_ai.memory.skill_memory import SkillMemory
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)

# ── Tunables ────────────────────────────────────────────────────────────────
_COMPOSE_THRESHOLD  = 3      # succeed this many times → compose
_MIN_SEQUENCE_LEN   = 2      # minimum skills in a composable sequence
_MAX_SEQUENCE_LEN   = 6      # don't compose overly long chains
_IMPROVE_INTERVAL_S = 1800   # improve metadata every 30 minutes
_HTTP_TIMEOUT_S     = 15

# ── Prompts ──────────────────────────────────────────────────────────────────

_COMPOSE_NAME_SYSTEM = """\
You name and describe new composed AI skills. A composed skill is a named pipeline
of existing skills that always runs together to accomplish a specific goal type.
Return ONLY valid JSON — no markdown, no extra text:
{"slug": "snake_case_name", "name": "Display Name", "description": "what it does in one sentence",
 "tags": ["tag1","tag2","tag3"], "best_used_when": "one sentence"}"""

_IMPROVE_SYSTEM = """\
You improve AI skill catalog metadata. Given a skill's name and real usage data,
write better tags, description, and best_used_when.
Return ONLY valid JSON:
{"tags": ["tag1","tag2",...], "description": "...", "best_used_when": "..."}
Keep descriptions under 120 chars. Tags should be single lowercase words or short phrases."""

_GENERATE_SYSTEM = """\
You are a skill generator for Gigia, an autonomous AI agent. Generate a Python skill module.

A skill is a Python module containing exactly ONE function:
  def run(input_data: dict) -> dict

Rules:
- Return a dict with meaningful keys (never None, never a non-dict)
- Return {"error": "..."} on expected failures instead of raising
- Allowed imports: json, re, datetime, math, collections, itertools, requests, httpx, bs4
- FORBIDDEN: eval, exec, compile, open, os.system, os.popen, subprocess.Popen/run/call
- Under 80 lines of code
- JSON-serializable output only (str, int, float, bool, list, dict, None)

Respond with ONLY valid JSON (no markdown fences):
{
  "code": "...full Python code as a string...",
  "slug": "snake_case_name",
  "name": "Display Name",
  "description": "What it does in one sentence (max 120 chars)",
  "tags": ["tag1", "tag2", "tag3"],
  "best_used_when": "one sentence describing ideal use case",
  "input_schema": {
    "type": "object",
    "properties": {
      "field_name": {"type": "string", "description": "..."}
    },
    "required": ["field_name"]
  },
  "credit_cost": 2
}"""

_GENERATE_USER_TEMPLATE = """\
A user tried to accomplish this goal but I had no suitable skill:
  "{goal}"

Available skills couldn't handle it (matched {candidate_count} candidates, none worked).

Generate a new Python skill that could handle this type of goal in the future.
Focus on the core capability gap, not the specific goal — make it reusable."""


class SkillFactoryBrain:
    """
    Ninth brain — discovers successful skill patterns and registers them as
    new composed skills; improves metadata on existing skills.

    Parameters
    ----------
    event_bus        : shared EventBus
    skill_memory     : SkillMemory (already init'd)
    llm_client       : LLMClient
    gateway_url      : almcp base URL, e.g. https://almcp.vercel.app
    giga_secret      : X-Giga-Secret value
    """

    def __init__(
        self,
        event_bus: EventBus,
        skill_memory: SkillMemory,
        llm_client: Any,
        gateway_url: str,
        giga_secret: str,
    ) -> None:
        self._bus          = event_bus
        self._skill_memory = skill_memory
        self._llm          = llm_client
        self._gateway_url  = gateway_url.rstrip("/")
        self._secret       = giga_secret

        # goal_id → list of (slug, success) in execution order
        self._goal_sequences: Dict[str, List[str]] = {}

        # normalized sequence tuple → success count
        self._sequence_counts: Dict[Tuple[str, ...], int] = defaultdict(int)

        # sequences already proposed (avoid re-creating)
        self._composed: Set[Tuple[str, ...]] = set()

        # slugs we've already improved this session
        self._improved: Set[str] = set()

        self._skill_task:    Optional[asyncio.Task] = None
        self._complete_task: Optional[asyncio.Task] = None
        self._improve_task:  Optional[asyncio.Task] = None
        self._gap_task:      Optional[asyncio.Task] = None

        # gap descriptions already being generated (avoid duplicate work)
        self._gaps_in_flight: Set[str] = set()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._skill_task    = asyncio.create_task(self._skill_listener(),    name="factory_skill")
        self._complete_task = asyncio.create_task(self._complete_listener(), name="factory_complete")
        self._improve_task  = asyncio.create_task(self._improve_loop(),      name="factory_improve")
        self._gap_task      = asyncio.create_task(self._gap_listener(),      name="factory_gap")
        log.info("SkillFactoryBrain: started")

    async def stop(self) -> None:
        for t in [self._skill_task, self._complete_task, self._improve_task, self._gap_task]:
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        log.info("SkillFactoryBrain: stopped")

    # ── Pattern tracking ──────────────────────────────────────────────────────

    async def _skill_listener(self) -> None:
        async for msg in self._bus.subscribe(EventType.SKILL_EXECUTED):
            try:
                p = msg.payload
                goal_id = p.get("goal_id", "")
                slug    = p.get("slug", "")
                success = p.get("success", False)
                if goal_id and slug and success:
                    if goal_id not in self._goal_sequences:
                        self._goal_sequences[goal_id] = []
                    if slug not in self._goal_sequences[goal_id]:  # dedupe same skill twice
                        self._goal_sequences[goal_id].append(slug)
            except Exception:
                pass

    async def _complete_listener(self) -> None:
        async for msg in self._bus.subscribe(EventType.GOAL_COMPLETED):
            try:
                p = msg.payload
                goal_id     = p.get("goal_id", "")
                success     = p.get("success", False)
                goal_desc   = p.get("goal_description", "")
                sequence    = self._goal_sequences.pop(goal_id, [])

                if not success or len(sequence) < _MIN_SEQUENCE_LEN:
                    continue
                if len(sequence) > _MAX_SEQUENCE_LEN:
                    continue
                # Skip self-improvement goals — don't compose from them
                if "_self_improvement" in goal_desc.lower():
                    continue

                key = tuple(sequence)
                self._sequence_counts[key] += 1
                count = self._sequence_counts[key]

                log.info("SkillFactoryBrain: sequence recorded", extra={
                    "sequence": sequence, "count": count
                })

                if count >= _COMPOSE_THRESHOLD and key not in self._composed:
                    self._composed.add(key)
                    asyncio.create_task(
                        self._compose(list(key), goal_desc, count),
                        name=f"factory_compose_{key[0][:8]}"
                    )
            except Exception as exc:
                log.error("SkillFactoryBrain: complete_listener error", extra={"e": str(exc)})

    # ── Composition ───────────────────────────────────────────────────────────

    async def _compose(self, sequence: List[str], example_goal: str, success_count: int) -> None:
        """Generate a name/description for this sequence and register it with almcp."""
        log.info("SkillFactoryBrain: composing", extra={"sequence": sequence, "count": success_count})

        # Ask gpt-4o to name it
        prompt = (
            f"Skill sequence that succeeded {success_count} times: {' → '.join(sequence)}\n"
            f"Example goal it accomplished: '{example_goal[:200]}'\n\n"
            f"Name this composed skill pipeline."
        )
        try:
            raw = await self._llm.complete(prompt, system_prompt=_COMPOSE_NAME_SYSTEM)
            raw = _strip_fences(raw)
            meta = json.loads(raw)
        except Exception as exc:
            log.error("SkillFactoryBrain: naming failed", extra={"e": str(exc)})
            return

        slug = meta.get("slug", "")
        if not slug or not re.match(r'^[a-z0-9_]+$', slug):
            slug = "composed_" + "_".join(s.split("_")[0] for s in sequence[:3])

        # Build workflow with output_as for all but last step
        workflow = []
        for i, skill in enumerate(sequence):
            step: Dict[str, str] = {"skill": skill}
            if i < len(sequence) - 1:
                step["output_as"] = f"step_{i}_{skill.split('_')[0]}"
            workflow.append(step)

        payload = {
            "slug":            slug,
            "name":            meta.get("name", slug.replace("_", " ").title()),
            "description":     meta.get("description", f"Composed: {' → '.join(sequence)}"),
            "tags":            meta.get("tags", []),
            "best_used_when":  meta.get("best_used_when", ""),
            "workflow":        workflow,
            "discovered_from": example_goal[:300],
            "success_count":   success_count,
        }

        ok, resp = await self._post("/api/skills/compose", payload)
        if ok:
            log.info("SkillFactoryBrain: composed skill registered", extra={
                "slug": slug, "steps": len(sequence), "credits": resp.get("credit_cost")
            })
        elif resp.get("error") == "slug_taken":
            log.info("SkillFactoryBrain: composition already exists", extra={"slug": slug})
        else:
            log.warning("SkillFactoryBrain: compose failed", extra={"resp": resp})

    # ── Metadata improvement ──────────────────────────────────────────────────

    async def _improve_loop(self) -> None:
        await asyncio.sleep(300)  # warmup
        while True:
            try:
                await self._run_improvement_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("SkillFactoryBrain: improve_loop error", extra={"e": str(exc)})
            await asyncio.sleep(_IMPROVE_INTERVAL_S)

    async def _run_improvement_cycle(self) -> None:
        """Find catalog skills with weak metadata and improve them."""
        ok, data = await self._get("/api/skills/catalog")
        if not ok:
            return

        catalog = data.get("skills", [])
        improved = 0

        for skill in catalog:
            slug = skill.get("slug", "")
            if slug in self._improved:
                continue

            tags = skill.get("tags") or []
            desc = skill.get("description") or ""
            buw  = skill.get("best_used_when") or ""

            needs_improvement = (
                len(tags) == 0 or
                len(desc) < 20 or
                not buw
            )
            if not needs_improvement:
                continue

            # Enrich with real execution data from SkillMemory
            profiles = await self._skill_memory.get_profiles([slug])
            profile  = profiles.get(slug, {})
            use_cases    = profile.get("use_cases", [])
            output_keys  = profile.get("output_keys", [])
            reliability  = profile.get("reliability", None)

            prompt = (
                f"Skill name: {skill.get('name', slug)}\n"
                f"Current description: {desc or '(none)'}\n"
                f"Current tags: {tags or '(none)'}\n"
                f"Real use cases observed: {use_cases[:5] or '(no data yet)'}\n"
                f"Output fields it returns: {output_keys[:10] or '(unknown)'}\n"
                f"Reliability: {f'{reliability*100:.0f}%' if reliability else 'unknown'}\n\n"
                f"Write better tags, description, and best_used_when for this skill."
            )

            try:
                raw  = await self._llm.complete(prompt, system_prompt=_IMPROVE_SYSTEM)
                raw  = _strip_fences(raw)
                meta = json.loads(raw)
            except Exception as exc:
                log.warning("SkillFactoryBrain: metadata generation failed", extra={"slug": slug, "e": str(exc)})
                continue

            updates: Dict[str, Any] = {}
            if meta.get("tags") and len(meta["tags"]) > len(tags):
                updates["tags"] = meta["tags"]
            if meta.get("description") and len(meta["description"]) > len(desc):
                updates["description"] = meta["description"]
            if meta.get("best_used_when") and not buw:
                updates["best_used_when"] = meta["best_used_when"]

            if not updates:
                self._improved.add(slug)
                continue

            ok2, resp = await self._patch("/api/skills/improve", {
                "slug": slug,
                "updates": updates,
                "reason": f"SkillFactoryBrain improvement — {len(use_cases)} real use cases observed",
            })
            if ok2:
                self._improved.add(slug)
                improved += 1
                log.info("SkillFactoryBrain: improved skill", extra={
                    "slug": slug, "fields": list(updates.keys())
                })
            else:
                log.warning("SkillFactoryBrain: improve failed", extra={"slug": slug, "resp": resp})

            await asyncio.sleep(1)  # don't hammer the API

        if improved:
            log.info("SkillFactoryBrain: improvement cycle done", extra={"improved": improved})

    # ── Gap detection → code generation ──────────────────────────────────────

    async def _gap_listener(self) -> None:
        async for msg in self._bus.subscribe(EventType.SKILL_GAP_DETECTED):
            try:
                p = msg.payload
                goal_desc      = p.get("goal_description", "")
                candidate_count = p.get("candidate_count", 0)
                if not goal_desc:
                    continue
                # Dedupe: don't regenerate for a very similar gap in-flight
                key = goal_desc[:120].lower()
                if key in self._gaps_in_flight:
                    continue
                self._gaps_in_flight.add(key)
                asyncio.create_task(
                    self._generate_from_gap(goal_desc, candidate_count),
                    name="factory_generate",
                )
            except Exception as exc:
                log.error("SkillFactoryBrain: gap_listener error", extra={"e": str(exc)})

    async def _generate_from_gap(self, goal_desc: str, candidate_count: int) -> None:
        """Generate a new Python skill for an identified capability gap."""
        log.info("SkillFactoryBrain: generating skill for gap", extra={
            "goal": goal_desc[:80], "candidates": candidate_count
        })

        prompt = _GENERATE_USER_TEMPLATE.format(
            goal=goal_desc[:300],
            candidate_count=candidate_count,
        )
        try:
            raw = await self._llm.complete(prompt, system_prompt=_GENERATE_SYSTEM)
            raw = _strip_fences(raw)
            meta = json.loads(raw)
        except Exception as exc:
            log.error("SkillFactoryBrain: skill generation LLM failed", extra={"e": str(exc)})
            self._gaps_in_flight.discard(goal_desc[:120].lower())
            return

        slug = meta.get("slug", "")
        code = meta.get("code", "")
        if not slug or not code:
            log.warning("SkillFactoryBrain: generation returned no slug/code")
            self._gaps_in_flight.discard(goal_desc[:120].lower())
            return

        if not re.match(r'^[a-z0-9_]+$', slug):
            slug = re.sub(r'[^a-z0-9_]', '_', slug.lower())

        # 1. Store skill code on the VPS executor
        ok_vps, resp_vps = await self._post("/skills/register", {
            "slug":     slug,
            "code":     code,
            "metadata": {k: v for k, v in meta.items() if k != "code"},
        })
        if not ok_vps:
            log.error("SkillFactoryBrain: VPS skill register failed", extra={
                "slug": slug, "resp": resp_vps
            })
            self._gaps_in_flight.discard(goal_desc[:120].lower())
            return

        # 2. Register the endpoint in almcp catalog
        ok_catalog, resp_catalog = await self._post("/api/skills/generate", {
            "slug":           slug,
            "name":           meta.get("name", slug.replace("_", " ").title()),
            "description":    meta.get("description", f"Generated: {goal_desc[:80]}"),
            "tags":           meta.get("tags", []),
            "best_used_when": meta.get("best_used_when", ""),
            "avoid_when":     meta.get("avoid_when", ""),
            "input_schema":   meta.get("input_schema"),
            "credit_cost":    meta.get("credit_cost", 2),
            "generated_from": goal_desc[:300],
        })

        if ok_catalog:
            log.info("SkillFactoryBrain: new skill generated and registered", extra={
                "slug": slug, "credits": resp_catalog.get("credit_cost")
            })
        elif resp_catalog.get("error") == "slug_taken":
            log.info("SkillFactoryBrain: generated slug already exists", extra={"slug": slug})
        else:
            log.warning("SkillFactoryBrain: catalog registration failed", extra={
                "slug": slug, "resp": resp_catalog
            })

        self._gaps_in_flight.discard(goal_desc[:120].lower())

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        return {"X-Giga-Secret": self._secret, "Content-Type": "application/json"}

    async def _get(self, path: str) -> Tuple[bool, dict]:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"{self._gateway_url}{path}",
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_S),
                ) as r:
                    data = await r.json()
                    return r.status < 300, data
        except Exception as exc:
            log.error("SkillFactoryBrain: GET failed", extra={"path": path, "e": str(exc)})
            return False, {}

    async def _post(self, path: str, body: dict) -> Tuple[bool, dict]:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{self._gateway_url}{path}",
                    json=body,
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_S),
                ) as r:
                    data = await r.json()
                    return r.status < 300, data
        except Exception as exc:
            log.error("SkillFactoryBrain: POST failed", extra={"path": path, "e": str(exc)})
            return False, {}

    async def _patch(self, path: str, body: dict) -> Tuple[bool, dict]:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.patch(
                    f"{self._gateway_url}{path}",
                    json=body,
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_S),
                ) as r:
                    data = await r.json()
                    return r.status < 300, data
        except Exception as exc:
            log.error("SkillFactoryBrain: PATCH failed", extra={"path": path, "e": str(exc)})
            return False, {}


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return s
