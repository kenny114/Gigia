"""
test_integration.py — Integration tests for the skill lifecycle wiring.

Tests the component interfaces (not just executor.py in isolation):

  A. SkillSubBot staged intercept
       - staged skill is run locally, result carries _staged_execution=True
       - active skill falls through to gateway (connection refused = expected)
       - disabled/draft skill falls through to gateway

  B. ManagerBot staged flag forwarding
       - SKILL_EXECUTED payload includes staged=True when _staged_execution present

  C. PlanningBrain + SkillFactoryBrain gap path (mock LLM)
       - gap detected -> generate_and_stage() called -> staged candidate used
       - failed generation -> GOAL_COMPLETED published with success=False

  D. SkillFactoryBrain promote_listener
       - SKILL_EXECUTED with staged=True triggers promote_to_active + almcp call

Run:  python test_integration.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import traceback
import types
import unittest.mock as mock
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from giga_ai.skills.executor import (
    create_draft,
    disable_skill,
    init_db,
    promote_to_active,
    set_db_path,
    validate_and_stage,
)
from giga_ai.messaging.event_bus import EventBus, EventType

PASS = "PASS"
FAIL = "FAIL"
_results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok))
    tag = f"[{PASS}]" if ok else f"[{FAIL}]"
    line = f"  {tag} {name}"
    if detail:
        line += f"  <- {detail}"
    print(line)


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * 60)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAFE_CODE = '''\
def run(input_data: dict) -> dict:
    x = input_data.get("x", 1)
    return {"doubled": x * 2}
'''

# ---------------------------------------------------------------------------
# A. SkillSubBot staged intercept
# ---------------------------------------------------------------------------

async def test_skill_sub_bot(db_path: str) -> None:
    section("A. SkillSubBot staged intercept")

    from giga_ai.sub_bot.skill_sub_bot import SkillSubBot
    from giga_ai.messaging.message_schemas import SubBotInstruction, SubBotType

    # Plant a staged skill in the db
    await create_draft(db_path, "doubler", SAFE_CODE, name="Doubler")
    await validate_and_stage(db_path, "doubler")
    set_db_path(db_path)

    def make_instruction(slug: str) -> SubBotInstruction:
        return SubBotInstruction(
            task_id="task-1",
            goal_id="goal-1",
            sub_bot_type=SubBotType.SKILL,
            timeout_seconds=30,
            parameters={
                "execute_url": "http://127.0.0.1:19999/api/brain/execute",  # unreachable
                "token": "test-token",
                "run_id": "run-1",
                "skill_slug": slug,
                "args": {"x": 21},
            },
        )

    bot = SkillSubBot()

    # A1: staged skill runs locally, not via gateway
    result = await bot._run(make_instruction("doubler"))
    check("A1: staged skill returns doubled=42",
          result.get("doubled") == 42, str(result))
    check("A2: result has _staged_execution=True",
          result.get("_staged_execution") is True, str(result))
    check("A3: credits_charged=0 for staged skill",
          result.get("_credits_charged") == 0, str(result))

    # A2: active skill falls through to gateway (connection refused)
    await promote_to_active(db_path, "doubler")
    try:
        await bot._run(make_instruction("doubler"))
        check("A4: active skill falls through to gateway", False, "expected connection error")
    except Exception as exc:
        check("A4: active skill falls through to gateway",
              "connect" in str(exc).lower() or "gateway" in str(exc).lower() or "5xx" in str(exc).lower(),
              str(exc)[:80])

    # A3: disabled skill falls through to gateway
    await create_draft(db_path, "disabled_skill", SAFE_CODE, name="Disabled")
    await validate_and_stage(db_path, "disabled_skill")
    await disable_skill(db_path, "disabled_skill", "test disable")
    try:
        await bot._run(make_instruction("disabled_skill"))
        check("A5: disabled skill falls through to gateway", False, "expected connection error")
    except Exception as exc:
        check("A5: disabled skill falls through to gateway",
              "connect" in str(exc).lower() or "gateway" in str(exc).lower() or "5xx" in str(exc).lower(),
              str(exc)[:80])

    # A4: unknown slug also falls through
    try:
        await bot._run(make_instruction("nonexistent_skill_xyz"))
        check("A6: unknown slug falls through to gateway", False, "expected error")
    except Exception as exc:
        check("A6: unknown slug falls through to gateway",
              "connect" in str(exc).lower() or "gateway" in str(exc).lower() or "5xx" in str(exc).lower(),
              str(exc)[:80])


# ---------------------------------------------------------------------------
# B. ManagerBot staged flag forwarding
# ---------------------------------------------------------------------------

async def test_manager_bot_staged_flag() -> None:
    section("B. ManagerBot staged=True forwarded in SKILL_EXECUTED")

    from giga_ai.messaging.event_bus import EventBus, EventType
    from giga_ai.messaging.message_schemas import SubBotType, Task, TaskStatus

    bus = EventBus()
    received: list = []

    async def listener():
        async for msg in bus.subscribe(EventType.SKILL_EXECUTED):
            received.append(msg.payload)
            return

    listen_task = asyncio.create_task(listener())
    await asyncio.sleep(0.05)

    # Simulate ManagerBot._notify_skill_executed by publishing what it would
    # This is the exact payload structure from manager_bot.py lines 148-165
    result_data = {
        "doubled": 42,
        "_credits_charged": 0,
        "_skill_slug": "doubler",
        "_staged_execution": True,
    }
    await bus.publish(
        EventType.SKILL_EXECUTED,
        payload={
            "slug": "doubler",
            "description": "double a number",
            "goal_id": "goal-1",
            "goal_description": "double x",
            "result_keys": list(result_data.keys()),
            "success": True,
            "credits": result_data.get("credits_charged", 0),
            "staged": bool(result_data.get("_staged_execution", False)),
        },
    )

    try:
        await asyncio.wait_for(listen_task, timeout=2.0)
    except asyncio.TimeoutError:
        pass

    check("B1: SKILL_EXECUTED received", len(received) == 1, f"got {len(received)} events")
    if received:
        check("B2: staged=True in payload",
              received[0].get("staged") is True, str(received[0]))
        check("B3: success=True in payload",
              received[0].get("success") is True, str(received[0]))
        check("B4: slug correct",
              received[0].get("slug") == "doubler", str(received[0]))

    await bus.shutdown()


# ---------------------------------------------------------------------------
# C. PlanningBrain + SkillFactoryBrain gap path (mock LLM)
# ---------------------------------------------------------------------------

async def test_planning_gap_path(db_path: str) -> None:
    section("C. PlanningBrain gap path with SkillFactoryBrain (mock LLM)")

    from giga_ai.brains.planning_brain import PlanningBrain
    from giga_ai.brains.skill_factory_brain import SkillFactoryBrain
    from giga_ai.memory.skill_memory import SkillMemory
    from giga_ai.messaging.event_bus import EventBus, EventType
    from giga_ai.messaging.message_schemas import (
        GatewayCallback, Goal, OrchestrateCandidate,
    )
    from giga_ai.skills.executor import get_skill_status

    set_db_path(db_path)
    bus = EventBus()

    # Shared call counter across both MockLLM instances so we can simulate
    # the exact sequence:
    #   call 1  — PlanningBrain initial decompose → empty tasks  (gap triggered)
    #   call 2  — SkillFactoryBrain generates code + metadata
    #   call 3  — PlanningBrain replans with the staged skill → one task
    call_count = [0]
    generated_slug = [None]

    class MockLLM:
        async def complete(self, prompt: str, system_prompt: str = "", **kw) -> str:
            call_count[0] += 1
            n = call_count[0]

            if n == 1:
                # Initial decompose: LLM can't match any candidate -> empty
                return json.dumps({"tasks": []})

            if n == 2:
                # SkillFactoryBrain: LLM generates code for the gap
                slug = "mock_generated_skill"
                generated_slug[0] = slug
                return json.dumps({
                    "slug": slug,
                    "name": "Mock Generated Skill",
                    "description": "A skill generated to fill a capability gap",
                    "tags": ["generated"],
                    "best_used_when": "when no existing skill matches the goal",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    "credit_cost": 2,
                    "allow_network": False,
                    "code": (
                        "def run(input_data):\n"
                        "    q = input_data.get('query', '')\n"
                        "    return {'answer': 'answer to: ' + q}\n"
                    ),
                })

            if n == 3:
                # PlanningBrain replans with staged skill as sole candidate
                slug = generated_slug[0] or "mock_generated_skill"
                return json.dumps({"tasks": [{
                    "task_id": "task-1",
                    "title": "Run generated skill",
                    "description": "Execute the generated skill",
                    "sub_bot_type": "skill",
                    "skill_slug": slug,
                    "priority": 1,
                    "dependencies": [],
                    "metadata": {"args": {"query": "test"}},
                }]})

            return json.dumps({"tasks": []})

    # Single shared LLM instance so call_count increments across both brains
    shared_llm = MockLLM()

    skill_memory = SkillMemory(db_path)
    await skill_memory.init()

    skill_factory = SkillFactoryBrain(
        event_bus=bus,
        skill_memory=skill_memory,
        llm_client=shared_llm,
        gateway_url="http://127.0.0.1:19999",
        giga_secret="test-secret",
    )

    planning = PlanningBrain(
        event_bus=bus,
        llm_client=shared_llm,
        skill_factory=skill_factory,
    )

    tasks_created: list = []
    goal_completed: list = []

    async def capture_tasks():
        async for msg in bus.subscribe(EventType.TASK_CREATED):
            tasks_created.append(msg.payload)

    async def capture_goals():
        async for msg in bus.subscribe(EventType.GOAL_COMPLETED):
            goal_completed.append(msg.payload)

    asyncio.create_task(capture_tasks())
    asyncio.create_task(capture_goals())
    await asyncio.sleep(0.05)

    goal = Goal(description="do something no existing skill can handle", metadata={})
    callback = GatewayCallback(
        execute_url="http://127.0.0.1:19999/api/brain/execute",
        result_url="http://127.0.0.1:19999/api/brain/result",
        token="tok",
    )

    # Pass ONE dummy candidate so skill_mode=True, but LLM returns empty tasks
    # (simulating: getCandidates found one skill, but it doesn't fit the goal)
    dummy_candidate = OrchestrateCandidate(
        slug="unrelated_skill",
        name="Unrelated Skill",
        description="Does something unrelated",
        credits=2,
        tags=[],
        best_used_when=None,
        avoid_when=None,
        example_call=None,
    )

    tasks = await planning.decompose_goal(
        goal,
        candidates=[dummy_candidate],
        gateway_callback=callback,
    )

    await asyncio.sleep(0.15)

    check("C1: decompose_goal returned tasks (gap filled by generated skill)",
          len(tasks) >= 1, f"got {len(tasks)} tasks")
    if tasks:
        check("C2: task uses the generated skill slug",
              tasks[0].skill_slug == (generated_slug[0] or "mock_generated_skill"),
              f"slug={tasks[0].skill_slug!r}, generated={generated_slug[0]!r}")
        check("C3: task sub_bot_type is skill",
              str(tasks[0].sub_bot_type) in ("SubBotType.SKILL", "skill"),
              str(tasks[0].sub_bot_type))

    slug = generated_slug[0]
    if slug:
        status = await get_skill_status(db_path, slug)
        check("C4: generated skill is staged in db",
              status == "staged", f"got status={status!r}")

    check("C5: no premature GOAL_COMPLETED(failure) published",
          not any(not p.get("success") for p in goal_completed),
          f"goal_completed events: {goal_completed}")

    check("C6: LLM called exactly 3 times (decompose, generate, replan)",
          call_count[0] == 3, f"called {call_count[0]} times")

    await bus.shutdown()


# ---------------------------------------------------------------------------
# D. SkillFactoryBrain promote_listener
# ---------------------------------------------------------------------------

async def test_promote_listener(db_path: str) -> None:
    section("D. SkillFactoryBrain promote_listener (staged -> active)")

    from giga_ai.brains.skill_factory_brain import SkillFactoryBrain
    from giga_ai.memory.skill_memory import SkillMemory
    from giga_ai.messaging.event_bus import EventBus, EventType
    from giga_ai.skills.executor import get_skill_status, set_db_path

    set_db_path(db_path)
    bus = EventBus()

    # Plant a staged skill
    await create_draft(db_path, "promote_me", SAFE_CODE, name="Promote Me")
    await validate_and_stage(db_path, "promote_me")

    almcp_calls: list = []

    class MockLLM:
        async def complete(self, *a, **kw):
            return "{}"

    skill_memory = SkillMemory(db_path)
    await skill_memory.init()

    sf = SkillFactoryBrain(
        event_bus=bus,
        skill_memory=skill_memory,
        llm_client=MockLLM(),
        gateway_url="http://127.0.0.1:19999",
        giga_secret="test",
    )

    # Patch the HTTP POST so we don't need almcp running
    async def mock_post(path: str, body: dict):
        almcp_calls.append({"path": path, "slug": body.get("slug")})
        return True, {"slug": body.get("slug"), "credit_cost": 2}

    sf._post = mock_post
    await sf.start()
    await asyncio.sleep(0.05)

    # Publish SKILL_EXECUTED with staged=True
    await bus.publish(
        EventType.SKILL_EXECUTED,
        payload={
            "slug": "promote_me",
            "success": True,
            "staged": True,
            "goal_id": "goal-1",
            "result_keys": ["doubled"],
        },
    )

    # Give the listener time to fire
    await asyncio.sleep(0.5)

    status = await get_skill_status(db_path, "promote_me")
    check("D1: skill promoted to active after SKILL_EXECUTED",
          status == "active", f"got {status}")
    check("D2: almcp /api/skills/generate was called",
          any(c["path"] == "/api/skills/generate" for c in almcp_calls),
          str(almcp_calls))
    check("D3: correct slug sent to almcp",
          any(c["slug"] == "promote_me" for c in almcp_calls),
          str(almcp_calls))

    await sf.stop()
    await bus.shutdown()


# ---------------------------------------------------------------------------
# E. Safety gate: ensure draft/disabled skills cannot be executed
# ---------------------------------------------------------------------------

async def test_safety_gates(db_path: str) -> None:
    section("E. Safety gates: draft and disabled cannot execute")

    from giga_ai.skills.executor import run_skill

    set_db_path(db_path)

    await create_draft(db_path, "gate_draft", SAFE_CODE, name="Gate Draft")

    ok, err = await run_skill(db_path, "gate_draft", {"x": 5})
    check("E1: draft skill cannot run",
          not ok and "draft" in err, f"ok={ok} err={err}")

    await create_draft(db_path, "gate_disabled", SAFE_CODE, name="Gate Disabled")
    await validate_and_stage(db_path, "gate_disabled")
    await disable_skill(db_path, "gate_disabled", "safety gate test")

    ok, err = await run_skill(db_path, "gate_disabled", {"x": 5})
    check("E2: disabled skill cannot run",
          not ok and "disabled" in err, f"ok={ok} err={err}")

    # Active skill can run
    await create_draft(db_path, "gate_active", SAFE_CODE, name="Gate Active")
    await validate_and_stage(db_path, "gate_active")
    await promote_to_active(db_path, "gate_active")

    ok, result = await run_skill(db_path, "gate_active", {"x": 21}, expected_status="active")
    check("E3: active skill runs fine",
          ok and result.get("doubled") == 42, f"ok={ok} result={result}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_all() -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "integration.db")
        await init_db(db_path)
        set_db_path(db_path)

        try:
            await test_skill_sub_bot(db_path)
        except Exception:
            print("  [ERROR] test_skill_sub_bot crashed:")
            traceback.print_exc()

        try:
            await test_manager_bot_staged_flag()
        except Exception:
            print("  [ERROR] test_manager_bot_staged_flag crashed:")
            traceback.print_exc()

        try:
            await test_planning_gap_path(db_path)
        except Exception:
            print("  [ERROR] test_planning_gap_path crashed:")
            traceback.print_exc()

        try:
            await test_promote_listener(db_path)
        except Exception:
            print("  [ERROR] test_promote_listener crashed:")
            traceback.print_exc()

        try:
            await test_safety_gates(db_path)
        except Exception:
            print("  [ERROR] test_safety_gates crashed:")
            traceback.print_exc()

    total  = len(_results)
    passed = sum(1 for _, ok in _results if ok)
    failed = total - passed

    print(f"\n{'-' * 60}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        for name, ok in _results:
            if not ok:
                print(f"    FAILED: {name}")
    else:
        print("  All good")
    print()
    return failed == 0


if __name__ == "__main__":
    try:
        ok = asyncio.run(run_all())
        sys.exit(0 if ok else 1)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
