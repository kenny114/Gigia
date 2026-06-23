"""
test_lifecycle.py — Quick smoke test for the generated-skill lifecycle.

Covers:
  1. AST validation blocks dangerous patterns
  2. AST validation passes safe code
  3. create_draft stores metadata correctly
  4. validate_and_stage: draft -> staged on valid code
  5. validate_and_stage: draft -> disabled on code that fails the test run
  6. validate_and_stage: draft -> disabled on code with blocked imports
  7. validate_and_stage: draft -> disabled when run() is missing
  8. run_skill: staged skill executes and returns correct output
  9. promote_to_active: staged -> active
 10. get_skill_status / get_staged_metadata
 11. disable_skill: any -> disabled
 12. Module-level set_db_path / get_db_path
 13. /skills/run/{slug} endpoint rejects non-active skills

Run:  python test_lifecycle.py
"""

import asyncio
import os
import sys
import tempfile
import traceback

# Make sure imports resolve from project root
sys.path.insert(0, os.path.dirname(__file__))

from giga_ai.skills.executor import (
    SkillSafetyError,
    create_draft,
    disable_skill,
    get_db_path,
    get_skill_status,
    get_staged_metadata,
    init_db,
    list_skills,
    promote_to_active,
    run_skill,
    set_db_path,
    validate_and_stage,
    validate_skill_code,
)

# -- Helpers --------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok))
    icon = PASS if ok else FAIL
    line = f"  [{icon}] {name}"
    if detail:
        line += f"  <- {detail}"
    print(line)


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * 50)


# -- Safe / unsafe code fixtures -----------------------------------------------

SAFE_CODE = """\
import json
import re

def run(input_data: dict) -> dict:
    text = input_data.get("text", "")
    words = len(text.split())
    return {"word_count": words, "char_count": len(text)}
"""

SAFE_CODE_WITH_SCHEMA = """\
import json
import re
import math

def run(input_data: dict) -> dict:
    n = int(input_data.get("n", 10))
    primes = []
    for candidate in range(2, n + 1):
        if all(candidate % i != 0 for i in range(2, int(math.sqrt(candidate)) + 1)):
            primes.append(candidate)
    return {"primes": primes, "count": len(primes)}
"""

BROKEN_CODE = """\
def run(input_data: dict) -> dict:
    raise RuntimeError("I always fail")
"""

CODE_WITH_OS_IMPORT = """\
import os

def run(input_data: dict) -> dict:
    return {"cwd": os.getcwd()}
"""

CODE_WITH_SUBPROCESS = """\
import subprocess

def run(input_data: dict) -> dict:
    result = subprocess.run(["echo", "hi"], capture_output=True)
    return {"output": result.stdout.decode()}
"""

CODE_WITH_OPEN = """\
def run(input_data: dict) -> dict:
    with open("/etc/passwd") as f:
        return {"contents": f.read()[:100]}
"""

CODE_WITH_EVAL = """\
def run(input_data: dict) -> dict:
    return {"result": eval(input_data.get("expr", "1+1"))}
"""

CODE_MISSING_RUN = """\
import json

def process(data):
    return {"ok": True}
"""

CODE_WITH_BAD_DUNDER = """\
def run(input_data: dict) -> dict:
    cls = input_data.__class__
    bases = cls.__bases__
    return {"bases": str(bases)}
"""


async def run_tests():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_skills.db")
        await init_db(db_path)

        # -- Section 1: AST validation -----------------------------------------
        section("1. AST validation — blocked patterns")

        for name, code, pattern in [
            ("blocks os import",         CODE_WITH_OS_IMPORT,     "blocked import: os"),
            ("blocks subprocess import", CODE_WITH_SUBPROCESS,    "blocked import: subprocess"),
            ("blocks open() builtin",    CODE_WITH_OPEN,          "blocked builtin: open"),
            ("blocks eval() builtin",    CODE_WITH_EVAL,          "blocked builtin: eval"),
            ("blocks missing run()",     CODE_MISSING_RUN,        "must define"),
            ("blocks bad dunder access", CODE_WITH_BAD_DUNDER,    "blocked dunder"),
        ]:
            try:
                validate_skill_code(code, "test_slug")
                check(name, False, "expected SkillSafetyError — none raised")
            except SkillSafetyError as e:
                check(name, True, str(e)[:80])
            except Exception as e:
                check(name, False, f"wrong exception: {e}")

        section("2. AST validation — safe code passes")
        try:
            validate_skill_code(SAFE_CODE, "safe_slug")
            check("safe stdlib imports pass", True)
        except SkillSafetyError as e:
            check("safe stdlib imports pass", False, str(e))

        try:
            validate_skill_code(SAFE_CODE_WITH_SCHEMA, "safe_schema_slug")
            check("safe code with math/re passes", True)
        except SkillSafetyError as e:
            check("safe code with math/re passes", False, str(e))

        # -- Section 3: Lifecycle transitions ---------------------------------
        section("3. create_draft")

        await create_draft(
            db_path, "word_counter", SAFE_CODE,
            name="Word Counter",
            description="Counts words and chars in text",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            output_schema={"type": "object", "properties": {"word_count": {"type": "integer"}, "char_count": {"type": "integer"}}, "required": ["word_count", "char_count"]},
        )
        status = await get_skill_status(db_path, "word_counter")
        check("draft stored with status=draft", status == "draft", f"got {status}")

        section("4. validate_and_stage — valid skill -> staged")

        ok, msg = await validate_and_stage(db_path, "word_counter")
        check("validate_and_stage returns True", ok, msg)
        check("validate_and_stage returns slug", msg == "word_counter", f"got '{msg}'")
        status = await get_skill_status(db_path, "word_counter")
        check("status is now staged", status == "staged", f"got {status}")

        meta = await get_staged_metadata(db_path, "word_counter")
        check("get_staged_metadata returns dict", meta is not None)
        check("metadata has correct name", meta and meta["name"] == "Word Counter", str(meta))
        check("metadata has safety_status=passed", meta and meta["safety_status"] == "passed", str(meta))

        section("5. validate_and_stage — broken skill -> disabled")

        await create_draft(db_path, "broken_skill", BROKEN_CODE,
                           name="Broken", description="Always fails")
        ok, msg = await validate_and_stage(db_path, "broken_skill")
        check("broken skill: validate_and_stage returns False", not ok, msg)
        status = await get_skill_status(db_path, "broken_skill")
        check("broken skill: status is disabled", status == "disabled", f"got {status}")

        section("6. validate_and_stage — blocked import -> disabled")

        await create_draft(db_path, "os_skill", CODE_WITH_OS_IMPORT,
                           name="OS Skill", description="Tries to use os")
        ok, msg = await validate_and_stage(db_path, "os_skill")
        check("os import skill: validate_and_stage returns False", not ok, msg)
        status = await get_skill_status(db_path, "os_skill")
        check("os import skill: status is disabled", status == "disabled", f"got {status}")

        section("7. run_skill — execute staged skill")

        ok, result = await run_skill(db_path, "word_counter", {"text": "hello world foo bar"})
        check("run_skill returns True", ok, str(result) if not ok else "")
        check("run_skill returns word_count", isinstance(result, dict) and "word_count" in result, str(result))
        check("word_count is correct (4)", result.get("word_count") == 4, f"got {result}")
        check("char_count is correct (19)", result.get("char_count") == 19, f"got {result}")

        section("8. run_skill — primes skill with output schema")

        await create_draft(
            db_path, "find_primes", SAFE_CODE_WITH_SCHEMA,
            name="Find Primes",
            description="Find prime numbers up to N",
            input_schema={"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
            output_schema={"type": "object", "properties": {"primes": {"type": "array"}, "count": {"type": "integer"}}, "required": ["primes", "count"]},
        )
        ok, msg = await validate_and_stage(db_path, "find_primes")
        check("primes skill: staged", ok, msg)

        ok, result = await run_skill(db_path, "find_primes", {"n": 20})
        check("primes skill: run returns True", ok, str(result))
        check("primes up to 20: count=8", result.get("count") == 8, f"got {result}")
        check("primes up to 20: [2,3,5,7,11,13,17,19]",
              result.get("primes") == [2, 3, 5, 7, 11, 13, 17, 19],
              f"got {result.get('primes')}")

        section("9. promote_to_active")

        ok = await promote_to_active(db_path, "word_counter", endpoint_url="registered:word_counter")
        check("promote_to_active returns True", ok)
        status = await get_skill_status(db_path, "word_counter")
        check("status is now active", status == "active", f"got {status}")

        # get_staged_metadata should return None for active (it's no longer staged)
        meta = await get_staged_metadata(db_path, "word_counter")
        check("get_staged_metadata returns None for active skill", meta is None)

        section("10. run_skill on active skill")

        ok, result = await run_skill(db_path, "word_counter", {"text": "one two three"}, expected_status="active")
        check("active skill: run returns True", ok, str(result))
        check("active skill: word_count=3", result.get("word_count") == 3, str(result))

        section("11. disable_skill")

        await disable_skill(db_path, "find_primes", "test: explicit disable")
        status = await get_skill_status(db_path, "find_primes")
        check("disabled skill: status=disabled", status == "disabled", f"got {status}")

        # Disabled skill should not run
        ok, err = await run_skill(db_path, "find_primes", {"n": 10})
        check("disabled skill: run returns False", not ok, str(err)[:80])

        section("12. Module-level set_db_path / get_db_path")

        original = get_db_path()
        set_db_path("/tmp/test_path.db")
        check("set_db_path: get_db_path returns new value", get_db_path() == "/tmp/test_path.db")
        set_db_path(original)
        check("set_db_path: restored original", get_db_path() == original)

        section("13. list_skills")

        skills = await list_skills(db_path)
        slugs = [s["slug"] for s in skills]
        check("list_skills: all slugs present",
              all(s in slugs for s in ["word_counter", "broken_skill", "os_skill", "find_primes"]),
              str(slugs))
        statuses = {s["slug"]: s["status"] for s in skills}
        check("list_skills: word_counter is active",   statuses.get("word_counter") == "active")
        check("list_skills: find_primes is disabled",  statuses.get("find_primes") == "disabled")
        check("list_skills: broken_skill is disabled", statuses.get("broken_skill") == "disabled")

        section("14. validate_and_stage rejects non-draft")

        ok, msg = await validate_and_stage(db_path, "word_counter")
        check("validate_and_stage rejects active skill", not ok, msg)

        section("15. run_skill rejects draft (not yet staged/active)")

        await create_draft(db_path, "unvalidated", SAFE_CODE,
                           name="Unvalidated", description="Draft only")
        ok, err = await run_skill(db_path, "unvalidated", {})
        check("run_skill rejects draft skill", not ok, str(err)[:80])

    # -- Summary ---------------------------------------------------------------
    total  = len(_results)
    passed = sum(1 for _, ok in _results if ok)
    failed = total - passed
    print(f"\n{'-' * 50}")
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
        ok = asyncio.run(run_tests())
        sys.exit(0 if ok else 1)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
