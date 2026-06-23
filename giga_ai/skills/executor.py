"""
executor.py — Generated skill storage and sandboxed execution.

Generated skills are Python functions stored as code in a local SQLite table.
Execution happens in a subprocess via run_harness.py, which provides:
  - Process isolation (crash can't bring down Gigia)
  - No parent env vars (API keys / secrets can't be exfiltrated)
  - 30-second timeout (prevents runaway generated code)
  - JSON stdio interface (clean data handoff)

Before storage, every skill is AST-validated to block the most dangerous
patterns (eval, exec, os.system, subprocess). Network access is allowed so
skills can make HTTP calls to public APIs.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import sys
import tempfile
import pathlib
from typing import Any, Dict, Optional, Tuple

import aiosqlite

from giga_ai.utils.logger import get_logger

log = get_logger(__name__)

_HARNESS_PATH = str(pathlib.Path(__file__).parent / "run_harness.py")
_EXEC_TIMEOUT_S = 30

# Builtin names and attribute access patterns that indicate dangerous code.
# We block file I/O (skills must be stateless) and shell execution.
# We do NOT block network access (httpx/requests are legitimate).
_BLOCKED_BUILTINS = {"eval", "exec", "compile", "__import__", "open"}
_BLOCKED_ATTRS    = {
    # os module shell execution
    "system", "popen", "execve", "execvp", "execl", "execle", "execlp",
    "spawnv", "spawnve",
    # subprocess module
    "Popen", "call", "run", "check_output", "check_call",
    # ctypes / dynamic loading
    "cdll", "windll",
}


class SkillSafetyError(Exception):
    """Raised when generated code fails AST safety validation."""


def validate_skill_code(code: str, slug: str) -> None:
    """
    AST-validate generated skill code. Raises SkillSafetyError if dangerous
    patterns are detected. Also verifies that a run() function is defined.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SkillSafetyError(f"syntax error: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _BLOCKED_BUILTINS:
            raise SkillSafetyError(f"blocked builtin '{node.id}' — use safe alternatives")
        if isinstance(node, ast.Attribute) and node.attr in _BLOCKED_ATTRS:
            raise SkillSafetyError(f"blocked attribute '{node.attr}' — shell execution not permitted")

    has_run = any(
        isinstance(node, ast.FunctionDef) and node.name == "run"
        for node in ast.walk(tree)
    )
    if not has_run:
        raise SkillSafetyError("skill must define  def run(input_data: dict) -> dict")

    log.info("executor: skill code validated", extra={"slug": slug})


async def _ensure_table(db: aiosqlite.Connection) -> None:
    await db.execute("""
        CREATE TABLE IF NOT EXISTS generated_skills (
            slug        TEXT PRIMARY KEY,
            code        TEXT NOT NULL,
            metadata    TEXT NOT NULL DEFAULT '{}',
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            call_count  INTEGER NOT NULL DEFAULT 0
        )
    """)
    await db.commit()


async def store_skill(db_path: str, slug: str, code: str, metadata: dict) -> None:
    """Store a generated skill after AST validation. Overwrites existing."""
    validate_skill_code(code, slug)
    async with aiosqlite.connect(db_path) as db:
        await _ensure_table(db)
        await db.execute(
            """INSERT OR REPLACE INTO generated_skills (slug, code, metadata, created_at)
               VALUES (?, ?, ?, datetime('now'))""",
            (slug, code, json.dumps(metadata)),
        )
        await db.commit()
    log.info("executor: skill stored", extra={"slug": slug, "bytes": len(code)})


async def load_skill_code(db_path: str, slug: str) -> Optional[str]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        await _ensure_table(db)
        row = await (await db.execute(
            "SELECT code FROM generated_skills WHERE slug = ?", (slug,)
        )).fetchone()
    return row["code"] if row else None


async def list_skills(db_path: str) -> list:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        await _ensure_table(db)
        rows = await (await db.execute(
            "SELECT slug, metadata, created_at, call_count FROM generated_skills ORDER BY created_at DESC"
        )).fetchall()
    return [
        {
            "slug":       r["slug"],
            "metadata":   json.loads(r["metadata"]),
            "created_at": r["created_at"],
            "call_count": r["call_count"],
        }
        for r in rows
    ]


async def run_skill(
    db_path: str,
    slug: str,
    input_data: dict,
) -> Tuple[bool, Any]:
    """
    Execute a generated skill in a sandboxed subprocess.
    Returns (ok, result_dict) or (False, error_string).
    """
    code = await load_skill_code(db_path, slug)
    if code is None:
        return False, f"skill not found: {slug}"

    # Write code to a temp file; run_harness.py loads it via importlib.
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix=f"gigia_{slug}_",
        delete=False,
    ) as tf:
        tf.write(code)
        tmp_path = tf.name

    try:
        input_json = json.dumps(input_data).encode()

        # env={} — no parent env vars passed, so no secrets can leak
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            _HARNESS_PATH,
            tmp_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={},
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=input_json),
                timeout=_EXEC_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            log.warning("executor: skill timed out", extra={"slug": slug})
            return False, f"skill timed out after {_EXEC_TIMEOUT_S}s"

        raw = stdout.decode().strip()
        if not raw:
            err = stderr.decode()[:500]
            log.warning("executor: skill no output", extra={"slug": slug, "stderr": err})
            return False, f"skill produced no output: {err or '(empty stderr)'}"

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return False, f"skill returned non-JSON: {raw[:300]}"

        if not payload.get("ok"):
            err = payload.get("error", "unknown error")
            log.warning("executor: skill failed", extra={"slug": slug, "error": err[:200]})
            return False, err

        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "UPDATE generated_skills SET call_count = call_count + 1 WHERE slug = ?",
                (slug,),
            )
            await db.commit()

        log.info("executor: skill succeeded", extra={"slug": slug})
        return True, payload.get("result", {})

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
