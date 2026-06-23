"""
executor.py — Generated skill storage, lifecycle management, and sandboxed execution.

Lifecycle:  draft → staged → active → disabled

  draft    : code stored, not yet validated or tested
  staged   : passed AST check + local test run; usable by Gigia for current goal only
  active   : ran successfully in production with valid output; registered in almcp catalog
  disabled : failed any validation, test, or runtime check

Security model:
  - Strict AST validation blocks shell execution, filesystem writes, dangerous builtins
  - Subprocess runs with empty env (no secrets leak) and a temp working directory
  - 30-second timeout per execution
  - Non-root execution is enforced by Docker container config (see Dockerfile)
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

from giga_ai.utils.logger import get_logger

log = get_logger(__name__)

_HARNESS_PATH = str(pathlib.Path(__file__).parent / "run_harness.py")
_EXEC_TIMEOUT_S = 30

# Module-level db path — set once from api.py on startup so sub-modules
# (SkillSubBot, SkillFactoryBrain) can access it without passing it everywhere.
_db_path: str = ""


def set_db_path(path: str) -> None:
    global _db_path
    _db_path = path


def get_db_path() -> str:
    return _db_path


# ── AST safety rules ──────────────────────────────────────────────────────────

# Imports that are always blocked, regardless of allow_network.
_BLOCKED_IMPORTS: frozenset[str] = frozenset({
    "os", "sys", "subprocess", "socket", "pathlib", "shutil",
    "ctypes", "importlib", "pickle", "shelve", "glob",
    "pty", "pty", "resource", "signal", "threading", "multiprocessing",
    "tempfile", "io", "builtins",
})

# Imports blocked unless allow_network=True.
_NETWORK_IMPORTS: frozenset[str] = frozenset({
    "requests", "urllib", "httpx", "aiohttp", "http", "ftplib",
    "smtplib", "imaplib", "poplib", "xmlrpc",
})

# Individual builtins always blocked.
_BLOCKED_BUILTINS: frozenset[str] = frozenset({
    "eval", "exec", "compile", "__import__", "open",
    "globals", "locals", "vars",
    "breakpoint", "input",
})

# Attribute access patterns that indicate dangerous calls.
_BLOCKED_ATTRS: frozenset[str] = frozenset({
    # os / subprocess shell execution
    "system", "popen", "execve", "execvp", "execl", "execle", "execlp",
    "spawnv", "spawnve", "fork", "kill", "killpg",
    # subprocess variants
    "Popen", "call", "run", "check_output", "check_call",
    # environment variable access
    "environ", "getenv", "putenv", "unsetenv",
    # filesystem write
    "remove", "unlink", "rmdir", "makedirs", "mkdir",
    "rename", "replace", "symlink", "link",
    # ctypes / dynamic loading
    "cdll", "windll", "CDLL",
    # importlib dynamic loading
    "import_module", "spec_from_file_location", "exec_module",
})


class SkillSafetyError(Exception):
    """Raised when generated code fails AST safety validation."""


def validate_skill_code(
    code: str,
    slug: str,
    allow_network: bool = False,
) -> None:
    """
    Parse and AST-walk the generated code. Raises SkillSafetyError on any
    blocked pattern. Does not execute anything.

    allow_network=True permits requests / httpx / urllib imports.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SkillSafetyError(f"syntax error at line {e.lineno}: {e.msg}") from e

    blocked_imports = _BLOCKED_IMPORTS
    if not allow_network:
        blocked_imports = blocked_imports | _NETWORK_IMPORTS

    for node in ast.walk(tree):
        # import foo / import foo.bar
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in blocked_imports:
                    raise SkillSafetyError(f"blocked import: {alias.name}")

        # from foo import bar
        if isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in blocked_imports:
                    raise SkillSafetyError(f"blocked import: {node.module}")

        # Blocked built-in names: eval, exec, open, globals, locals …
        if isinstance(node, ast.Name) and node.id in _BLOCKED_BUILTINS:
            raise SkillSafetyError(f"blocked builtin: {node.id}()")

        # Blocked attribute access: os.system, subprocess.Popen, os.environ …
        if isinstance(node, ast.Attribute) and node.attr in _BLOCKED_ATTRS:
            raise SkillSafetyError(
                f"blocked attribute access: .{node.attr} — "
                "shell execution and filesystem writes are not permitted"
            )

        # Double-underscore dunder abuse: __class__.__bases__ etc.
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            if node.attr not in {"__init__", "__repr__", "__str__", "__len__",
                                  "__iter__", "__next__", "__getitem__", "__setitem__",
                                  "__contains__", "__enter__", "__exit__",
                                  "__name__", "__doc__", "__file__"}:
                raise SkillSafetyError(f"blocked dunder access: {node.attr}")

    # Must define a run() function
    has_run = any(
        isinstance(n, ast.FunctionDef) and n.name == "run"
        for n in ast.walk(tree)
    )
    if not has_run:
        raise SkillSafetyError(
            "skill must define  def run(input_data: dict) -> dict"
        )

    log.info("executor: AST validation passed", extra={"slug": slug})


# ── DB helpers ────────────────────────────────────────────────────────────────

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS generated_skills (
    slug            TEXT    PRIMARY KEY,
    name            TEXT    NOT NULL DEFAULT '',
    description     TEXT    NOT NULL DEFAULT '',
    code            TEXT    NOT NULL,
    code_hash       TEXT    NOT NULL,
    input_schema    TEXT    DEFAULT NULL,
    output_schema   TEXT    DEFAULT NULL,
    status          TEXT    NOT NULL DEFAULT 'draft',
    safety_status   TEXT    NOT NULL DEFAULT 'pending',
    last_test_result TEXT   DEFAULT NULL,
    endpoint_url    TEXT    DEFAULT NULL,
    version         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    promoted_at     TEXT    DEFAULT NULL,
    created_by      TEXT    NOT NULL DEFAULT 'skill_factory_brain',
    allow_network   INTEGER NOT NULL DEFAULT 0
)
"""


async def _ensure_table(db: aiosqlite.Connection) -> None:
    await db.execute(_CREATE_SQL)
    await db.commit()


async def init_db(db_path: str) -> None:
    """Create the generated_skills table if it doesn't exist."""
    async with aiosqlite.connect(db_path) as db:
        await _ensure_table(db)


async def _get_row(db_path: str, slug: str) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        await _ensure_table(db)
        row = await (await db.execute(
            "SELECT * FROM generated_skills WHERE slug = ?", (slug,)
        )).fetchone()
    return dict(row) if row else None


async def _set_status(
    db_path: str,
    slug: str,
    status: str,
    *,
    safety_status: Optional[str] = None,
    last_test_result: Optional[dict] = None,
    promoted_at: Optional[str] = None,
    endpoint_url: Optional[str] = None,
) -> None:
    updates: List[str] = ["status = ?"]
    params: List[Any] = [status]
    if safety_status is not None:
        updates.append("safety_status = ?"); params.append(safety_status)
    if last_test_result is not None:
        updates.append("last_test_result = ?"); params.append(json.dumps(last_test_result))
    if promoted_at is not None:
        updates.append("promoted_at = ?"); params.append(promoted_at)
    if endpoint_url is not None:
        updates.append("endpoint_url = ?"); params.append(endpoint_url)
    params.append(slug)
    async with aiosqlite.connect(db_path) as db:
        await _ensure_table(db)
        await db.execute(
            f"UPDATE generated_skills SET {', '.join(updates)} WHERE slug = ?",
            params,
        )
        await db.commit()


# ── Public lifecycle API ──────────────────────────────────────────────────────

async def create_draft(
    db_path: str,
    slug: str,
    code: str,
    *,
    name: str = "",
    description: str = "",
    input_schema: Optional[dict] = None,
    output_schema: Optional[dict] = None,
    allow_network: bool = False,
    created_by: str = "skill_factory_brain",
) -> None:
    """Store a new skill as draft (no validation yet)."""
    code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
    async with aiosqlite.connect(db_path) as db:
        await _ensure_table(db)
        await db.execute(
            """INSERT OR REPLACE INTO generated_skills
               (slug, name, description, code, code_hash, input_schema, output_schema,
                status, safety_status, created_at, created_by, allow_network)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'draft', 'pending', datetime('now'), ?, ?)""",
            (
                slug, name, description, code, code_hash,
                json.dumps(input_schema) if input_schema else None,
                json.dumps(output_schema) if output_schema else None,
                created_by,
                1 if allow_network else 0,
            ),
        )
        await db.commit()
    log.info("executor: draft created", extra={"slug": slug, "hash": code_hash})


def _generate_test_input(input_schema: Optional[dict]) -> dict:
    """Build minimal synthetic test input from a JSON Schema object."""
    if not input_schema:
        return {}
    props = input_schema.get("properties", {})
    test: dict = {}
    for key, defn in props.items():
        typ = defn.get("type", "string")
        if typ == "string":
            test[key] = f"test_{key}"
        elif typ in ("number", "integer"):
            test[key] = 1
        elif typ == "boolean":
            test[key] = False
        elif typ == "array":
            test[key] = []
        elif typ == "object":
            test[key] = {}
        else:
            test[key] = f"test_{key}"
    return test


async def validate_and_stage(
    db_path: str,
    slug: str,
) -> Tuple[bool, str]:
    """
    Transition draft → staged (or draft → disabled).

    1. AST validation with the skill's allow_network flag
    2. Local test run with synthetic input
    3. Output schema check

    Returns (True, slug) on success, (False, error_message) on failure.
    """
    row = await _get_row(db_path, slug)
    if not row:
        return False, "skill not found"
    if row["status"] != "draft":
        return False, f"expected status=draft, got {row['status']}"

    code = row["code"]
    allow_network = bool(row.get("allow_network", 0))
    input_schema = json.loads(row["input_schema"]) if row["input_schema"] else None
    output_schema = json.loads(row["output_schema"]) if row["output_schema"] else None

    # ── 1. AST validation ────────────────────────────────────────────────────
    try:
        validate_skill_code(code, slug, allow_network=allow_network)
    except SkillSafetyError as exc:
        reason = str(exc)
        await _set_status(
            db_path, slug, "disabled",
            safety_status="failed",
            last_test_result={"phase": "ast", "error": reason},
        )
        log.warning("executor: AST validation failed — disabling skill",
                    extra={"slug": slug, "reason": reason})
        return False, f"AST validation failed: {reason}"

    # ── 2. Test run ──────────────────────────────────────────────────────────
    test_input = _generate_test_input(input_schema)
    ok, result = await _run_in_subprocess(code, test_input, allow_network=allow_network)

    test_record: dict = {
        "phase": "test_run",
        "ok": ok,
        "test_input": test_input,
        "result": result if ok else None,
        "error": result if not ok else None,
    }

    if not ok:
        await _set_status(
            db_path, slug, "disabled",
            safety_status="passed",  # AST passed; runtime failed
            last_test_result=test_record,
        )
        log.warning("executor: test run failed — disabling skill",
                    extra={"slug": slug, "error": str(result)[:200]})
        return False, f"test run failed: {result}"

    # ── 3. Output schema check ───────────────────────────────────────────────
    if output_schema and isinstance(result, dict):
        required = output_schema.get("required", [])
        missing = [k for k in required if k not in result]
        if missing:
            test_record["schema_error"] = f"missing required output keys: {missing}"
            await _set_status(
                db_path, slug, "disabled",
                safety_status="passed",
                last_test_result=test_record,
            )
            log.warning("executor: output schema mismatch — disabling skill",
                        extra={"slug": slug, "missing": missing})
            return False, f"output schema mismatch: missing {missing}"

    test_record["schema_ok"] = True

    now = datetime.now(timezone.utc).isoformat()
    await _set_status(
        db_path, slug, "staged",
        safety_status="passed",
        last_test_result=test_record,
        promoted_at=now,
    )
    log.info("executor: skill staged", extra={"slug": slug})
    return True, slug


async def promote_to_active(db_path: str, slug: str, endpoint_url: Optional[str] = None) -> bool:
    """Transition staged → active (called after successful production run)."""
    row = await _get_row(db_path, slug)
    if not row or row["status"] != "staged":
        return False
    now = datetime.now(timezone.utc).isoformat()
    await _set_status(db_path, slug, "active", promoted_at=now, endpoint_url=endpoint_url)
    log.info("executor: skill promoted to active", extra={"slug": slug})
    return True


async def disable_skill(db_path: str, slug: str, reason: str) -> None:
    """Mark a skill disabled from any state."""
    await _set_status(
        db_path, slug, "disabled",
        last_test_result={"disabled_reason": reason, "at": datetime.now(timezone.utc).isoformat()},
    )
    log.warning("executor: skill disabled", extra={"slug": slug, "reason": reason[:200]})


async def get_skill_status(db_path: str, slug: str) -> Optional[str]:
    """Return the skill's current lifecycle status, or None if not found."""
    row = await _get_row(db_path, slug)
    return row["status"] if row else None


async def get_staged_metadata(db_path: str, slug: str) -> Optional[Dict[str, Any]]:
    """Return metadata for a staged skill (used by PlanningBrain to build a candidate)."""
    row = await _get_row(db_path, slug)
    if not row or row["status"] != "staged":
        return None
    return {
        "slug":           row["slug"],
        "name":           row["name"] or slug.replace("_", " ").title(),
        "description":    row["description"] or f"Generated skill: {slug}",
        "input_schema":   json.loads(row["input_schema"]) if row["input_schema"] else None,
        "output_schema":  json.loads(row["output_schema"]) if row["output_schema"] else None,
        "code_hash":      row["code_hash"],
        "status":         row["status"],
        "safety_status":  row["safety_status"],
        "created_at":     row["created_at"],
        "created_by":     row["created_by"],
        "allow_network":  bool(row.get("allow_network", 0)),
    }


async def list_skills(db_path: str) -> List[Dict[str, Any]]:
    """Return all generated skills ordered by creation time."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        await _ensure_table(db)
        rows = await (await db.execute(
            """SELECT slug, name, description, code_hash, status, safety_status,
                      last_test_result, endpoint_url, version, created_at, promoted_at,
                      created_by, allow_network
               FROM generated_skills ORDER BY created_at DESC"""
        )).fetchall()
    return [
        {
            "slug":           r["slug"],
            "name":           r["name"],
            "description":    r["description"],
            "code_hash":      r["code_hash"],
            "status":         r["status"],
            "safety_status":  r["safety_status"],
            "last_test_result": json.loads(r["last_test_result"]) if r["last_test_result"] else None,
            "endpoint_url":   r["endpoint_url"],
            "version":        r["version"],
            "created_at":     r["created_at"],
            "promoted_at":    r["promoted_at"],
            "created_by":     r["created_by"],
            "allow_network":  bool(r["allow_network"]),
        }
        for r in rows
    ]


# ── Execution engine ──────────────────────────────────────────────────────────

async def _run_in_subprocess(
    code: str,
    input_data: dict,
    *,
    allow_network: bool = False,
) -> Tuple[bool, Any]:
    """
    Execute code in an isolated subprocess.
    - Empty env (no secrets accessible)
    - Temporary working directory per execution
    - 30s timeout, process killed on expiry
    - stdout → JSON result; stderr captured for debugging
    """
    with tempfile.TemporaryDirectory(prefix="gigia_exec_") as exec_dir:
        # Write skill code to a temp file inside the isolated dir
        skill_file = os.path.join(exec_dir, "_skill.py")
        with open(skill_file, "w") as f:
            f.write(code)

        input_json = json.dumps(input_data).encode()

        # Minimal safe env: only PATH so Python can find its stdlib
        safe_env: Dict[str, str] = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        }
        # Allow network libs if explicitly approved (they're already in stdlib/site-packages)
        # No extra env needed — requests etc. are importable without secrets

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            _HARNESS_PATH,
            skill_file,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=exec_dir,   # isolated working dir
            env=safe_env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=input_json),
                timeout=_EXEC_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            log.warning("executor: subprocess timed out", extra={"timeout": _EXEC_TIMEOUT_S})
            return False, f"skill timed out after {_EXEC_TIMEOUT_S}s"

        raw = stdout.decode().strip()
        if not raw:
            err_preview = stderr.decode()[:500]
            return False, f"no output from skill: {err_preview or '(empty stderr)'}"

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return False, f"skill returned non-JSON: {raw[:300]}"

        if not payload.get("ok"):
            return False, payload.get("error", "unknown error")

        return True, payload.get("result", {})


async def run_skill(
    db_path: str,
    slug: str,
    input_data: dict,
    *,
    expected_status: str = "staged",  # "staged" | "active" | None (any)
) -> Tuple[bool, Any]:
    """
    Execute a skill by slug. By default only runs staged/active skills.
    Returns (ok, result_dict_or_error_string).
    """
    row = await _get_row(db_path, slug)
    if not row:
        return False, f"skill not found: {slug}"
    if expected_status and row["status"] not in (expected_status, "staged", "active"):
        return False, f"skill {slug} is {row['status']}, not executable"

    ok, result = await _run_in_subprocess(
        row["code"],
        input_data,
        allow_network=bool(row.get("allow_network", 0)),
    )

    # Increment call counter
    if ok:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "UPDATE generated_skills SET version = version WHERE slug = ?", (slug,)
            )
            await db.commit()

    return ok, result


# ── Legacy store_skill (used by /skills/register HTTP endpoint) ───────────────

async def store_skill(db_path: str, slug: str, code: str, metadata: dict) -> None:
    """
    Backward-compat: store code + metadata as a draft.
    Called from the /skills/register HTTP endpoint in api.py.
    """
    await create_draft(
        db_path, slug, code,
        name=metadata.get("name", ""),
        description=metadata.get("description", ""),
        input_schema=metadata.get("input_schema"),
        output_schema=metadata.get("output_schema"),
        allow_network=bool(metadata.get("allow_network", False)),
        created_by=metadata.get("created_by", "http_register"),
    )
