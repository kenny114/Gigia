"""
global_memory.py – SQLite-backed persistent memory store using aiosqlite.

Schema
------
  strategies
    entry_id TEXT PRIMARY KEY
    problem  TEXT
    solution TEXT
    success_rate REAL
    created_at   TEXT   (ISO-8601)
    updated_at   TEXT
    tags         TEXT   (JSON array)
    metadata     TEXT   (JSON object)

  failure_logs
    log_id       TEXT PRIMARY KEY
    problem      TEXT
    context      TEXT   (JSON object)
    error_details TEXT  (JSON object)
    logged_at    TEXT
    resolved     INTEGER  (0/1)
    resolution_notes TEXT

Usage
-----
    memory = GlobalMemory("giga_ai.db")
    await memory.init()

    await memory.store_strategy("proxy blocked", "rotate proxy", 0.8)
    entry = await memory.get_strategy("proxy blocked")
    await memory.update_success_rate("proxy blocked", 0.9)
    await memory.log_failure("captcha", {"task_id": "t-1"}, {"type": "CaptchaDetected"})
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiosqlite

from giga_ai.memory.schemas import FailureLog, StrategyRecord
from giga_ai.messaging.message_schemas import MemoryEntry
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)


def _utcnow_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_strategy(row: aiosqlite.Row) -> StrategyRecord:
    return StrategyRecord(
        entry_id=row["entry_id"],
        problem=row["problem"],
        solution=row["solution"],
        success_rate=float(row["success_rate"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        tags=json.loads(row["tags"] or "[]"),
        metadata=json.loads(row["metadata"] or "{}"),
    )


def _row_to_memory_entry(row: aiosqlite.Row) -> MemoryEntry:
    return MemoryEntry(
        entry_id=row["entry_id"],
        problem=row["problem"],
        solution=row["solution"],
        success_rate=float(row["success_rate"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        tags=json.loads(row["tags"] or "[]"),
        metadata=json.loads(row["metadata"] or "{}"),
    )


class GlobalMemory:
    """
    Async SQLite-backed persistent memory store.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created automatically if absent.
    """

    def __init__(self, db_path: str = "giga_ai.db") -> None:
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Open the database and create tables if they don't exist."""
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._create_tables()
        log.info("GlobalMemory initialised", extra={"db_path": self._db_path})

    async def close(self) -> None:
        """Close the underlying database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def _create_tables(self) -> None:
        assert self._db is not None
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS strategies (
                entry_id     TEXT PRIMARY KEY,
                problem      TEXT NOT NULL,
                solution     TEXT NOT NULL,
                success_rate REAL NOT NULL DEFAULT 0.0,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                tags         TEXT DEFAULT '[]',
                metadata     TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_strategies_problem
                ON strategies(problem);

            CREATE TABLE IF NOT EXISTS failure_logs (
                log_id            TEXT PRIMARY KEY,
                problem           TEXT NOT NULL,
                context           TEXT DEFAULT '{}',
                error_details     TEXT DEFAULT '{}',
                logged_at         TEXT NOT NULL,
                resolved          INTEGER NOT NULL DEFAULT 0,
                resolution_notes  TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_failure_logs_problem
                ON failure_logs(problem);
        """)
        await self._db.commit()

    def _ensure_open(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("GlobalMemory not initialised – call await memory.init() first.")
        return self._db

    # ------------------------------------------------------------------
    # Strategy operations
    # ------------------------------------------------------------------

    async def store_strategy(
        self,
        problem: str,
        solution: str,
        success_rate: float,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StrategyRecord:
        """
        Insert a new strategy record.

        If a strategy for *problem* already exists it is updated in-place
        (success_rate and solution are overwritten).
        """
        db = self._ensure_open()
        now = _utcnow_str()
        entry_id = str(uuid.uuid4())
        tags_json = json.dumps(tags or [])
        meta_json = json.dumps(metadata or {})

        # Upsert: if problem already exists, update it
        existing = await self.get_strategy(problem)
        if existing:
            await db.execute(
                """UPDATE strategies
                   SET solution=?, success_rate=?, updated_at=?, tags=?, metadata=?
                   WHERE problem=?""",
                (solution, success_rate, now, tags_json, meta_json, problem),
            )
            await db.commit()
            updated = await self.get_strategy(problem)
            log.info("GlobalMemory: strategy updated", extra={"problem": problem})
            return updated  # type: ignore[return-value]

        await db.execute(
            """INSERT INTO strategies
               (entry_id, problem, solution, success_rate, created_at, updated_at, tags, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (entry_id, problem, solution, success_rate, now, now, tags_json, meta_json),
        )
        await db.commit()
        log.info("GlobalMemory: strategy stored", extra={"problem": problem, "entry_id": entry_id})

        return StrategyRecord(
            entry_id=entry_id,
            problem=problem,
            solution=solution,
            success_rate=success_rate,
            tags=tags or [],
            metadata=metadata or {},
        )

    async def get_strategy(self, problem: str) -> Optional[StrategyRecord]:
        """
        Retrieve the strategy record for *problem*.

        Returns ``None`` if no matching record exists.
        """
        db = self._ensure_open()
        async with db.execute(
            "SELECT * FROM strategies WHERE problem = ? LIMIT 1", (problem,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_strategy(row)

    async def update_success_rate(self, problem: str, new_rate: float) -> bool:
        """
        Update the success_rate for *problem*.

        Returns ``True`` if a row was updated, ``False`` if not found.
        """
        db = self._ensure_open()
        now = _utcnow_str()
        cursor = await db.execute(
            "UPDATE strategies SET success_rate=?, updated_at=? WHERE problem=?",
            (new_rate, now, problem),
        )
        await db.commit()
        updated = cursor.rowcount > 0
        if updated:
            log.info("GlobalMemory: success_rate updated", extra={"problem": problem, "new_rate": new_rate})
        return updated

    async def get_all_strategies(self) -> List[MemoryEntry]:
        """Return all strategy records as MemoryEntry Pydantic models."""
        db = self._ensure_open()
        async with db.execute("SELECT * FROM strategies ORDER BY success_rate DESC") as cursor:
            rows = await cursor.fetchall()
        return [_row_to_memory_entry(r) for r in rows]

    async def query_strategies_by_keyword(self, keyword: str) -> List[MemoryEntry]:
        """
        Simple full-text search: return strategies whose *problem* or
        *solution* contains *keyword* (case-insensitive).
        """
        db = self._ensure_open()
        pattern = f"%{keyword}%"
        async with db.execute(
            """SELECT * FROM strategies
               WHERE problem LIKE ? OR solution LIKE ?
               ORDER BY success_rate DESC""",
            (pattern, pattern),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_memory_entry(r) for r in rows]

    # ------------------------------------------------------------------
    # Failure log operations
    # ------------------------------------------------------------------

    async def log_failure(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> FailureLog:
        """Insert a new failure log entry."""
        db = self._ensure_open()
        log_id = str(uuid.uuid4())
        now = _utcnow_str()
        ctx_json = json.dumps(context or {})
        err_json = json.dumps(error_details or {})

        await db.execute(
            """INSERT INTO failure_logs
               (log_id, problem, context, error_details, logged_at, resolved, resolution_notes)
               VALUES (?, ?, ?, ?, ?, 0, NULL)""",
            (log_id, problem, ctx_json, err_json, now),
        )
        await db.commit()
        log.info("GlobalMemory: failure logged", extra={"problem": problem, "log_id": log_id})

        return FailureLog(
            log_id=log_id,
            problem=problem,
            context=context or {},
            error_details=error_details or {},
        )

    async def resolve_failure(self, log_id: str, notes: Optional[str] = None) -> bool:
        """Mark a failure log entry as resolved."""
        db = self._ensure_open()
        cursor = await db.execute(
            "UPDATE failure_logs SET resolved=1, resolution_notes=? WHERE log_id=?",
            (notes, log_id),
        )
        await db.commit()
        return cursor.rowcount > 0

    async def get_unresolved_failures(self) -> List[FailureLog]:
        """Return all unresolved failure log entries."""
        db = self._ensure_open()
        async with db.execute(
            "SELECT * FROM failure_logs WHERE resolved=0 ORDER BY logged_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()

        result = []
        for row in rows:
            result.append(FailureLog(
                log_id=row["log_id"],
                problem=row["problem"],
                context=json.loads(row["context"] or "{}"),
                error_details=json.loads(row["error_details"] or "{}"),
                logged_at=datetime.fromisoformat(row["logged_at"]),
                resolved=bool(row["resolved"]),
                resolution_notes=row["resolution_notes"],
            ))
        return result
