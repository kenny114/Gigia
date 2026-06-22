"""
skill_memory.py – Persistent skill knowledge store.

Tracks every skill execution, builds profiles over time, and records
which skills work well together. Used by SkillBrain to enrich planning.

Schema
------
  skill_profiles
    slug             TEXT PRIMARY KEY
    description      TEXT
    output_keys      TEXT   (JSON array  — keys seen in result dicts)
    use_cases        TEXT   (JSON array  — recent goal fragments)
    success_count    INTEGER
    fail_count       INTEGER
    avg_credits      REAL
    reliability      REAL   (0-1, success_count / total)
    last_seen        TEXT   (ISO-8601)
    notes            TEXT   (free-form observations)

  skill_interactions
    skill_a          TEXT
    skill_b          TEXT
    sequence_count   INTEGER  (a ran immediately before b in same goal)
    co_count         INTEGER  (both appeared in same goal, any order)
    success_count    INTEGER  (goal completed successfully)
    PRIMARY KEY (skill_a, skill_b)

  skill_sequences
    seq_hash         TEXT PRIMARY KEY
    sequence         TEXT   (JSON array of slugs, ordered)
    goal_type        TEXT   (first 80 chars of goal)
    count            INTEGER
    success_count    INTEGER
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiosqlite

from giga_ai.utils.logger import get_logger

log = get_logger(__name__)

_MAX_USE_CASES = 10   # keep the most recent N goal fragments per skill
_MAX_OUTPUT_KEYS = 20 # cap output_keys tracked per skill


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class SkillMemory:
    """SQLite-backed store of accumulated skill knowledge."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS skill_profiles (
                    slug          TEXT PRIMARY KEY,
                    description   TEXT DEFAULT '',
                    output_keys   TEXT DEFAULT '[]',
                    use_cases     TEXT DEFAULT '[]',
                    success_count INTEGER DEFAULT 0,
                    fail_count    INTEGER DEFAULT 0,
                    avg_credits   REAL    DEFAULT 0,
                    reliability   REAL    DEFAULT 0.5,
                    last_seen     TEXT,
                    notes         TEXT    DEFAULT ''
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS skill_interactions (
                    skill_a        TEXT NOT NULL,
                    skill_b        TEXT NOT NULL,
                    sequence_count INTEGER DEFAULT 0,
                    co_count       INTEGER DEFAULT 1,
                    success_count  INTEGER DEFAULT 0,
                    PRIMARY KEY (skill_a, skill_b)
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS skill_sequences (
                    seq_hash      TEXT PRIMARY KEY,
                    sequence      TEXT NOT NULL,
                    goal_type     TEXT DEFAULT '',
                    count         INTEGER DEFAULT 1,
                    success_count INTEGER DEFAULT 0
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_sp_reliability ON skill_profiles(reliability DESC)"
            )
            await db.commit()
        log.info("SkillMemory: tables ready")

    # ------------------------------------------------------------------
    # Record a single skill execution
    # ------------------------------------------------------------------

    async def record_execution(
        self,
        slug: str,
        description: str,
        goal_description: str,
        result_keys: List[str],
        success: bool,
        credits: float = 0,
    ) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row

            row = await (await db.execute(
                "SELECT * FROM skill_profiles WHERE slug = ?", (slug,)
            )).fetchone()

            if row is None:
                await db.execute(
                    """INSERT INTO skill_profiles
                       (slug, description, output_keys, use_cases,
                        success_count, fail_count, avg_credits, reliability, last_seen)
                       VALUES (?,?,?,?, ?,?,?,?,?)""",
                    (
                        slug, description,
                        json.dumps(result_keys[:_MAX_OUTPUT_KEYS]),
                        json.dumps([goal_description[:80]]),
                        1 if success else 0,
                        0 if success else 1,
                        credits, 1.0 if success else 0.0,
                        _utcnow(),
                    ),
                )
            else:
                sc = row["success_count"] + (1 if success else 0)
                fc = row["fail_count"] + (0 if success else 1)
                total = sc + fc
                reliability = sc / total if total > 0 else 0.5

                # Merge output keys
                existing_keys = json.loads(row["output_keys"] or "[]")
                merged_keys = list(dict.fromkeys(existing_keys + result_keys))[:_MAX_OUTPUT_KEYS]

                # Rolling average credits
                old_avg = row["avg_credits"] or 0
                new_avg = old_avg + (credits - old_avg) / total if total > 0 else credits

                # Keep most recent use cases
                cases = json.loads(row["use_cases"] or "[]")
                fragment = goal_description[:80]
                if fragment not in cases:
                    cases = ([fragment] + cases)[:_MAX_USE_CASES]

                # Use provided description if we have one and stored is empty
                desc = description if description else row["description"]

                await db.execute(
                    """UPDATE skill_profiles SET
                       description=?, output_keys=?, use_cases=?,
                       success_count=?, fail_count=?, avg_credits=?,
                       reliability=?, last_seen=?
                       WHERE slug=?""",
                    (
                        desc,
                        json.dumps(merged_keys),
                        json.dumps(cases),
                        sc, fc, new_avg, reliability,
                        _utcnow(), slug,
                    ),
                )

            await db.commit()

    # ------------------------------------------------------------------
    # Record that a set of skills ran together in one goal
    # ------------------------------------------------------------------

    async def record_goal_completion(
        self,
        skill_sequence: List[str],  # ordered list of slugs that ran in this goal
        goal_description: str,
        success: bool,
    ) -> None:
        if len(skill_sequence) < 2:
            return

        async with aiosqlite.connect(self._db_path) as db:
            # Pairwise interactions
            seen_pairs = set()
            for i, a in enumerate(skill_sequence):
                for j, b in enumerate(skill_sequence):
                    if a == b:
                        continue
                    pair = (min(a, b), max(a, b))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    is_sequence = (j == i + 1)  # b ran directly after a
                    row = await (await db.execute(
                        "SELECT * FROM skill_interactions WHERE skill_a=? AND skill_b=?",
                        (a, b),
                    )).fetchone()

                    if row is None:
                        await db.execute(
                            """INSERT INTO skill_interactions
                               (skill_a, skill_b, sequence_count, co_count, success_count)
                               VALUES (?,?,?,?,?)""",
                            (a, b, 1 if is_sequence else 0, 1, 1 if success else 0),
                        )
                    else:
                        await db.execute(
                            """UPDATE skill_interactions SET
                               sequence_count = sequence_count + ?,
                               co_count = co_count + 1,
                               success_count = success_count + ?
                               WHERE skill_a=? AND skill_b=?""",
                            (1 if is_sequence else 0, 1 if success else 0, a, b),
                        )

            # Full sequence record (3+ skills)
            if len(skill_sequence) >= 3:
                seq_hash = hashlib.md5(json.dumps(skill_sequence).encode()).hexdigest()
                row = await (await db.execute(
                    "SELECT * FROM skill_sequences WHERE seq_hash=?", (seq_hash,)
                )).fetchone()

                if row is None:
                    await db.execute(
                        """INSERT INTO skill_sequences
                           (seq_hash, sequence, goal_type, count, success_count)
                           VALUES (?,?,?,?,?)""",
                        (
                            seq_hash,
                            json.dumps(skill_sequence),
                            goal_description[:80],
                            1,
                            1 if success else 0,
                        ),
                    )
                else:
                    await db.execute(
                        """UPDATE skill_sequences SET
                           count = count + 1,
                           success_count = success_count + ?
                           WHERE seq_hash=?""",
                        (1 if success else 0, seq_hash),
                    )

            await db.commit()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def get_profiles(self, slugs: List[str]) -> Dict[str, dict]:
        """Return profiles for a list of slugs (only those seen before)."""
        if not slugs:
            return {}
        placeholders = ",".join("?" * len(slugs))
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (await db.execute(
                f"SELECT * FROM skill_profiles WHERE slug IN ({placeholders})", slugs
            )).fetchall()

        return {
            row["slug"]: {
                "slug": row["slug"],
                "description": row["description"],
                "output_keys": json.loads(row["output_keys"] or "[]"),
                "use_cases": json.loads(row["use_cases"] or "[]"),
                "success_count": row["success_count"],
                "fail_count": row["fail_count"],
                "avg_credits": round(row["avg_credits"] or 0, 2),
                "reliability": round(row["reliability"] or 0.5, 2),
                "last_seen": row["last_seen"],
                "notes": row["notes"] or "",
            }
            for row in rows
        }

    async def get_interactions(self, slug: str, min_count: int = 2, limit: int = 5) -> List[dict]:
        """Return skills that co-occur with slug at least min_count times."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (await db.execute(
                """SELECT skill_a, skill_b, sequence_count, co_count, success_count
                   FROM skill_interactions
                   WHERE (skill_a=? OR skill_b=?) AND co_count >= ?
                   ORDER BY success_count DESC, co_count DESC
                   LIMIT ?""",
                (slug, slug, min_count, limit),
            )).fetchall()

        result = []
        for row in rows:
            partner = row["skill_b"] if row["skill_a"] == slug else row["skill_a"]
            result.append({
                "partner": partner,
                "co_count": row["co_count"],
                "sequence_count": row["sequence_count"],
                "success_count": row["success_count"],
            })
        return result

    async def get_all_profiles(self) -> List[dict]:
        """Return all known skill profiles ordered by reliability."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (await db.execute(
                "SELECT * FROM skill_profiles ORDER BY reliability DESC"
            )).fetchall()
        return [
            {
                "slug": r["slug"],
                "reliability": round(r["reliability"] or 0.5, 2),
                "success_count": r["success_count"],
                "fail_count": r["fail_count"],
                "avg_credits": round(r["avg_credits"] or 0, 2),
                "use_cases": json.loads(r["use_cases"] or "[]"),
                "output_keys": json.loads(r["output_keys"] or "[]"),
            }
            for r in rows
        ]

    async def update_notes(self, slug: str, notes: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE skill_profiles SET notes=? WHERE slug=?", (notes, slug)
            )
            await db.commit()
