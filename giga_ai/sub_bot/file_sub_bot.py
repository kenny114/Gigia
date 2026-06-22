"""
file_sub_bot.py – Filesystem read/write sub-bot (workspace-scoped).

All paths are resolved relative to ``config.workspace.base_dir`` and
path traversal attempts (``../``) are blocked.

Instruction parameters (SubBotInstruction.parameters)
------------------------------------------------------
operation   (str)   – "read" | "write" | "append" | "list" | "delete" (required)
path        (str)   – Relative path within workspace (required for all except list)
content     (str)   – Content to write (required for "write" / "append")
encoding    (str)   – Text encoding, default "utf-8"
binary      (bool)  – If true, return base64-encoded bytes instead of text

Returns (varies by operation)
-------
read    → {"path": str, "content": str, "size": int}
write   → {"path": str, "written": int}
append  → {"path": str, "written": int}
list    → {"path": str, "entries": list[dict]}  (name, size, is_dir, modified)
delete  → {"path": str, "deleted": bool}

Raises (translated to ErrorReport)
-----------------------------------
  PermissionDenied  – Path escapes workspace or operation is denied
  ParseError        – File not found or unreadable
  ExecutionError    – OS-level error
"""

from __future__ import annotations

import base64
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from giga_ai.messaging.message_schemas import ErrorType, SubBotInstruction
from giga_ai.sub_bot.sub_bot import ParseErrorException, SubBot, SubBotException
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)

_MAX_READ = 5_000_000  # 5 MB read cap


class FileSubBot(SubBot):
    """Read/write files scoped to the configured workspace directory."""

    def _resolve(self, rel_path: str) -> Path:
        base = Path(self._config.workspace.base_dir).resolve()
        resolved = (base / rel_path).resolve()
        # Block path traversal
        try:
            resolved.relative_to(base)
        except ValueError:
            exc = SubBotException(
                f"Path '{rel_path}' escapes workspace '{base}' — access denied"
            )
            exc.error_type = ErrorType.PERMISSION_DENIED
            exc.retryable = False
            raise exc
        return resolved

    async def _run(self, instruction: SubBotInstruction) -> dict:
        p = instruction.parameters
        operation: str = p.get("operation", "").lower()
        rel_path: str = p.get("path", "")
        content: str = p.get("content", "")
        encoding: str = p.get("encoding", "utf-8")
        binary: bool = bool(p.get("binary", False))

        if operation not in ("read", "write", "append", "list", "delete"):
            raise ValueError(f"FileSubBot: unknown operation '{operation}'. Use read/write/append/list/delete")

        if operation == "list":
            return self._op_list(rel_path or ".")

        path = self._resolve(rel_path)

        if operation == "read":
            return self._op_read(path, rel_path, encoding, binary)
        if operation == "write":
            return self._op_write(path, rel_path, content, encoding, append=False)
        if operation == "append":
            return self._op_write(path, rel_path, content, encoding, append=True)
        if operation == "delete":
            return self._op_delete(path, rel_path)

    # ------------------------------------------------------------------

    def _op_read(self, path: Path, rel: str, encoding: str, binary: bool) -> dict:
        if not path.exists():
            raise ParseErrorException(f"File not found: {rel}")
        size = path.stat().st_size
        if size > _MAX_READ:
            raise ParseErrorException(f"File too large to read ({size} bytes, max {_MAX_READ})")
        log.info("FileSubBot: reading file", extra={"path": rel, "size": size})
        if binary:
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return {"path": rel, "content": data, "encoding": "base64", "size": size}
        text = path.read_text(encoding=encoding, errors="replace")
        return {"path": rel, "content": text, "size": size}

    def _op_write(self, path: Path, rel: str, content: str, encoding: str, append: bool) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        log.info("FileSubBot: writing file", extra={"path": rel, "append": append, "bytes": len(content)})
        path.open(mode, encoding=encoding).write(content)
        return {"path": rel, "written": len(content.encode(encoding))}

    def _op_list(self, rel: str) -> dict:
        path = self._resolve(rel)
        if not path.exists():
            raise ParseErrorException(f"Directory not found: {rel}")
        entries = []
        for entry in sorted(path.iterdir()):
            stat = entry.stat()
            entries.append({
                "name": entry.name,
                "is_dir": entry.is_dir(),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            })
        log.info("FileSubBot: listed directory", extra={"path": rel, "count": len(entries)})
        return {"path": rel, "entries": entries}

    def _op_delete(self, path: Path, rel: str) -> dict:
        if not path.exists():
            return {"path": rel, "deleted": False}
        if path.is_dir():
            import shutil
            shutil.rmtree(path)
        else:
            path.unlink()
        log.info("FileSubBot: deleted", extra={"path": rel})
        return {"path": rel, "deleted": True}
