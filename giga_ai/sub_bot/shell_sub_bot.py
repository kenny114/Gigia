"""
shell_sub_bot.py – Whitelisted shell command execution sub-bot.

Runs shell commands on the VPS via asyncio subprocess. Commands are
checked against a configurable whitelist (first token must match an
allowed prefix). Set ``workspace.shell_whitelist: ""`` to allow any
command (not recommended in production).

Instruction parameters (SubBotInstruction.parameters)
------------------------------------------------------
command         (str)   – Shell command string (required)
working_dir     (str)   – Relative working dir within workspace (default: workspace root)
timeout         (int)   – Override exec_timeout_seconds from config
env             (dict)  – Extra environment variables for the subprocess

Returns
-------
dict with keys:
  stdout       – Full captured stdout (trimmed to 50 KB)
  stderr       – Full captured stderr (trimmed to 10 KB)
  exit_code    – Process exit code
  command      – The command that was run

Raises (translated to ErrorReport)
-----------------------------------
  PermissionDenied   – Command not in whitelist
  ExecutionError     – Non-zero exit code
  Timeout            – Exceeded timeout
"""

from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path

from giga_ai.messaging.message_schemas import ErrorType, SubBotInstruction
from giga_ai.sub_bot.sub_bot import SubBot, SubBotException, TimeoutException
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)

_MAX_STDOUT = 50_000
_MAX_STDERR = 10_000


class ShellSubBot(SubBot):
    """Runs whitelisted shell commands in the workspace directory."""

    def _whitelist(self) -> list[str]:
        raw = self._config.workspace.shell_whitelist
        if not raw.strip():
            return []  # empty = allow all
        return [tok.strip().lower() for tok in raw.split(",") if tok.strip()]

    def _check_allowed(self, command: str) -> None:
        whitelist = self._whitelist()
        if not whitelist:
            return  # open mode
        tokens = shlex.split(command)
        if not tokens:
            return
        # Check the first real token (ignore leading env var assignments)
        first = None
        for tok in tokens:
            if "=" not in tok:
                first = os.path.basename(tok).lower()
                break
        if first is None:
            return
        if not any(first == allowed or first.startswith(allowed) for allowed in whitelist):
            exc = SubBotException(
                f"Command '{first}' is not in the shell whitelist. "
                f"Allowed: {', '.join(whitelist)}"
            )
            exc.error_type = ErrorType.PERMISSION_DENIED
            exc.retryable = False
            raise exc

    async def _run(self, instruction: SubBotInstruction) -> dict:
        p = instruction.parameters
        command: str = p.get("command", "").strip()
        if not command:
            raise ValueError("ShellSubBot: 'command' parameter is required")

        self._check_allowed(command)

        timeout: int = int(p.get("timeout") or self._config.workspace.exec_timeout_seconds)
        extra_env: dict = p.get("env") or {}
        rel_cwd: str = p.get("working_dir", ".")

        base = Path(self._config.workspace.base_dir).resolve()
        cwd = (base / rel_cwd).resolve()
        # Ensure working dir is inside workspace
        try:
            cwd.relative_to(base)
        except ValueError:
            exc = SubBotException(f"working_dir '{rel_cwd}' escapes workspace")
            exc.error_type = ErrorType.PERMISSION_DENIED
            exc.retryable = False
            raise exc

        cwd.mkdir(parents=True, exist_ok=True)

        env = {**os.environ, **extra_env}

        log.info(
            "ShellSubBot: running command",
            extra={"command": command[:120], "cwd": str(cwd), "timeout": timeout},
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            raise TimeoutException(f"Command exceeded {timeout}s timeout: {command[:80]}")

        stdout = stdout_bytes.decode("utf-8", errors="replace")[:_MAX_STDOUT]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:_MAX_STDERR]
        exit_code = proc.returncode or 0

        if exit_code != 0:
            log.warning(
                "ShellSubBot: command exited with non-zero code",
                extra={"exit_code": exit_code, "stderr": stderr[:200]},
            )
            exc = SubBotException(
                f"Command exited with code {exit_code}. stderr: {stderr[:300]}"
            )
            exc.error_type = ErrorType.EXECUTION_ERROR
            exc.retryable = False
            raise exc

        log.info("ShellSubBot: command succeeded", extra={"exit_code": exit_code})
        return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code, "command": command}
