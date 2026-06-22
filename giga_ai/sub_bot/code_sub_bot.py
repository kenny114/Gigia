"""
code_sub_bot.py – Sandboxed Python code execution sub-bot.

Runs arbitrary Python code in a subprocess with a configurable timeout.
The script receives injected variables via a JSON sidecar file and can
return structured output by printing a JSON-serialisable dict to stdout
on the last line (or anywhere prefixed with ``RESULT:``).

Instruction parameters (SubBotInstruction.parameters)
------------------------------------------------------
code            (str)   – Python source to execute (required)
input_data      (dict)  – Variables injected as ``input_data`` in the script
timeout         (int)   – Override exec_timeout_seconds from config
packages        (list)  – pip packages to install before running (optional)

Returns
-------
dict with keys:
  stdout       – Full captured stdout (trimmed to 50 KB)
  stderr       – Full captured stderr (trimmed to 10 KB)
  exit_code    – Process exit code
  result       – Parsed dict/value from the last ``RESULT: {...}`` line, if any

Raises (translated to ErrorReport)
-----------------------------------
  ExecutionError   – Non-zero exit code
  Timeout          – Exceeded timeout
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from typing import Any

from giga_ai.messaging.message_schemas import ErrorType, SubBotInstruction
from giga_ai.sub_bot.sub_bot import SubBot, TimeoutException
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)

_RESULT_RE = re.compile(r"RESULT:\s*(\{.*\}|\[.*\]|\".*\"|[\d.]+|true|false|null)", re.DOTALL)
_MAX_STDOUT = 50_000
_MAX_STDERR = 10_000


class CodeSubBot(SubBot):
    """Runs Python code in an isolated subprocess."""

    async def _run(self, instruction: SubBotInstruction) -> dict:
        p = instruction.parameters
        code: str = p.get("code", "")
        if not code.strip():
            raise ValueError("CodeSubBot: 'code' parameter is required and must not be empty")

        input_data: dict = p.get("input_data") or {}
        timeout: int = int(p.get("timeout") or self._config.workspace.exec_timeout_seconds)
        packages: list[str] = p.get("packages") or []
        python_bin: str = self._config.workspace.python_bin

        with tempfile.TemporaryDirectory(prefix="gigia_code_") as tmpdir:
            # Write the input sidecar
            input_path = os.path.join(tmpdir, "input.json")
            with open(input_path, "w") as f:
                json.dump(input_data, f)

            # Wrap user code: inject input_data, ensure cwd is tmpdir
            script = _build_script(code, input_path, tmpdir)
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w") as f:
                f.write(script)

            # Optionally install packages
            if packages:
                await self._install_packages(python_bin, packages, timeout)

            log.info(
                "CodeSubBot: executing script",
                extra={"lines": code.count("\n") + 1, "timeout": timeout},
            )

            stdout_bytes, stderr_bytes, exit_code = await _run_subprocess(
                [python_bin, script_path],
                timeout=timeout,
                cwd=tmpdir,
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")[:_MAX_STDOUT]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:_MAX_STDERR]

        result = _extract_result(stdout)

        if exit_code != 0:
            log.warning(
                "CodeSubBot: script exited with non-zero code",
                extra={"exit_code": exit_code, "stderr": stderr[:200]},
            )
            from giga_ai.sub_bot.sub_bot import SubBotException
            exc = SubBotException(
                f"Script exited with code {exit_code}. stderr: {stderr[:300]}"
            )
            exc.error_type = ErrorType.EXECUTION_ERROR
            exc.retryable = False
            raise exc

        log.info("CodeSubBot: script succeeded", extra={"exit_code": exit_code})
        return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code, "result": result}

    async def _install_packages(self, python_bin: str, packages: list[str], timeout: int) -> None:
        log.info("CodeSubBot: installing packages", extra={"packages": packages})
        _, stderr_bytes, exit_code = await _run_subprocess(
            [python_bin, "-m", "pip", "install", "--quiet", *packages],
            timeout=timeout,
        )
        if exit_code != 0:
            log.warning("CodeSubBot: pip install failed", extra={"stderr": stderr_bytes.decode()[:200]})


def _build_script(user_code: str, input_path: str, tmpdir: str) -> str:
    return f"""
import json, os, sys
os.chdir({tmpdir!r})
with open({input_path!r}) as _f:
    input_data = json.load(_f)

{user_code}
"""


async def _run_subprocess(
    cmd: list[str], timeout: int, cwd: str | None = None
) -> tuple[bytes, bytes, int]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return stdout, stderr, proc.returncode or 0
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        raise TimeoutException(f"Script exceeded {timeout}s timeout")


def _extract_result(stdout: str) -> Any:
    """Parse the last RESULT: <json> line from stdout, if present."""
    for match in reversed(list(_RESULT_RE.finditer(stdout))):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None
