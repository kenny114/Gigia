"""
manager_bot.py – Task orchestrator: spawns sub-bots, handles retries, escalates.

Responsibilities
----------------
- Execute a single ``Task`` end-to-end.
- Build ``SubBotInstruction`` objects from Task metadata.
- Spawn the appropriate sub-bot (scraper / selenium) for each instruction.
- Retry failed sub-bots with alternative strategies (proxy rotation, etc.).
- Escalate genuinely unsolvable problems to the LearningBrain via the
  EventBus ``ESCALATION`` event.
- Expose a ``get_status()`` snapshot for ExecutionBrain polling.

Usage
-----
    manager = ManagerBot(task=task, event_bus=bus, memory=mem, config=cfg)
    await manager.run()        # usually called by ExecutionBrain
"""

from __future__ import annotations

import asyncio
import itertools
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

from urllib.parse import urlparse

from giga_ai.messaging.event_bus import EventBus, EventType
from giga_ai.messaging.message_schemas import (
    ErrorReport,
    ErrorType,
    EscalationReport,
    ManagerStatus,
    ManagerStatusEnum,
    Result,
    SubBotInstruction,
    SubBotType,
    Task,
)
from giga_ai.utils.logger import get_logger


log = get_logger(__name__)


class ManagerBot:
    """
    Mid-layer task orchestrator.

    Parameters
    ----------
    task:
        The Task this manager is responsible for completing.
    event_bus:
        Shared EventBus.
    memory:
        GlobalMemory instance (optional; used for strategy lookup).
    memory_context:
        Pre-fetched context dict from GlobalMemory.
    config:
        Config override; loaded from singleton if not supplied.
    on_complete_callback:
        Optional async callable invoked when the manager finishes.
        Signature: ``async def cb(manager_id, problem, solution, success_rate)``.
    """

    def __init__(
        self,
        task: Task,
        event_bus: EventBus,
        memory=None,
        memory_context: Optional[Dict[str, Any]] = None,
        config=None,
        on_complete_callback: Optional[Callable[..., Coroutine]] = None,
    ) -> None:
        self.manager_id: str = str(uuid.uuid4())
        self.task = task
        self._bus = event_bus
        self._memory = memory
        self._memory_context: Dict[str, Any] = memory_context or {}
        self._on_complete_callback = on_complete_callback

        if config is None:
            from giga_ai.config import get_config
            config = get_config()
        self._config = config

        self._status: ManagerStatusEnum = ManagerStatusEnum.IDLE
        self._active_sub_bot_count: int = 0
        self._error_count: int = 0
        self._last_heartbeat: datetime = datetime.now(timezone.utc)
        self._results: List[Result] = []
        self._error_history: List[ErrorReport] = []

        # Proxy rotation (round-robin)
        self._proxy_cycle = itertools.cycle(
            [p for p in self._config.proxies.list if p] or [""]
        )

        self._logger = get_logger(__name__, correlation_id=task.correlation_id)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main execution loop for this manager.

        1. Build sub-bot instructions from the task metadata.
        2. Spawn sub-bots concurrently (up to the configured limit).
        3. Retry failures up to ``max_attempts`` times.
        4. Escalate if exhausted.
        """
        self._status = ManagerStatusEnum.RUNNING
        self._logger.info(
            "ManagerBot: starting",
            extra={"manager_id": self.manager_id, "task_id": self.task.task_id},
        )

        try:
            instructions = self._build_instructions()
            if not instructions:
                await self.escalate(
                    f"Task '{self.task.task_id}' produced no executable instructions "
                    f"(sub_bot_type={self.task.sub_bot_type} — missing required metadata: "
                    f"url for scraper/browser, code for code, operation for file, command for shell)"
                )
                self._status = ManagerStatusEnum.COMPLETED
                return
            results = await self.spawn_sub_bots(instructions)
            self._results.extend(results)
            self._status = ManagerStatusEnum.COMPLETED
            self._logger.info(
                "ManagerBot: task completed",
                extra={
                    "manager_id": self.manager_id,
                    "task_id": self.task.task_id,
                    "result_count": len(results),
                },
            )
            # Notify SkillBrain after every SKILL task so it can learn
            if self.task.sub_bot_type == SubBotType.SKILL and results:
                slug = self.task.skill_slug or self.task.metadata.get("skill_slug", "")
                result_keys = list(results[0].data.keys()) if results else []
                await self._bus.publish(
                    EventType.SKILL_EXECUTED,
                    payload={
                        "slug": slug,
                        "description": self.task.description,
                        "goal_id": self.task.goal_id,
                        "goal_description": self.task.metadata.get("goal_description", self.task.title),
                        "result_keys": result_keys,
                        "success": True,
                        "credits": results[0].data.get("credits_charged", 0),
                    },
                    correlation_id=self.task.correlation_id,
                )
            if self._on_complete_callback:
                await self._on_complete_callback(
                    manager_id=self.manager_id,
                    problem=self.task.metadata.get("original_problem", ""),
                    solution=str(results[0].data if results else {}),
                    success_rate=1.0 if results else 0.0,
                )
        except Exception as exc:
            self._status = ManagerStatusEnum.CRASHED
            self._logger.error(
                "ManagerBot: unhandled exception in run()",
                extra={"manager_id": self.manager_id, "error": str(exc)},
            )
            raise
        finally:
            self._last_heartbeat = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Sub-bot spawning
    # ------------------------------------------------------------------

    async def spawn_sub_bots(
        self,
        instructions: List[SubBotInstruction],
    ) -> List[Result]:
        """
        Execute sub-bot instructions concurrently, respecting the concurrency
        limit from config.

        Parameters
        ----------
        instructions:
            List of instructions to execute.

        Returns
        -------
        List[Result]
            Successful results collected across all instructions.
        """
        semaphore = asyncio.Semaphore(self._config.manager_bot.max_concurrent_sub_bots)
        results: List[Result] = []

        async def _run_one(instruction: SubBotInstruction) -> Optional[Result]:
            async with semaphore:
                return await self._execute_with_retry(instruction)

        tasks = [asyncio.create_task(_run_one(inst)) for inst in instructions]
        self._active_sub_bot_count = len(tasks)

        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                if result is not None:
                    results.append(result)
                    await self._bus.publish(
                        EventType.SUB_BOT_RESULT,
                        payload=result.model_dump(mode="json"),
                        correlation_id=self.task.correlation_id,
                    )
            except Exception as exc:
                self._error_count += 1
                self._logger.error(
                    "ManagerBot: unexpected error from sub-bot coroutine",
                    extra={"error": str(exc)},
                )
            finally:
                self._active_sub_bot_count = max(0, self._active_sub_bot_count - 1)

        self._last_heartbeat = datetime.now(timezone.utc)
        return results

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    async def handle_sub_bot_failure(
        self,
        error: "SubBotError",  # type: ignore[name-defined]
        instruction: SubBotInstruction,
    ) -> Optional[Result]:
        """
        Attempt recovery after a sub-bot failure.

        Strategy:
        - ``CaptchaDetected`` → rotate proxy, retry
        - ``HTTP404``         → mark as non-retryable
        - ``Timeout``         → retry immediately
        - Others              → retry up to max_attempts

        Parameters
        ----------
        error:
            The SubBotError describing what went wrong.
        instruction:
            The original instruction that failed.

        Returns
        -------
        Optional[Result]
            A successful Result if recovery succeeded, otherwise None.
        """
        self._error_history.append(error.error_report)
        self._error_count += 1

        err_type = error.error_report.error_type

        # Non-retryable errors
        if err_type == ErrorType.HTTP_404 or not error.error_report.retryable:
            self._logger.warning(
                "ManagerBot: non-retryable error – skipping instruction",
                extra={
                    "instruction_id": instruction.instruction_id,
                    "error_type": err_type,
                },
            )
            return None

        # Rotate proxy for captcha / 403 blocks
        if err_type in (ErrorType.CAPTCHA_DETECTED, ErrorType.HTTP_403):
            new_proxy = next(self._proxy_cycle)
            instruction = instruction.model_copy(
                update={"parameters": {**instruction.parameters, "proxy": new_proxy}}
            )
            self._logger.info(
                "ManagerBot: rotated proxy after block",
                extra={"error_type": err_type, "new_proxy": new_proxy or "(direct)"},
            )

        # Retry — discard if the retry also produced an ErrorReport
        recovered = await self._run_sub_bot(instruction)
        if isinstance(recovered, Result):
            return recovered
        return None

    async def escalate(self, problem: str) -> None:
        """
        Send an EscalationReport to the LearningBrain via the event bus.

        Parameters
        ----------
        problem:
            Natural-language description of the unsolvable problem.
        """
        report = EscalationReport(
            manager_id=self.manager_id,
            task_id=self.task.task_id,
            problem=problem,
            context={
                "task_title": self.task.title,
                "task_description": self.task.description,
                **self._memory_context,
            },
            error_history=list(self._error_history),
            correlation_id=self.task.correlation_id,
        )

        self._logger.warning(
            "ManagerBot: escalating problem",
            extra={"manager_id": self.manager_id, "problem": problem[:120]},
        )

        await self._bus.publish(
            EventType.ESCALATION,
            payload=report.model_dump(mode="json"),
            correlation_id=self.task.correlation_id,
        )

        # Tell SkillBrain this skill failed so reliability scores stay accurate
        if self.task.sub_bot_type == SubBotType.SKILL:
            slug = self.task.skill_slug or self.task.metadata.get("skill_slug", "")
            if slug:
                await self._bus.publish(
                    EventType.SKILL_EXECUTED,
                    payload={
                        "slug": slug,
                        "description": self.task.description,
                        "goal_id": self.task.goal_id,
                        "goal_description": self.task.metadata.get("goal_description", self.task.title),
                        "result_keys": [],
                        "success": False,
                        "credits": 0,
                    },
                    correlation_id=self.task.correlation_id,
                )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> ManagerStatus:
        """Return a snapshot of this manager's current state."""
        return ManagerStatus(
            manager_id=self.manager_id,
            task_id=self.task.task_id,
            status=self._status,
            active_sub_bots=self._active_sub_bot_count,
            last_heartbeat=self._last_heartbeat,
            error_count=self._error_count,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # Domains/URL patterns that require a full browser (JS rendering).
    # Any URL whose host+path contains one of these keys gets upgraded to BROWSER.
    _BROWSER_URL_PATTERNS = (
        "maps.google.com",
        "google.com/maps",
        "google.com/search",
        "yelp.com/search",
        "yelp.com/biz",
        "tripadvisor.com",
        "facebook.com",
        "instagram.com",
        "linkedin.com",
        "zillow.com",
        "redfin.com",
        "realtor.com",
        "doordash.com",
        "ubereats.com",
    )

    @staticmethod
    def _needs_browser(url: str) -> bool:
        """Return True if *url* requires a full browser (JS rendering)."""
        try:
            parsed = urlparse(url)
            hostpath = (parsed.netloc + parsed.path).lower()
            return any(pat in hostpath for pat in ManagerBot._BROWSER_URL_PATTERNS)
        except Exception:
            return False

    def _build_instructions(self) -> List[SubBotInstruction]:
        """
        Build SubBotInstruction objects from the task's metadata.

        The Task's ``metadata`` dict may contain:
          - ``instructions``:  List[dict] of raw instruction payloads
          - ``url``:           Single URL shorthand (produces one instruction)
          - Any browser parameters (wait_for, css_selectors, scroll_feed, etc.)
            that are promoted directly into the instruction parameters.

        URL-based auto-upgrade:
          If the URL matches a known JS-heavy domain (Google Maps, Yelp, etc.)
          the sub_bot_type is automatically upgraded to BROWSER regardless of
          what the LLM planned, and domain-specific defaults are merged in.

        Returns an empty list when no URL is available for a scraper/browser
        task (run() will escalate cleanly).
        """
        # CODE tasks: pass code + input_data from metadata directly.
        if self.task.sub_bot_type == SubBotType.CODE:
            code = self.task.metadata.get("code", "")
            if not code.strip():
                self._logger.warning("ManagerBot: CODE task has no 'code' in metadata", extra={"task_id": self.task.task_id})
                return []
            return [SubBotInstruction(
                task_id=self.task.task_id,
                sub_bot_type=SubBotType.CODE,
                parameters={
                    "code": code,
                    "input_data": self.task.metadata.get("input_data") or {},
                    "timeout": self.task.metadata.get("timeout"),
                    "packages": self.task.metadata.get("packages") or [],
                },
                correlation_id=self.task.correlation_id,
                timeout_seconds=self._config.manager_bot.sub_bot_timeout_seconds,
            )]

        # FILE tasks: pass operation + path + content from metadata.
        if self.task.sub_bot_type == SubBotType.FILE:
            operation = self.task.metadata.get("operation", "")
            if not operation:
                self._logger.warning("ManagerBot: FILE task has no 'operation' in metadata", extra={"task_id": self.task.task_id})
                return []
            return [SubBotInstruction(
                task_id=self.task.task_id,
                sub_bot_type=SubBotType.FILE,
                parameters={k: v for k, v in self.task.metadata.items()},
                correlation_id=self.task.correlation_id,
                timeout_seconds=self._config.manager_bot.sub_bot_timeout_seconds,
            )]

        # SHELL tasks: pass command + working_dir + env from metadata.
        if self.task.sub_bot_type == SubBotType.SHELL:
            command = self.task.metadata.get("command", "")
            if not command.strip():
                self._logger.warning("ManagerBot: SHELL task has no 'command' in metadata", extra={"task_id": self.task.task_id})
                return []
            return [SubBotInstruction(
                task_id=self.task.task_id,
                sub_bot_type=SubBotType.SHELL,
                parameters={k: v for k, v in self.task.metadata.items()},
                correlation_id=self.task.correlation_id,
                timeout_seconds=self._config.manager_bot.sub_bot_timeout_seconds,
            )]

        # Skill tasks don't need a URL — they call back to the gateway.
        if self.task.sub_bot_type == SubBotType.SKILL:
            slug = self.task.skill_slug or self.task.metadata.get("skill_slug")
            if not slug:
                self._logger.warning(
                    "ManagerBot: SKILL task has no skill_slug — escalating",
                    extra={"task_id": self.task.task_id},
                )
                return []
            return [SubBotInstruction(
                task_id=self.task.task_id,
                sub_bot_type=SubBotType.SKILL,
                parameters={
                    "execute_url": self.task.metadata["execute_url"],
                    "token": self.task.metadata["token"],
                    "run_id": self.task.metadata.get("run_id", self.task.goal_id),
                    "skill_slug": slug,
                    "args": self.task.metadata.get("args") or {},
                },
                correlation_id=self.task.correlation_id,
                timeout_seconds=self._config.manager_bot.sub_bot_timeout_seconds,
            )]

        raw_instructions = self.task.metadata.get("instructions", [])

        if not raw_instructions:
            url = self.task.metadata.get("url")
            if url:
                # Promote all metadata fields into the instruction params
                # so wait_for, css_selectors, scroll_feed etc. flow through.
                raw_instructions = [{
                    k: v for k, v in self.task.metadata.items()
                    if k != "instructions"
                }]
            else:
                # No URL — cannot execute a scraper/browser task
                return []

        result = []
        for raw in raw_instructions:
            url = raw.get("url", "")
            proxy = next(self._proxy_cycle)

            # Determine the actual sub_bot_type to use
            sub_bot_type = self.task.sub_bot_type

            if url and self._needs_browser(url):
                # Force BROWSER for JS-heavy URLs regardless of LLM choice
                sub_bot_type = SubBotType.BROWSER
                self._logger.info(
                    "ManagerBot: auto-upgraded sub_bot_type to BROWSER",
                    extra={"url": url, "original_type": self.task.sub_bot_type},
                )

            params = {**raw, "proxy": proxy}
            inst = SubBotInstruction(
                task_id=self.task.task_id,
                sub_bot_type=sub_bot_type,
                parameters=params,
                correlation_id=self.task.correlation_id,
                timeout_seconds=self._config.manager_bot.sub_bot_timeout_seconds,
            )
            result.append(inst)

        return result

    async def _execute_with_retry(self, instruction: SubBotInstruction) -> Optional[Result]:
        """
        Try executing *instruction* up to ``max_attempts`` times.

        Returns the first successful Result, or None if all attempts fail
        (after which ``escalate`` is called).
        """
        max_attempts = self._config.retry.max_attempts
        delay = self._config.retry.base_delay_seconds

        for attempt in range(1, max_attempts + 1):
            result_or_error = await self._run_sub_bot(instruction)

            if isinstance(result_or_error, Result):
                return result_or_error

            # It's an ErrorReport
            from giga_ai.messaging.message_schemas import SubBotError as _SubBotError
            error = _SubBotError(
                error_report=result_or_error,
                attempt_number=attempt,
                instruction=instruction,
            )

            if attempt < max_attempts:
                recovered = await self.handle_sub_bot_failure(error, instruction)
                if recovered is not None:
                    return recovered
                wait = min(delay * (self._config.retry.backoff_factor ** (attempt - 1)),
                           self._config.retry.max_delay_seconds)
                self._logger.info(
                    "ManagerBot: waiting before retry",
                    extra={"attempt": attempt, "wait": wait},
                )
                await asyncio.sleep(wait)
            else:
                self._logger.warning(
                    "ManagerBot: all retry attempts exhausted",
                    extra={"instruction_id": instruction.instruction_id, "attempts": attempt},
                )
                await self.escalate(
                    f"Sub-bot failed after {attempt} attempts: "
                    f"{result_or_error.error_type} – {result_or_error.message}"
                )
                return None

        return None

    async def _run_sub_bot(
        self, instruction: SubBotInstruction
    ) -> "Result | ErrorReport":  # type: ignore[name-defined]
        """Instantiate the correct sub-bot type and execute the instruction."""
        from giga_ai.sub_bot.bot_factory import BotFactory as SubBotFactory  # type: ignore
        sub_bot = SubBotFactory.create(instruction.sub_bot_type, self._config)
        self._active_sub_bot_count += 1
        try:
            result = await asyncio.wait_for(
                sub_bot.execute(instruction),
                timeout=instruction.timeout_seconds,
            )
        except asyncio.TimeoutError:
            result = ErrorReport(
                instruction_id=instruction.instruction_id,
                task_id=instruction.task_id,
                error_type=ErrorType.TIMEOUT,
                message=f"Sub-bot timed out after {instruction.timeout_seconds}s",
                correlation_id=instruction.correlation_id,
            )
        finally:
            self._active_sub_bot_count = max(0, self._active_sub_bot_count - 1)
        self._last_heartbeat = datetime.now(timezone.utc)
        return result
