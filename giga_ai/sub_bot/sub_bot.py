"""
sub_bot.py – Base sub-bot worker class.

All concrete sub-bots inherit from ``SubBot`` and implement
``execute(instruction) -> Result | ErrorReport``.

Error taxonomy
--------------
The module defines a hierarchy of exceptions that sub-bots raise internally.
These are caught by ``SubBot.execute`` (or by the caller) and translated into
structured ``ErrorReport`` objects returned to the ManagerBot.

Structured error types (ErrorType enum, defined in message_schemas):
  CaptchaDetected   – CAPTCHA/bot-detection page encountered
  Timeout           – request or browser action timed out
  HTTP404           – resource not found (non-retryable)
  HTTP403           – access forbidden (may be retryable with proxy rotation)
  HTTP5xx           – server-side error (retryable)
  ParseError        – could not extract expected data from response
  BrowserCrash      – browser/driver process died unexpectedly
  Unknown           – catch-all

Usage
-----
    class MySubBot(SubBot):
        async def execute(self, instruction: SubBotInstruction) -> Result | ErrorReport:
            ...

    bot = MySubBot(config=cfg)
    outcome = await bot.execute(instruction)
    if isinstance(outcome, Result):
        print(outcome.data)
    else:
        print(outcome.error_type, outcome.message)
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Union

from giga_ai.messaging.message_schemas import (
    ErrorReport,
    ErrorType,
    Result,
    SubBotInstruction,
)
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal exception hierarchy (translated to ErrorReport before returning)
# ---------------------------------------------------------------------------

class SubBotException(Exception):
    """Base for all sub-bot internal errors."""
    error_type: ErrorType = ErrorType.UNKNOWN
    retryable: bool = True


class CaptchaDetectedException(SubBotException):
    error_type = ErrorType.CAPTCHA_DETECTED
    retryable = True


class TimeoutException(SubBotException):
    error_type = ErrorType.TIMEOUT
    retryable = True


class HTTP404Exception(SubBotException):
    error_type = ErrorType.HTTP_404
    retryable = False


class HTTP403Exception(SubBotException):
    error_type = ErrorType.HTTP_403
    retryable = True


class HTTP5xxException(SubBotException):
    error_type = ErrorType.HTTP_5XX
    retryable = True


class ParseErrorException(SubBotException):
    error_type = ErrorType.PARSE_ERROR
    retryable = False


class BrowserCrashException(SubBotException):
    error_type = ErrorType.BROWSER_CRASH
    retryable = True


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class SubBot(ABC):
    """
    Abstract base class for all sub-bot workers.

    Concrete sub-bots must implement ``_run(instruction)``; the public
    ``execute`` method wraps it with standardised error handling and
    produces a ``Result`` or ``ErrorReport``.

    Parameters
    ----------
    config:
        Config override; loaded from singleton if not supplied.
    """

    def __init__(self, config=None) -> None:
        if config is None:
            from giga_ai.config import get_config
            config = get_config()
        self._config = config
        self._logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public entry point (do not override)
    # ------------------------------------------------------------------

    async def execute(
        self, instruction: SubBotInstruction
    ) -> Union[Result, ErrorReport]:
        """
        Execute *instruction* and return either a ``Result`` or an
        ``ErrorReport``.

        This method should **not** be overridden.  Implement ``_run``
        instead.
        """
        self._logger.debug(
            "SubBot: executing instruction",
            extra={
                "instruction_id": instruction.instruction_id,
                "sub_bot_type": instruction.sub_bot_type,
                "task_id": instruction.task_id,
            },
        )

        try:
            data = await self._run(instruction)
            result = Result(
                instruction_id=instruction.instruction_id,
                task_id=instruction.task_id,
                data=data,
                correlation_id=instruction.correlation_id,
            )
            self._logger.info(
                "SubBot: instruction succeeded",
                extra={
                    "instruction_id": instruction.instruction_id,
                    "data_keys": list(data.keys()),
                },
            )
            return result

        except SubBotException as exc:
            return self._make_error_report(instruction, exc.error_type, str(exc), exc.retryable)
        except Exception as exc:
            return self._make_error_report(instruction, ErrorType.UNKNOWN, str(exc), retryable=True)

    # ------------------------------------------------------------------
    # Abstract implementation hook
    # ------------------------------------------------------------------

    @abstractmethod
    async def _run(self, instruction: SubBotInstruction) -> dict:
        """
        Perform the actual work described by *instruction*.

        Parameters
        ----------
        instruction:
            The fully formed SubBotInstruction with parameters.

        Returns
        -------
        dict
            Arbitrary dict of scraped / processed data.

        Raises
        ------
        SubBotException subclass
            Any of the typed exceptions defined in this module.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_error_report(
        self,
        instruction: SubBotInstruction,
        error_type: ErrorType,
        message: str,
        retryable: bool = True,
        details: dict | None = None,
    ) -> ErrorReport:
        self._logger.warning(
            "SubBot: instruction failed",
            extra={
                "instruction_id": instruction.instruction_id,
                "error_type": error_type,
                "error_msg": message[:200],
            },
        )
        return ErrorReport(
            instruction_id=instruction.instruction_id,
            task_id=instruction.task_id,
            error_type=error_type,
            message=message,
            details=details or {},
            correlation_id=instruction.correlation_id,
            retryable=retryable,
        )

    @staticmethod
    def _http_status_to_exception(status: int) -> SubBotException | None:
        """Map an HTTP status code to the appropriate SubBotException."""
        if status == 404:
            return HTTP404Exception(f"HTTP 404 Not Found")
        if status == 403:
            return HTTP403Exception(f"HTTP 403 Forbidden")
        if 500 <= status < 600:
            return HTTP5xxException(f"HTTP {status} Server Error")
        return None
