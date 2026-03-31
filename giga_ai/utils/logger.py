"""
logger.py – Structured JSON logger with optional correlation IDs.

Usage
-----
    from giga_ai.utils.logger import get_logger

    log = get_logger("my_module", correlation_id="abc-123")
    log.info("Something happened", extra={"key": "value"})

Every record is emitted as a single JSON line:
    {"timestamp": "...", "level": "INFO", "name": "my_module",
     "correlation_id": "abc-123", "message": "Something happened", "key": "value"}
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from pythonjsonlogger import jsonlogger  # python-json-logger


# ---------------------------------------------------------------------------
# Custom JSON formatter
# ---------------------------------------------------------------------------

class _GigaJsonFormatter(jsonlogger.JsonFormatter):
    """
    Extends the base JSON formatter to:
      - always include ``timestamp``, ``level``, ``name``, ``correlation_id``
      - flatten ``extra`` keys into the top-level JSON object
    """

    def add_fields(
        self,
        log_record: dict,
        record: logging.LogRecord,
        message_dict: dict,
    ) -> None:
        super().add_fields(log_record, record, message_dict)

        # Rename / ensure standard keys
        log_record["timestamp"] = log_record.pop("asctime", None) or self.formatTime(record)
        log_record["level"] = record.levelname
        log_record["name"] = record.name
        log_record.setdefault("correlation_id", getattr(record, "correlation_id", None))
        log_record["message"] = record.getMessage()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_loggers: dict[str, logging.Logger] = {}


class _CorrelationAdapter(logging.LoggerAdapter):
    """Injects ``correlation_id`` into every log record's ``extra``."""

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("correlation_id", self.extra.get("correlation_id"))
        return msg, kwargs


def get_logger(
    name: str,
    correlation_id: Optional[str] = None,
    level: Optional[str] = None,
) -> logging.LoggerAdapter:
    """
    Return a structured JSON logger.

    Parameters
    ----------
    name:
        Logger name (typically ``__name__`` of the calling module).
    correlation_id:
        Optional trace / correlation identifier appended to every record.
    level:
        Override the log level for this logger only.  Defaults to the
        root-logger level (INFO unless reconfigured).
    """
    # Lazily configure the root handler once
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = _GigaJsonFormatter(
            fmt="%(timestamp)s %(level)s %(name)s %(correlation_id)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.setLevel(logging.INFO)

    logger = logging.getLogger(name)
    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    return _CorrelationAdapter(logger, {"correlation_id": correlation_id})
