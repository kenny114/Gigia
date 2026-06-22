"""
skill_sub_bot.py – Calls the almcp gateway /api/brain/execute endpoint.

This replaces the browser/scraper worker for skill-mode tasks. Instead of
driving a browser, it POSTs to the gateway's metered runner, which executes
the named skill, charges credits, and returns the result. Gigia's
retry/escalation machinery works identically — a gateway error (rate_limited,
insufficient_credits, timeout) surfaces as an ErrorReport and is handled by
ManagerBot.handle_sub_bot_failure just like a scraper failure.

The instruction's parameters must contain:
  execute_url  – https://…/api/brain/execute
  token        – Bearer API key for this owner
  run_id       – Gigia run id (correlates with gateway brain_requests)
  skill_slug   – almcp catalog slug to execute
  args         – dict of skill arguments (optional, defaults to {})
"""

from __future__ import annotations

import aiohttp

from giga_ai.messaging.message_schemas import ErrorType, SubBotInstruction
from giga_ai.sub_bot.sub_bot import (
    HTTP403Exception,
    HTTP404Exception,
    HTTP5xxException,
    SubBot,
    TimeoutException,
)
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)

# Gateway error codes that are not worth retrying
_NON_RETRYABLE_CODES = {"unknown_tool", "invalid_input"}


class SkillSubBot(SubBot):
    """
    Executes an almcp skill via the gateway's /api/brain/execute endpoint.

    One SkillSubBot instance per instruction; stateless beyond the config.
    """

    async def _run(self, instruction: SubBotInstruction) -> dict:
        p = instruction.parameters
        execute_url: str = p["execute_url"]
        token: str = p["token"]
        run_id: str = p["run_id"]
        skill_slug: str = p["skill_slug"]
        args: dict = p.get("args") or {}

        payload = {
            "run_id": run_id,
            "task_id": instruction.task_id,
            "skill_slug": skill_slug,
            "args": args,
        }

        timeout = aiohttp.ClientTimeout(total=instruction.timeout_seconds)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        self._logger.info(
            "SkillSubBot: calling gateway",
            extra={
                "execute_url": execute_url,
                "skill_slug": skill_slug,
                "task_id": instruction.task_id,
            },
        )

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(execute_url, json=payload, headers=headers) as resp:
                    body = await resp.json(content_type=None)

                    if resp.status == 401:
                        raise HTTP403Exception("Gateway rejected the API key (401 Unauthorized)")
                    if resp.status == 404:
                        raise HTTP404Exception(f"Gateway endpoint not found: {execute_url}")
                    if resp.status == 429:
                        raise HTTP5xxException("Gateway rate limited this request (429)")
                    if resp.status >= 500:
                        raise HTTP5xxException(f"Gateway returned {resp.status}")

                    if not body.get("ok"):
                        code = body.get("code", "unknown")
                        msg = body.get("error", "skill execution failed")
                        retryable = code not in _NON_RETRYABLE_CODES
                        if not retryable:
                            raise HTTP404Exception(f"{code}: {msg}")
                        raise HTTP5xxException(f"{code}: {msg}")

                    result = body.get("result") or {}
                    credits_charged = body.get("credits_charged", 0)

                    self._logger.info(
                        "SkillSubBot: skill executed",
                        extra={
                            "skill_slug": skill_slug,
                            "credits_charged": credits_charged,
                            "task_id": instruction.task_id,
                        },
                    )

                    if isinstance(result, dict):
                        return {**result, "_credits_charged": credits_charged, "_skill_slug": skill_slug}
                    return {"result": result, "_credits_charged": credits_charged, "_skill_slug": skill_slug}

        except aiohttp.ServerTimeoutError:
            raise TimeoutException(f"Gateway timed out executing skill '{skill_slug}'")
        except aiohttp.ClientConnectionError as exc:
            raise HTTP5xxException(f"Could not connect to gateway: {exc}")
