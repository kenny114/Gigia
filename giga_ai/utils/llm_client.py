"""
llm_client.py – Pluggable LLM client.

Classes
-------
LLMClient          Abstract base class
OpenAILLMClient    Real implementation via openai library
MockLLMClient      Deterministic stub for testing

Usage
-----
    from giga_ai.utils.llm_client import get_llm_client

    client = get_llm_client()          # reads from Config
    response = await client.complete("Decompose this goal: …")
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from giga_ai.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract LLM client – all implementations must expose ``complete``."""

    @abstractmethod
    async def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send *prompt* to the LLM and return the response text.

        Parameters
        ----------
        prompt:
            The user-facing prompt text.
        system_prompt:
            Optional system / context message.

        Returns
        -------
        str
            Raw text response from the model.
        """


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAILLMClient(LLMClient):
    """
    LLM client backed by the OpenAI Chat Completions API.

    Parameters
    ----------
    api_key:
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    model:
        Model identifier, e.g. ``"gpt-4o"``.
    temperature:
        Sampling temperature (0 = deterministic).
    max_tokens:
        Maximum tokens in the completion.
    timeout:
        Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 60,
    ) -> None:
        # Import lazily so the module is importable even without openai installed
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAILLMClient. "
                "Install it with: pip install openai"
            ) from exc

        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        log.info("Sending prompt to OpenAI", extra={"model": self._model, "prompt_len": len(prompt)})

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            log.error("OpenAI API call failed", extra={"error": type(exc).__name__, "detail": str(exc)})
            raise

        text: str = response.choices[0].message.content or ""
        log.info("Received response from OpenAI", extra={"response_len": len(text)})
        return text


# ---------------------------------------------------------------------------
# Mock implementation
# ---------------------------------------------------------------------------

_MOCK_TASK_TEMPLATE = [
    {
        "task_id": "task-{n}-1",
        "title": "Research: {goal}",
        "description": "Gather background information about: {goal}",
        "sub_bot_type": "scraper",
        "priority": 1,
        "dependencies": [],
    },
    {
        "task_id": "task-{n}-2",
        "title": "Validate: {goal}",
        "description": "Cross-check findings for: {goal}",
        "sub_bot_type": "scraper",
        "priority": 2,
        "dependencies": ["task-{n}-1"],
    },
    {
        "task_id": "task-{n}-3",
        "title": "Synthesise results: {goal}",
        "description": "Compile and summarise results for: {goal}",
        "sub_bot_type": "scraper",
        "priority": 3,
        "dependencies": ["task-{n}-1", "task-{n}-2"],
    },
]


class MockLLMClient(LLMClient):
    """
    Deterministic stub that returns structured fake task decompositions.

    Useful for unit tests and local development without an API key.
    The ``complete`` method returns a JSON array of task objects whose
    structure mirrors what ``PlanningBrain`` expects.
    """

    def __init__(self, call_delay: float = 0.05) -> None:
        self._call_count = 0
        self._call_delay = call_delay  # simulated latency in seconds

    async def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        await asyncio.sleep(self._call_delay)
        self._call_count += 1
        n = self._call_count

        # Extract a rough goal summary from the prompt
        goal_match = re.search(r"Goal:\s*(.+?)(?:\n|$)", prompt, re.IGNORECASE)
        goal_snippet = goal_match.group(1).strip() if goal_match else "task"
        goal_snippet = goal_snippet[:60]

        tasks = []
        for tmpl in _MOCK_TASK_TEMPLATE:
            task = {k: v.format(n=n, goal=goal_snippet) if isinstance(v, str) else v
                    for k, v in tmpl.items()}
            # Fix list items that contain format strings
            task["dependencies"] = [d.format(n=n, goal=goal_snippet) for d in task["dependencies"]]
            tasks.append(task)

        return json.dumps(tasks, indent=2)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_client(config=None) -> LLMClient:
    """
    Instantiate and return the appropriate LLMClient based on config.

    Parameters
    ----------
    config:
        A ``giga_ai.config.Config`` instance.  If ``None``, the process
        singleton is loaded via ``get_config()``.
    """
    if config is None:
        from giga_ai.config import get_config
        config = get_config()

    if config.llm.provider == "mock":
        log.info("Using MockLLMClient (provider=mock)")
        return MockLLMClient()

    log.info("Using OpenAILLMClient", extra={"model": config.llm.model})
    return OpenAILLMClient(
        api_key=config.llm.api_key,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        timeout=config.llm.timeout_seconds,
    )
