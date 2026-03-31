"""
scraper_sub_bot.py – Lightweight HTTP scraping sub-bot.

Uses ``aiohttp`` for async HTTP and ``BeautifulSoup`` for HTML parsing.

Instruction parameters (SubBotInstruction.parameters)
------------------------------------------------------
url         (str)            – Target URL (required)
css_selectors (dict)         – {field_name: css_selector} mapping
                                If omitted, the full page text is returned.
proxy       (str)            – HTTP/SOCKS proxy URL, e.g. "http://host:port"
headers     (dict)           – Additional HTTP headers
user_agent  (str)            – Custom User-Agent (overrides config default)
encoding    (str)            – Response encoding override (default: auto)
follow_redirects (bool)      – Whether to follow HTTP redirects (default: True)
verify_ssl  (bool)           – Whether to verify SSL certs (default: True)

Returns
-------
dict with keys:
  url         – Final URL after redirects
  status_code – HTTP response status
  fields      – Dict of extracted field → value(s) based on css_selectors
  raw_text    – Full page text (only included when css_selectors is empty)

Raises (translated to ErrorReport by SubBot base)
---------
  HTTP404Exception        – 404 status
  HTTP403Exception        – 403 status
  HTTP5xxException        – 5xx status
  CaptchaDetectedException – if captcha keywords found in page
  TimeoutException        – aiohttp timeout
  ParseErrorException     – BeautifulSoup extraction failure
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup

from giga_ai.messaging.message_schemas import SubBotInstruction
from giga_ai.sub_bot.sub_bot import (
    CaptchaDetectedException,
    HTTP403Exception,
    HTTP404Exception,
    HTTP5xxException,
    ParseErrorException,
    SubBot,
    TimeoutException,
)
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)

_CAPTCHA_KEYWORDS = frozenset([
    "captcha", "recaptcha", "hcaptcha", "are you a robot",
    "bot detection", "cloudflare", "challenge", "verify you are human",
])


class ScraperSubBot(SubBot):
    """
    Async HTTP scraping sub-bot using aiohttp + BeautifulSoup.

    Parameters
    ----------
    config:
        Config override; loaded from singleton if not supplied.
    session:
        Optional pre-existing aiohttp.ClientSession.  A new session is
        created and closed per-execute call if not provided.
    """

    def __init__(self, config=None, session: Optional[aiohttp.ClientSession] = None) -> None:
        super().__init__(config=config)
        self._external_session = session

    # ------------------------------------------------------------------
    # SubBot._run implementation
    # ------------------------------------------------------------------

    async def _run(self, instruction: SubBotInstruction) -> Dict[str, Any]:
        params = instruction.parameters
        url: str = params.get("url", "")
        if not url:
            raise ParseErrorException("No 'url' provided in instruction parameters")

        css_selectors: Dict[str, str] = params.get("css_selectors", {})
        proxy: Optional[str] = params.get("proxy") or None
        headers: Dict[str, str] = dict(params.get("headers", {}))
        user_agent: str = params.get(
            "user_agent", self._config.sub_bot.default_user_agent
        )
        encoding: Optional[str] = params.get("encoding")
        follow_redirects: bool = params.get("follow_redirects", True)
        verify_ssl: bool = params.get("verify_ssl", True)
        timeout_sec: int = params.get(
            "timeout_seconds", self._config.sub_bot.request_timeout_seconds
        )

        headers.setdefault("User-Agent", user_agent)

        connector = aiohttp.TCPConnector(ssl=verify_ssl)
        timeout = aiohttp.ClientTimeout(total=timeout_sec)

        own_session = self._external_session is None
        session = self._external_session or aiohttp.ClientSession(
            connector=connector, timeout=timeout
        )

        try:
            return await self._fetch_and_parse(
                session=session,
                url=url,
                headers=headers,
                proxy=proxy,
                css_selectors=css_selectors,
                encoding=encoding,
                allow_redirects=follow_redirects,
            )
        except aiohttp.ServerTimeoutError:
            raise TimeoutException(f"Request to {url} timed out after {timeout_sec}s")
        except aiohttp.ClientConnectionError as exc:
            raise TimeoutException(f"Connection error to {url}: {exc}")
        finally:
            if own_session:
                await session.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _fetch_and_parse(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: Dict[str, str],
        proxy: Optional[str],
        css_selectors: Dict[str, str],
        encoding: Optional[str],
        allow_redirects: bool,
    ) -> Dict[str, Any]:
        self._logger.debug(
            "ScraperSubBot: fetching",
            extra={"url": url, "proxy": proxy or "(direct)"},
        )

        async with session.get(
            url,
            headers=headers,
            proxy=proxy,
            allow_redirects=allow_redirects,
        ) as resp:
            status = resp.status
            final_url = str(resp.url)

            # Error status codes
            exc = self._http_status_to_exception(status)
            if exc is not None:
                raise exc

            text = await resp.text(encoding=encoding, errors="replace")

        # Captcha detection
        lower_text = text.lower()
        for kw in _CAPTCHA_KEYWORDS:
            if kw in lower_text:
                raise CaptchaDetectedException(
                    f"Possible captcha/bot-detection at {final_url}: keyword '{kw}' found"
                )

        # Parse
        fields = self._extract_fields(text, css_selectors) if css_selectors else {}

        result: Dict[str, Any] = {
            "url": final_url,
            "status_code": status,
            "fields": fields,
        }
        if not css_selectors:
            result["raw_text"] = text[:50_000]  # cap at 50 KB

        return result

    def _extract_fields(
        self, html: str, css_selectors: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Extract fields from *html* using *css_selectors*.

        Returns a dict mapping field_name → list of extracted text strings.
        """
        soup = BeautifulSoup(html, "lxml")
        fields: Dict[str, Any] = {}

        for field_name, selector in css_selectors.items():
            try:
                elements = soup.select(selector)
                values: List[str] = [el.get_text(strip=True) for el in elements]
                fields[field_name] = values
            except Exception as exc:
                self._logger.warning(
                    "ScraperSubBot: CSS selector extraction failed",
                    extra={"field": field_name, "selector": selector, "error": str(exc)},
                )
                fields[field_name] = []

        if not any(fields.values()):
            raise ParseErrorException(
                f"No data extracted from page using provided CSS selectors: "
                f"{list(css_selectors.values())}"
            )

        return fields
