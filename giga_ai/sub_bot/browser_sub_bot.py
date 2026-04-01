"""
browser_sub_bot.py – Full-featured browser sub-bot using Playwright.

Designed for JavaScript-heavy pages that plain HTTP cannot handle:
  - Google Maps (business/location searches)
  - Google Search results
  - Any SPA or dynamically loaded content

How it works
------------
1. Launches a stealth Playwright Chromium browser (mimics a real user).
2. Navigates to the URL and waits for a configurable CSS selector to appear.
3. Optionally scrolls the result feed to load more items.
4. Runs a sequence of browser actions (click, type, wait, scroll).
5. Extracts structured fields via CSS selectors OR returns raw page text.
6. Returns a structured dict that ManagerBot collects as a Result.

Instruction parameters (SubBotInstruction.parameters)
------------------------------------------------------
url             (str)            – Target URL (required)
wait_for        (str)            – CSS selector to wait for before extracting
                                   Default: "body"
wait_timeout    (float)          – Seconds to wait for selector (default: 15)
scroll_feed     (bool)           – Scroll the page to load more results (default: False)
scroll_count    (int)            – How many times to scroll (default: 3)
scroll_pause    (float)          – Seconds between scrolls (default: 1.5)
scroll_selector (str)            – Selector of the element to scroll inside
                                   (e.g. Google Maps feed). If omitted, scrolls window.
actions         (list[dict])     – Pre-extraction browser actions (same schema as
                                   SeleniumSubBot actions)
css_selectors   (dict)           – {field_name: css_selector} extraction map.
                                   If omitted, raw_text is returned instead.
proxy           (str)            – HTTP proxy URL
user_agent      (str)            – Custom User-Agent (overrides config default)
screenshot_on_error (bool)       – Save screenshot on failure (default: config value)
screenshot_path (str)            – Override screenshot directory

Returns
-------
dict with keys:
  url           – Final URL after navigation
  fields        – Dict of field_name → list[str] (when css_selectors provided)
  raw_text      – Visible page text (when css_selectors omitted, capped 100 KB)
  screenshot    – Base64 PNG string (only present on error screenshots)

Raises (translated to ErrorReport by SubBot base)
-------------------------------------------------
  ParseErrorException      – No url provided, or selector extraction found nothing
  TimeoutException         – Page/selector did not load within wait_timeout
  CaptchaDetectedException – Bot-detection page detected
  BrowserCrashException    – Playwright crashed or unexpected error
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from giga_ai.messaging.message_schemas import SubBotInstruction
from giga_ai.sub_bot.sub_bot import (
    BrowserCrashException,
    CaptchaDetectedException,
    ParseErrorException,
    SubBot,
    TimeoutException,
)
from giga_ai.utils.logger import get_logger

log = get_logger(__name__)

_CAPTCHA_KEYWORDS = frozenset([
    "captcha", "recaptcha", "hcaptcha", "are you a robot",
    "bot detection", "cloudflare", "challenge", "verify you are human",
    "unusual traffic", "automated queries",
])

# Stealth user-agent that blends in well
_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Per-domain defaults: what to wait for and how to scroll
_DOMAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "maps.google.com": {
        "wait_for": "div[role='feed'], div[role='main']",
        "scroll_feed": True,
        "scroll_count": 5,
        "scroll_pause": 1.8,
        "scroll_selector": "div[role='feed']",
        "css_selectors": {
            "names":    "div[role='feed'] a[aria-label]",
            "links":    "div[role='feed'] a[href*='maps']",
            "ratings":  "div[role='feed'] span[aria-label*='star']",
            "addresses":"div[role='feed'] .fontBodyMedium span",
        },
    },
    "google.com/maps": {
        "wait_for": "div[role='feed'], div[role='main']",
        "scroll_feed": True,
        "scroll_count": 5,
        "scroll_pause": 1.8,
        "scroll_selector": "div[role='feed']",
        "css_selectors": {
            "names":    "div[role='feed'] a[aria-label]",
            "links":    "div[role='feed'] a[href*='maps']",
            "ratings":  "div[role='feed'] span[aria-label*='star']",
            "addresses":"div[role='feed'] .fontBodyMedium span",
        },
    },
    "google.com/search": {
        "wait_for": "#search, #rso",
        "scroll_feed": False,
        "css_selectors": {
            "titles":   "#rso .g h3",
            "links":    "#rso .g a[href]",
            "snippets": "#rso .g .VwiC3b",
        },
    },
    "yelp.com": {
        "wait_for": "[data-testid='serp-ia-card'], .businessName__09f24__EYSZE",
        "scroll_feed": True,
        "scroll_count": 3,
        "scroll_pause": 1.2,
        "css_selectors": {
            "names":   "h3.css-1agk4wl a",
            "ratings": "div[aria-label*='star rating']",
            "reviews": "span.css-chan6m",
        },
    },
}


def _get_domain_profile(url: str) -> Dict[str, Any]:
    """Return domain-specific defaults for the given URL, or empty dict."""
    try:
        parsed = urlparse(url)
        # Build a host+path prefix to match against profiles
        hostpath = (parsed.netloc + parsed.path).lower()
        for key, profile in _DOMAIN_PROFILES.items():
            if key in hostpath:
                return profile
    except Exception:
        pass
    return {}


class BrowserSubBot(SubBot):
    """
    Full-featured Playwright browser sub-bot.

    Handles JS-heavy pages, Google Maps, Google Search, and any site
    requiring real browser rendering.
    """

    async def _run(self, instruction: SubBotInstruction) -> Dict[str, Any]:
        params = instruction.parameters
        url: str = params.get("url", "")
        if not url:
            raise ParseErrorException("No 'url' provided in instruction parameters")

        # Merge domain profile defaults under instruction params
        # (instruction params take precedence over domain defaults)
        profile = _get_domain_profile(url)
        effective: Dict[str, Any] = {**profile, **params}

        wait_for: str = effective.get("wait_for", "body")
        wait_timeout: float = float(effective.get("wait_timeout", 15))
        scroll_feed: bool = bool(effective.get("scroll_feed", False))
        scroll_count: int = int(effective.get("scroll_count", 3))
        scroll_pause: float = float(effective.get("scroll_pause", 1.5))
        scroll_selector: Optional[str] = effective.get("scroll_selector")
        actions: List[dict] = effective.get("actions", [])
        css_selectors: Dict[str, str] = effective.get("css_selectors", {})
        proxy_url: Optional[str] = effective.get("proxy") or None
        user_agent: str = effective.get("user_agent", _DEFAULT_UA)
        screenshot_on_error: bool = effective.get(
            "screenshot_on_error", self._config.sub_bot.screenshot_on_error
        )
        screenshot_path_override: Optional[str] = effective.get("screenshot_path")

        try:
            from playwright.async_api import async_playwright, TimeoutError as PWTimeout
        except ImportError as exc:
            raise BrowserCrashException(
                "Playwright is not installed. Run: pip install playwright && playwright install chromium"
            ) from exc

        proxy_config = {"server": proxy_url} if proxy_url else None
        timeout_ms = int(wait_timeout * 1000)

        self._logger.info(
            "BrowserSubBot: launching browser",
            extra={"url": url, "wait_for": wait_for, "scroll_feed": scroll_feed},
        )

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                proxy=proxy_config,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                    "--window-size=1920,1080",
                ],
            )

            context = await browser.new_context(
                user_agent=user_agent,
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                timezone_id="America/New_York",
                # Mask automation signals
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )

            # Inject stealth JS to remove navigator.webdriver fingerprint
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                window.chrome = { runtime: {} };
            """)

            page = await context.new_page()
            screenshot_b64: Optional[str] = None

            try:
                await page.goto(url, timeout=timeout_ms * 2, wait_until="domcontentloaded")

                # Wait for the key selector that signals content is ready
                try:
                    # wait_for may be a comma-separated list of selectors — try each
                    selectors = [s.strip() for s in wait_for.split(",")]
                    found = False
                    for sel in selectors:
                        try:
                            await page.wait_for_selector(sel, timeout=timeout_ms)
                            found = True
                            break
                        except Exception:
                            continue
                    if not found:
                        raise TimeoutException(
                            f"None of the wait selectors appeared within {wait_timeout}s: {wait_for}"
                        )
                except TimeoutException:
                    raise
                except Exception as exc:
                    raise TimeoutException(
                        f"Timed out waiting for '{wait_for}' on {url}"
                    ) from exc

                # Run pre-extraction actions
                if actions:
                    await self._run_actions(page, actions, timeout_ms)

                # Scroll to load more results if needed
                if scroll_feed:
                    await self._scroll_to_load(
                        page, scroll_selector, scroll_count, scroll_pause
                    )

                final_url = page.url
                page_content = await page.content()

                # Captcha / bot-detection check
                lower = page_content.lower()
                for kw in _CAPTCHA_KEYWORDS:
                    if kw in lower:
                        raise CaptchaDetectedException(
                            f"Bot-detection triggered at {final_url}: keyword '{kw}' found"
                        )

                # Extract fields or fall back to visible text
                if css_selectors:
                    fields = await self._extract_fields(page, css_selectors)
                    return {"url": final_url, "fields": fields}
                else:
                    raw_text = await page.evaluate(
                        "() => document.body.innerText"
                    )
                    return {"url": final_url, "raw_text": raw_text[:100_000]}

            except (CaptchaDetectedException, ParseErrorException, TimeoutException):
                raise
            except Exception as exc:
                if screenshot_on_error:
                    screenshot_b64 = await self._take_screenshot(
                        page, screenshot_path_override
                    )
                raise BrowserCrashException(f"Browser error on {url}: {exc}") from exc
            finally:
                await browser.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _run_actions(self, page, actions: List[dict], timeout_ms: int) -> None:
        """Execute a list of browser action dicts."""
        for action in actions:
            atype = action.get("type", "")
            try:
                if atype == "click":
                    await page.click(action["selector"], timeout=timeout_ms)
                elif atype == "type":
                    await page.fill(action["selector"], action.get("text", ""))
                elif atype == "wait":
                    await asyncio.sleep(float(action.get("seconds", 1)))
                elif atype == "scroll":
                    pixels = int(action.get("pixels", 300))
                    await page.evaluate(f"window.scrollBy(0, {pixels})")
                elif atype == "wait_for":
                    await page.wait_for_selector(
                        action["selector"],
                        timeout=action.get("timeout", 10) * 1000,
                    )
            except Exception as exc:
                self._logger.warning(
                    "BrowserSubBot: action failed (continuing)",
                    extra={"action_type": atype, "error": str(exc)},
                )

    async def _scroll_to_load(
        self,
        page,
        scroll_selector: Optional[str],
        scroll_count: int,
        scroll_pause: float,
    ) -> None:
        """
        Scroll a feed element (or the whole window) to trigger lazy loading.
        Used for Google Maps result lists, Yelp, etc.
        """
        self._logger.debug(
            "BrowserSubBot: scrolling to load more results",
            extra={"scroll_count": scroll_count, "selector": scroll_selector or "window"},
        )
        for i in range(scroll_count):
            try:
                if scroll_selector:
                    # Scroll inside a specific scrollable element
                    await page.evaluate(
                        """(selector) => {
                            const el = document.querySelector(selector);
                            if (el) el.scrollTop += 800;
                        }""",
                        scroll_selector,
                    )
                else:
                    await page.evaluate("window.scrollBy(0, 800)")
            except Exception as exc:
                self._logger.warning(
                    "BrowserSubBot: scroll step failed",
                    extra={"step": i, "error": str(exc)},
                )
            await asyncio.sleep(scroll_pause)

    async def _extract_fields(
        self, page, css_selectors: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Extract multiple fields from the page using CSS selectors.
        Returns a dict of field_name → list[str].
        Raises ParseErrorException if all selectors return empty results.
        """
        fields: Dict[str, Any] = {}
        for field_name, selector in css_selectors.items():
            try:
                elements = await page.query_selector_all(selector)
                values: List[str] = []
                for el in elements:
                    # Try aria-label first (better for Maps), then innerText
                    aria = await el.get_attribute("aria-label")
                    if aria and aria.strip():
                        values.append(aria.strip())
                    else:
                        text = await el.inner_text()
                        if text and text.strip():
                            values.append(text.strip())
                fields[field_name] = values
            except Exception as exc:
                self._logger.warning(
                    "BrowserSubBot: selector extraction failed",
                    extra={"field": field_name, "selector": selector, "error": str(exc)},
                )
                fields[field_name] = []

        if not any(fields.values()):
            raise ParseErrorException(
                f"No data extracted using CSS selectors: {list(css_selectors.keys())}"
            )

        return fields

    async def _take_screenshot(
        self, page, path_override: Optional[str]
    ) -> Optional[str]:
        """Save a screenshot and return base64-encoded PNG, or None on failure."""
        try:
            screenshot_dir = Path(
                path_override or self._config.sub_bot.screenshot_dir
            )
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            path = screenshot_dir / f"browser_error_{ts}.png"
            await page.screenshot(path=str(path), full_page=False)
            with open(path, "rb") as fh:
                return base64.b64encode(fh.read()).decode()
        except Exception as exc:
            self._logger.warning(
                "BrowserSubBot: screenshot failed", extra={"error": str(exc)}
            )
            return None
