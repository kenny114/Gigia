"""
selenium_sub_bot.py – Dynamic-site sub-bot using Selenium or Playwright.

This sub-bot is designed for pages that require JavaScript execution,
user-interaction simulation, or detection of dynamically loaded content.

Driver back-ends (selected by instruction parameter ``driver``)
---------------------------------------------------------------
  "selenium"    – uses undetected-chromedriver (default)
  "playwright"  – uses Playwright async API (chromium)

Instruction parameters (SubBotInstruction.parameters)
------------------------------------------------------
url               (str)            – Target URL (required)
driver            (str)            – "selenium" | "playwright" (default: "selenium")
actions           (list[dict])     – Sequence of browser actions to perform
                                      before extracting data (see _run_actions).
wait_for          (str)            – CSS selector to wait for after navigation
wait_timeout      (float)          – Seconds to wait (default: 10)
css_selectors     (dict)           – {field_name: css_selector} for extraction
screenshot_on_error (bool)         – Override config screenshot setting
screenshot_path   (str)            – Override config screenshot directory
proxy             (str)            – HTTP proxy URL
user_agent        (str)            – Custom User-Agent

Action dict schema (for ``actions`` list)
-----------------------------------------
Each action is a dict with a ``type`` key.  Supported types:
  {"type": "click",       "selector": "css-selector"}
  {"type": "type",        "selector": "css-selector", "text": "..."}
  {"type": "wait",        "seconds": 1.5}
  {"type": "scroll",      "pixels": 500}
  {"type": "wait_for",    "selector": "css-selector", "timeout": 10}

Returns
-------
dict with keys:
  url           – Final URL after navigation
  page_source   – Full page HTML (capped at 500 KB)
  fields        – Extracted CSS-selector data (if css_selectors provided)
  screenshot    – Base64-encoded PNG if screenshot_on_error triggered
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    "cloudflare", "challenge", "verify you are human",
])


# ---------------------------------------------------------------------------
# Helper: run sync selenium code in executor
# ---------------------------------------------------------------------------

def _run_in_executor(func, *args):
    """Run a blocking function in the default thread-pool executor."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, func, *args)


# ---------------------------------------------------------------------------
# SeleniumSubBot
# ---------------------------------------------------------------------------

class SeleniumSubBot(SubBot):
    """
    Browser automation sub-bot.

    Supports both undetected-chromedriver (Selenium) and Playwright.
    All browser I/O is run in a thread-pool executor to avoid blocking
    the asyncio event loop.

    Parameters
    ----------
    config:
        Config override; loaded from singleton if not supplied.
    """

    # ------------------------------------------------------------------
    # SubBot._run implementation
    # ------------------------------------------------------------------

    async def _run(self, instruction: SubBotInstruction) -> Dict[str, Any]:
        params = instruction.parameters
        url: str = params.get("url", "")
        if not url:
            raise ParseErrorException("No 'url' provided in instruction parameters")

        driver_type: str = params.get("driver", "selenium").lower()

        if driver_type == "playwright":
            return await self._run_playwright(params, url)
        else:
            return await self._run_selenium(params, url)

    # ------------------------------------------------------------------
    # Selenium (undetected-chromedriver)
    # ------------------------------------------------------------------

    async def _run_selenium(self, params: dict, url: str) -> Dict[str, Any]:
        """Execute via Selenium/undetected-chromedriver in a thread executor."""
        def _sync_selenium() -> Dict[str, Any]:
            try:
                import undetected_chromedriver as uc  # type: ignore
            except ImportError:
                try:
                    from selenium import webdriver
                    from selenium.webdriver.chrome.options import Options
                    uc = None
                except ImportError as e:
                    raise BrowserCrashException(
                        "Neither undetected-chromedriver nor selenium is installed"
                    ) from e

            options = self._build_chrome_options(params)
            driver = None
            screenshot_b64: Optional[str] = None

            try:
                if uc is not None:
                    driver = uc.Chrome(options=options, use_subprocess=True)
                else:
                    from selenium.webdriver.chrome.options import Options as _Opts
                    driver = webdriver.Chrome(options=options)

                driver.set_page_load_timeout(
                    params.get("wait_timeout",
                               self._config.sub_bot.request_timeout_seconds)
                )

                self._logger.debug("SeleniumSubBot: navigating", extra={"url": url})
                driver.get(url)

                # Perform scripted actions
                actions: List[dict] = params.get("actions", [])
                self._selenium_run_actions(driver, actions)

                # Wait for element
                wait_for_sel: Optional[str] = params.get("wait_for")
                if wait_for_sel:
                    self._selenium_wait_for(driver, wait_for_sel,
                                            params.get("wait_timeout", 10))

                page_source: str = driver.page_source
                final_url: str = driver.current_url

                # Captcha check
                lower = page_source.lower()
                for kw in _CAPTCHA_KEYWORDS:
                    if kw in lower:
                        raise CaptchaDetectedException(
                            f"Captcha/bot-detection at {final_url}: keyword '{kw}'"
                        )

                # Extract fields
                fields = {}
                css_selectors: dict = params.get("css_selectors", {})
                if css_selectors:
                    fields = self._selenium_extract_fields(driver, css_selectors)

                return {
                    "url": final_url,
                    "page_source": page_source[:500_000],
                    "fields": fields,
                    "screenshot": screenshot_b64,
                }

            except (CaptchaDetectedException, ParseErrorException):
                raise
            except Exception as exc:
                # Take screenshot on error
                should_screenshot = params.get(
                    "screenshot_on_error", self._config.sub_bot.screenshot_on_error
                )
                if driver and should_screenshot:
                    screenshot_b64 = self._take_screenshot(driver, params)
                raise BrowserCrashException(f"Selenium error: {exc}") from exc
            finally:
                if driver:
                    try:
                        driver.quit()
                    except Exception:
                        pass

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _sync_selenium)
        except asyncio.CancelledError:
            raise TimeoutException("Selenium task was cancelled (timeout)")

    def _build_chrome_options(self, params: dict):
        """Build Chrome options shared by both uc and standard selenium."""
        try:
            import undetected_chromedriver as uc
            options = uc.ChromeOptions()
        except ImportError:
            from selenium.webdriver.chrome.options import Options
            options = Options()

        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")

        ua: str = params.get("user_agent", self._config.sub_bot.default_user_agent)
        options.add_argument(f"--user-agent={ua}")

        proxy: Optional[str] = params.get("proxy")
        if proxy:
            options.add_argument(f"--proxy-server={proxy}")

        return options

    def _selenium_run_actions(self, driver, actions: List[dict]) -> None:
        """Execute a list of browser action dicts using Selenium."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        for action in actions:
            atype = action.get("type", "")
            try:
                if atype == "click":
                    el = driver.find_element(By.CSS_SELECTOR, action["selector"])
                    el.click()
                elif atype == "type":
                    el = driver.find_element(By.CSS_SELECTOR, action["selector"])
                    el.clear()
                    el.send_keys(action.get("text", ""))
                elif atype == "wait":
                    time.sleep(float(action.get("seconds", 1)))
                elif atype == "scroll":
                    pixels = int(action.get("pixels", 300))
                    driver.execute_script(f"window.scrollBy(0, {pixels});")
                elif atype == "wait_for":
                    WebDriverWait(driver, action.get("timeout", 10)).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, action["selector"])
                        )
                    )
            except Exception as exc:
                self._logger.warning(
                    "SeleniumSubBot: action failed (continuing)",
                    extra={"action": atype, "error": str(exc)},
                )

    def _selenium_wait_for(self, driver, selector: str, timeout: float) -> None:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        try:
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
        except Exception:
            raise TimeoutException(
                f"Element '{selector}' not found after {timeout}s"
            )

    def _selenium_extract_fields(self, driver, css_selectors: dict) -> dict:
        from selenium.webdriver.common.by import By
        fields: Dict[str, Any] = {}
        for field, selector in css_selectors.items():
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                fields[field] = [el.text.strip() for el in elements]
            except Exception as exc:
                self._logger.warning(
                    "SeleniumSubBot: field extraction failed",
                    extra={"field": field, "error": str(exc)},
                )
                fields[field] = []
        return fields

    def _take_screenshot(self, driver, params: dict) -> Optional[str]:
        """Save screenshot and return base64-encoded PNG string."""
        try:
            screenshot_dir = Path(
                params.get("screenshot_path", self._config.sub_bot.screenshot_dir)
            )
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            path = screenshot_dir / f"error_{ts}.png"
            driver.save_screenshot(str(path))
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception as exc:
            self._logger.warning(
                "SeleniumSubBot: screenshot failed", extra={"error": str(exc)}
            )
            return None

    # ------------------------------------------------------------------
    # Playwright
    # ------------------------------------------------------------------

    async def _run_playwright(self, params: dict, url: str) -> Dict[str, Any]:
        """Execute via Playwright (async API)."""
        try:
            from playwright.async_api import async_playwright, TimeoutError as PWTimeout
        except ImportError as exc:
            raise BrowserCrashException(
                "Playwright is not installed.  Run: pip install playwright && playwright install"
            ) from exc

        proxy_config = None
        proxy: Optional[str] = params.get("proxy")
        if proxy:
            proxy_config = {"server": proxy}

        ua: str = params.get("user_agent", self._config.sub_bot.default_user_agent)
        timeout_ms = int(
            params.get("wait_timeout",
                       self._config.sub_bot.request_timeout_seconds) * 1000
        )

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                proxy=proxy_config,
            )
            context = await browser.new_context(user_agent=ua)
            page = await context.new_page()

            try:
                self._logger.debug("SeleniumSubBot[playwright]: navigating", extra={"url": url})
                await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")

                # Actions
                actions: List[dict] = params.get("actions", [])
                await self._playwright_run_actions(page, actions, timeout_ms)

                # Wait for element
                wait_for_sel: Optional[str] = params.get("wait_for")
                if wait_for_sel:
                    try:
                        await page.wait_for_selector(wait_for_sel, timeout=timeout_ms)
                    except PWTimeout:
                        raise TimeoutException(
                            f"Element '{wait_for_sel}' not found within {timeout_ms}ms"
                        )

                page_source = await page.content()
                final_url = page.url

                # Captcha check
                lower = page_source.lower()
                for kw in _CAPTCHA_KEYWORDS:
                    if kw in lower:
                        raise CaptchaDetectedException(
                            f"Captcha/bot-detection at {final_url}: keyword '{kw}'"
                        )

                # Extract fields
                fields: Dict[str, Any] = {}
                css_selectors: dict = params.get("css_selectors", {})
                if css_selectors:
                    fields = await self._playwright_extract_fields(page, css_selectors)

                return {
                    "url": final_url,
                    "page_source": page_source[:500_000],
                    "fields": fields,
                    "screenshot": None,
                }

            except (CaptchaDetectedException, ParseErrorException, TimeoutException):
                raise
            except PWTimeout as exc:
                raise TimeoutException(f"Playwright timeout: {exc}") from exc
            except Exception as exc:
                # Screenshot on error
                should_screenshot = params.get(
                    "screenshot_on_error", self._config.sub_bot.screenshot_on_error
                )
                screenshot_b64: Optional[str] = None
                if should_screenshot:
                    try:
                        screenshot_dir = Path(
                            params.get("screenshot_path",
                                       self._config.sub_bot.screenshot_dir)
                        )
                        screenshot_dir.mkdir(parents=True, exist_ok=True)
                        ts = int(time.time())
                        path = screenshot_dir / f"pw_error_{ts}.png"
                        await page.screenshot(path=str(path))
                        with open(path, "rb") as fh:
                            screenshot_b64 = base64.b64encode(fh.read()).decode()
                    except Exception:
                        pass
                raise BrowserCrashException(f"Playwright error: {exc}") from exc
            finally:
                await browser.close()

    async def _playwright_run_actions(self, page, actions: List[dict], timeout_ms: int) -> None:
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
                        action["selector"], timeout=action.get("timeout", 10) * 1000
                    )
            except Exception as exc:
                self._logger.warning(
                    "SeleniumSubBot[playwright]: action failed (continuing)",
                    extra={"action": atype, "error": str(exc)},
                )

    async def _playwright_extract_fields(self, page, css_selectors: dict) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        for field, selector in css_selectors.items():
            try:
                elements = await page.query_selector_all(selector)
                fields[field] = [
                    (await el.text_content() or "").strip()
                    for el in elements
                ]
            except Exception as exc:
                self._logger.warning(
                    "SeleniumSubBot[playwright]: field extraction failed",
                    extra={"field": field, "error": str(exc)},
                )
                fields[field] = []
        return fields
