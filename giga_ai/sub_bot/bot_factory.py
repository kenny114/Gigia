"""
bot_factory.py – Sub-bot factory (SubBotType → SubBot instance).

Routing:
  SCRAPER  → ScraperSubBot   (aiohttp + BeautifulSoup, fast, static pages)
  SELENIUM → SeleniumSubBot  (undetected-chromedriver or Playwright, legacy)
  BROWSER  → BrowserSubBot   (Playwright with stealth, JS-heavy / Maps / SPAs)
  GENERIC  → BrowserSubBot   (safe default — handles anything a plain scraper can't)
  SKILL    → SkillSubBot     (calls back to almcp /api/brain/execute)
  CODE     → CodeSubBot      (sandboxed Python subprocess execution)
  FILE     → FileSubBot      (read/write files scoped to workspace dir)
  SHELL    → ShellSubBot     (whitelisted shell commands on the VPS)
"""

from __future__ import annotations

from giga_ai.messaging.message_schemas import SubBotType
from giga_ai.sub_bot.sub_bot import SubBot


class BotFactory:
    """Static factory that maps a SubBotType to the correct SubBot class."""

    @staticmethod
    def create(sub_bot_type: SubBotType, config=None) -> SubBot:
        if sub_bot_type == SubBotType.SCRAPER:
            from giga_ai.sub_bot.scraper_sub_bot import ScraperSubBot
            return ScraperSubBot(config=config)

        if sub_bot_type == SubBotType.SELENIUM:
            from giga_ai.sub_bot.selenium_sub_bot import SeleniumSubBot
            return SeleniumSubBot(config=config)

        if sub_bot_type == SubBotType.SKILL:
            from giga_ai.sub_bot.skill_sub_bot import SkillSubBot
            return SkillSubBot(config=config)

        if sub_bot_type == SubBotType.CODE:
            from giga_ai.sub_bot.code_sub_bot import CodeSubBot
            return CodeSubBot(config=config)

        if sub_bot_type == SubBotType.FILE:
            from giga_ai.sub_bot.file_sub_bot import FileSubBot
            return FileSubBot(config=config)

        if sub_bot_type == SubBotType.SHELL:
            from giga_ai.sub_bot.shell_sub_bot import ShellSubBot
            return ShellSubBot(config=config)

        # BROWSER and GENERIC both use BrowserSubBot (full Playwright stack)
        from giga_ai.sub_bot.browser_sub_bot import BrowserSubBot
        return BrowserSubBot(config=config)
