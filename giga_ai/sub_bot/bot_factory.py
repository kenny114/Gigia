"""
bot_factory.py – Sub-bot factory (SubBotType → SubBot instance).

This is separate from manager_bot/bot_factory.py which creates ManagerBots.
manager_bot.py imports this via ``giga_ai.sub_bot.bot_factory``.
"""

from __future__ import annotations

from giga_ai.messaging.message_schemas import SubBotType
from giga_ai.sub_bot.sub_bot import SubBot


class BotFactory:
    """Static factory that maps a SubBotType to the correct SubBot class."""

    @staticmethod
    def create(sub_bot_type: SubBotType, config=None) -> SubBot:
        """
        Instantiate and return the correct SubBot for *sub_bot_type*.

        Parameters
        ----------
        sub_bot_type:
            One of ``SubBotType.SCRAPER``, ``SubBotType.SELENIUM``,
            ``SubBotType.GENERIC``.
        config:
            Config override; each sub-bot falls back to the singleton.

        Returns
        -------
        SubBot
            A ready-to-use sub-bot instance.
        """
        if sub_bot_type == SubBotType.SELENIUM:
            from giga_ai.sub_bot.selenium_sub_bot import SeleniumSubBot
            return SeleniumSubBot(config=config)

        # Default (SCRAPER + GENERIC) – use the lightweight HTTP scraper
        from giga_ai.sub_bot.scraper_sub_bot import ScraperSubBot
        return ScraperSubBot(config=config)
