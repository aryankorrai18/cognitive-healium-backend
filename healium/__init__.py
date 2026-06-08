"""
Cognitive Healium — Self-Healing Test Automation SDK

Public API:
    HealiumMemory        — ChromaDB vector memory for past healings
    SelfHealingPage      — Playwright page with self-healing locators
    SelfHealingDriver    — Selenium driver with self-healing locators
    get_providers        — Auto-detect cache/storage/event-store providers
    HealingEvent         — Data record of a healing occurrence
"""

from healium.models import ElementIntent, HealingSuggestion, HealingEvent
from healium.memory import HealiumMemory
from healium.providers import get_providers


def __getattr__(name):
    """Lazy-load heavy modules only when actually used.
    Avoids importing Playwright/Selenium until they're needed.
    """
    if name == "SelfHealingPage":
        from healium.healing_agent import SelfHealingPage
        return SelfHealingPage
    if name == "SelfHealingDriver":
        from healium.selenium_wrapper import SelfHealingDriver
        return SelfHealingDriver
    raise AttributeError(f"module 'healium' has no attribute {name}")


__all__ = [
    "ElementIntent",
    "HealingSuggestion",
    "HealingEvent",
    "HealiumMemory",
    "SelfHealingPage",
    "SelfHealingDriver",
    "get_providers",
]