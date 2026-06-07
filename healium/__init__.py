from healium.models import ElementIntent, HealingSuggestion, HealingEvent
from healium.memory import HealiumMemory
from healium.providers import get_providers
from healium.dom_capture import DOMCapture


def __getattr__(name):
    """Lazy-load heavy modules only when actually used.
    Prevents playwright/selenium from being imported just by doing 'import healium'.
    """
    if name == "SelfHealingPage":
        from healium.healing_agent import SelfHealingPage
        return SelfHealingPage
    if name == "SelfHealingDriver":
        from healium.selenium_wrapper import SelfHealingDriver
        return SelfHealingDriver
    raise AttributeError(f"module 'healium' has no attribute '{name}'")


__all__ = [
    "ElementIntent",
    "HealingSuggestion",
    "HealingEvent",
    "HealiumMemory",
    "SelfHealingPage",
    "SelfHealingDriver",
    "get_providers",
    "DOMCapture",
]