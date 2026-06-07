from dataclasses import dataclass
from typing import Optional


@dataclass
class ElementIntent:
    description: str
    role: str = ""
    action_type: str = "click"
    expected_effect: str = ""


@dataclass
class HealingSuggestion:
    locator: str
    locator_type: str = "css"
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class HealingEvent:
    timestamp: str
    original_locator: str
    intent: str
    action: str
    healed_locator: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    status: str = "failed"
    source: str = ""
    screenshot_path: str = ""
    dom_context: str = ""
    rag_context: str = ""
