"""
healium/selenium_wrapper.py

Self-Healing Selenium WebDriver wrapper for Cognitive Healium.
"""

import os
import re
import json
import logging
from datetime import datetime
from collections import deque
from typing import Optional, List, Dict

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

from healium.memory import HealiumMemory
from healium.models import HealingEvent

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

import httpx
_original_httpx_init = httpx.Client.__init__
def _patched_httpx_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _original_httpx_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_httpx_init

logger = logging.getLogger("healium")

BY_MAP = {
    "ID":                By.ID,
    "NAME":              By.NAME,
    "CSS_SELECTOR":      By.CSS_SELECTOR,
    "CSS":               By.CSS_SELECTOR,
    "XPATH":             By.XPATH,
    "CLASS_NAME":        By.CLASS_NAME,
    "CLASS":             By.CLASS_NAME,
    "TAG_NAME":          By.TAG_NAME,
    "TAG":               By.TAG_NAME,
    "LINK_TEXT":         By.LINK_TEXT,
    "PARTIAL_LINK_TEXT": By.PARTIAL_LINK_TEXT,
}

SELENIUM_SYSTEM_PROMPT = """You are a Self-Healing Web Automation Engine for Selenium.
A Selenium test has failed because a web element locator is broken due to a UI change.

Analyze the provided HTML and find the correct replacement locator.

Locator priority (BEST to WORST):
  1. ID           e.g. "search-bar-v3"
  2. NAME         e.g. "q"
  3. CSS_SELECTOR e.g. "input.search-bar-v3"
  4. XPATH        e.g. "//input[@placeholder='Search']"
  5. CLASS_NAME   - last resort

RAG context shows past successful healings with similar intent - use these patterns first.

RESPOND ONLY WITH VALID JSON. No markdown. No explanation outside the JSON:
{
  "suggestions": [
    {
      "locator_type": "CSS_SELECTOR",
      "locator_value": "input.search-bar-v3",
      "confidence": 0.95,
      "reasoning": "ID was renamed; new class is visible in the DOM"
    }
  ]
}

locator_type must be exactly one of: ID, NAME, CSS_SELECTOR, XPATH, CLASS_NAME, TAG_NAME
Maximum 3 suggestions, ranked by confidence descending."""


def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class SelfHealingDriver:
    def __init__(
        self,
        driver: WebDriver,
        memory: HealiumMemory,
        tenant_id: str = "default",
        cache_provider=None,
        storage_provider=None,
        event_store=None,
    ):
        self._driver        = driver
        self._memory        = memory
        self.tenant_id      = tenant_id
        self.cache          = cache_provider
        self.storage        = storage_provider
        self.event_store    = event_store
        self.healing_events: List[HealingEvent] = []

        raw_keys = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", "")).split(",")
        self._key_pool = deque([k.strip() for k in raw_keys if k.strip()])
        if not self._key_pool:
            logger.warning("No GROQ_API_KEYS found. Selenium healing will not work.")

        logger.info(f"SelfHealingDriver initialised (tenant: {tenant_id})")

    def __getattr__(self, name):
        return getattr(self._driver, name)

    def get(self, url: str):
        self._driver.get(url)

    def quit(self):
        self._driver.quit()

    def execute_script(self, script: str, *args):
        return self._driver.execute_script(script, *args)

    def find_element(self, by: By, value: str, intent: str = "") -> WebElement:
        try:
            return self._driver.find_element(by, value)
        except NoSuchElementException as original_error:
            if not intent:
                raise

            logger.warning(f"SELENIUM LOCATOR FAILED: {by}='{value}' | intent: '{intent}'")

            cache_key = f"selenium:{by}:{value}"
            if self.cache:
                cached = self.cache.get(self.tenant_id, cache_key)
                if cached:
                    c_by, c_val = self._parse_locator_string(cached)
                    try:
                        elem = self._driver.find_element(c_by, c_val)
                        logger.info(f"  CACHE HIT: {cached}")
                        return elem
                    except NoSuchElementException:
                        logger.debug("  Cache entry stale - falling through to LLM")

            dom_snippet = self._driver.page_source[:4000]

            rag_list = self._memory.query_vector_memory(intent)
            rag_text = "\n".join(rag_list) if rag_list else "No past healings found."
            logger.info(f"  RAG: {len(rag_list)} past patterns retrieved")

            suggestions = self._ask_llm(
                failed_locator=f"{by}={value}",
                intent=intent,
                dom_context=dom_snippet,
                rag_context=rag_text,
            )
            if not suggestions:
                logger.error("  LLM returned no suggestions")
                raise original_error

            for suggestion in suggestions:
                s_type  = suggestion.get("locator_type", "CSS_SELECTOR")
                s_val   = suggestion.get("locator_value", "")
                conf    = float(suggestion.get("confidence", 0.5))
                reason  = suggestion.get("reasoning", "")
                by_enum = BY_MAP.get(s_type.upper(), By.CSS_SELECTOR)

                try:
                    elem = self._driver.find_element(by_enum, s_val)
                    logger.info(
                        f"  HEALED: {by}='{value}' -> {s_type}='{s_val}' ({conf:.0%})"
                    )

                    if self.cache:
                        self.cache.set(self.tenant_id, cache_key, f"{s_type}={s_val}")

                    event = self._record_event(
                        original    = f"{by}={value}",
                        intent      = intent,
                        action      = "find_element",
                        healed      = f"{s_type}={s_val}",
                        confidence  = conf,
                        source      = "ai",
                        reasoning   = reason,
                        dom_context = dom_snippet[:500],
                        rag_context = rag_text[:500],
                    )
                    self._memory.save_to_vector_memory(event)
                    return elem

                except NoSuchElementException:
                    logger.warning(f"  Suggestion failed: {s_type}='{s_val}'")

            raise original_error

    def fill(self, by: By, value: str, text: str, intent: str = ""):
        element = self.find_element(by, value, intent=intent)
        import time
        time.sleep(0.5)
        try:
            element.clear()
        except Exception:
            pass
        element.send_keys(text)

    def click(self, by: By, value: str, intent: str = ""):
        element = self.find_element(by, value, intent=intent)
        element.click()

    def get_text(self, by: By, value: str, intent: str = "") -> str:
        return self.find_element(by, value, intent=intent).text

    def _get_llm(self) -> ChatGroq:
        if not self._key_pool:
            raise ValueError("No GROQ_API_KEYS in .env")
        key = self._key_pool[0]
        self._key_pool.rotate(-1)
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=key,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def _ask_llm(
        self,
        failed_locator: str,
        intent: str,
        dom_context: str,
        rag_context: str,
    ) -> List[Dict]:
        prompt = (
            f"Failed locator: `{failed_locator}`\n"
            f"Intent: {intent}\n\n"
            f"Page HTML snippet:\n```html\n{dom_context[:3000]}\n```\n\n"
            f"Past healing patterns (RAG):\n{rag_context}\n\n"
            f"Provide the correct Selenium replacement locator."
        )
        try:
            llm      = self._get_llm()
            response = llm.invoke(
                [HumanMessage(content=SELENIUM_SYSTEM_PROMPT + "\n\n" + prompt)]
            )
            data        = json.loads(_strip_json(response.content))
            suggestions = data.get("suggestions", [])
            logger.info(f"  LLM returned {len(suggestions)} suggestions")
            return suggestions
        except Exception as e:
            logger.error(f"  LLM call failed: {e}")
            return []

    def _parse_locator_string(self, locator_str: str):
        if "=" in locator_str:
            method, val = locator_str.split("=", 1)
            return BY_MAP.get(method.upper(), By.CSS_SELECTOR), val
        return By.CSS_SELECTOR, locator_str

    def _record_event(
        self,
        original: str, intent: str, action: str, healed: str,
        confidence: float, source: str, reasoning: str = "",
        dom_context: str = "", rag_context: str = "",
    ) -> HealingEvent:
        event = HealingEvent(
            timestamp        = datetime.now().isoformat(),
            original_locator = original,
            intent           = intent or "N/A",
            action           = action,
            healed_locator   = healed,
            confidence       = confidence,
            source           = source,
            status           = "healed",
            reasoning        = reasoning,
            dom_context      = dom_context,
            rag_context      = rag_context,
        )
        self.healing_events.append(event)

        event_dict = {
            "tenant_id":        self.tenant_id,
            "timestamp":        event.timestamp,
            "original_locator": original,
            "intent":           intent or "N/A",
            "action":           action,
            "healed_locator":   healed,
            "confidence":       confidence,
            "source":           source,
            "status":           "healed",
            "reasoning":        reasoning,
            "code_fix":         f"driver.find_element(By.{healed})",
            "dom_context":      dom_context,
            "rag_context":      rag_context,
            "framework":        "selenium",
        }

        if self.event_store:
            try:
                self.event_store.save_event(event_dict)
            except Exception as e:
                logger.error(f"EventStore save failed: {e}")

        try:
            import requests, urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            requests.post(
                "http://localhost:8000/api/heal",
                json=event_dict, timeout=1, verify=False,
            )
        except Exception:
            pass

        return event
