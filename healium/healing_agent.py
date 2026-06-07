"""
healium/healing_agent.py

LangGraph-based self-healing agent for Playwright.
"""

import os
import re
import json
import logging
from collections import deque
from datetime import datetime
from typing import TypedDict, Optional, List, Annotated, Literal

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from playwright.sync_api import Error as PlaywrightError

from healium.mcp_client import MCPBrowserClient, _resolve_to_playwright_locator
from healium.models import HealingEvent
import operator

logger = logging.getLogger("healium")

import httpx
_original_httpx_init = httpx.Client.__init__

def _patched_httpx_init(self, *args, **kwargs):
    kwargs['verify'] = False
    _original_httpx_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_httpx_init


class HealingState(TypedDict):
    selector: str
    intent: str
    action: str
    value: Optional[str]
    tenant_id: str
    context_type: str
    dom_context: str
    rag_context: str
    suggestions: List[dict]
    current_suggestion_idx: int
    healed_locator: str
    status: str
    retry_count: int
    screenshot_url: str
    reasoning: str
    confidence: float
    agent_log: Annotated[List[str], operator.add]


SYSTEM_PROMPT = """You are a Self-Healing Web Automation Engine.
A Playwright test has failed because a web element locator is broken due to a UI change.

Your task: analyze the provided page context and find the correct replacement locator.

Context you will receive:
  - Accessibility Tree: structured like [searchbox] "Search Amazon.in" placeholder="Search"
  - Raw HTML: cleaned DOM with id, class, name, data-testid attributes

Locator priority (BEST to WORST):
  1. getByTestId('value')                        <- most stable
  2. getByRole('role', name='visible-name')       <- semantic
  3. getByLabel('label') or getByPlaceholder('x') <- form elements
  4. CSS selector (#id or .class)                 <- fragile but common
  5. XPath //tag[@attr='value']                   <- last resort

RAG context shows past successful healings with similar intent - use these patterns.

RESPOND ONLY WITH VALID JSON. No markdown. No explanation outside the JSON:
{
  "suggestions": [
    {
      "locator": "getByRole('searchbox', name='Search Amazon.in')",
      "confidence": 0.97,
      "reasoning": "Searchbox role with exact accessible name from A11y tree"
    },
    {
      "locator": "getByPlaceholder('Search')",
      "confidence": 0.91,
      "reasoning": "Placeholder text is stable and present in DOM"
    }
  ]
}

Maximum 3 suggestions, ranked by confidence descending."""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def build_healing_agent(memory, page, cache, storage, tenant_id: str):
    mcp = MCPBrowserClient(page)

    raw_keys = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", "")).split(",")
    key_pool  = deque([k.strip() for k in raw_keys if k.strip()])
    if not key_pool:
        raise ValueError("No GROQ_API_KEYS found in .env")

    def _get_llm() -> ChatGroq:
        key = key_pool[0]
        key_pool.rotate(-1)
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=key,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    def cache_check(state: HealingState) -> dict:
        logger.info("Agent -> cache_check")
        if not cache:
            return {"status": "running", "agent_log": ["Cache: no provider configured"]}
        cached = cache.get(tenant_id, state["selector"])
        if cached:
            return {
                "healed_locator": cached,
                "status": "healed",
                "confidence": 1.0,
                "reasoning": "Retrieved from runtime cache",
                "screenshot_url": "",
                "agent_log": [f"Cache HIT: {cached}"],
            }
        return {"status": "running", "agent_log": ["Cache MISS"]}

    def capture_context(state: HealingState) -> dict:
        logger.info("Agent -> capture_context")
        tools          = mcp.get_tools()
        snapshot_tool  = next(t for t in tools if t.name == "browser_snapshot")
        snippet        = snapshot_tool.invoke({})

        if len(snippet) >= 50 and "failed" not in snippet.lower():
            return {
                "dom_context": snippet,
                "context_type": "A11y",
                "agent_log": [f"A11y tree captured ({len(snippet)} chars)"],
            }

        html_tool = next(t for t in tools if t.name == "browser_get_html")
        snippet   = html_tool.invoke({
            "selector_hint": state["selector"],
            "intent_hint":   state["intent"],
        })
        return {
            "dom_context": snippet,
            "context_type": "DOM",
            "agent_log": [f"A11y empty, HTML fallback ({len(snippet)} chars)"],
        }

    def rag_query(state: HealingState) -> dict:
        logger.info("Agent -> rag_query")
        results  = memory.query_vector_memory(state["intent"])
        rag_text = "\n".join(results) if results else "No similar past healings found."
        return {
            "rag_context": rag_text,
            "agent_log": [f"RAG: {len(results)} past patterns retrieved"],
        }

    def llm_reason(state: HealingState) -> dict:
        logger.info(f"Agent -> llm_reason (context: {state['context_type']})")
        llm    = _get_llm()
        prompt = (
            f"Failed locator: `{state['selector']}`\n"
            f"Action needed: {state['action']}\n"
            f"Intent: {state['intent']}\n"
            f"Context type: {state['context_type']}\n\n"
            f"Page context:\n```\n{state['dom_context'][:3000]}\n```\n\n"
            f"Past healing patterns (RAG):\n{state['rag_context']}\n\n"
            f"Find the correct replacement locator."
        )
        try:
            response = llm.invoke([HumanMessage(content=SYSTEM_PROMPT + "\n\n" + prompt)])
            data     = json.loads(_strip_json_fences(response.content))
            suggestions = [
                {
                    "locator":    s["locator"],
                    "confidence": float(s.get("confidence", 0.5)),
                    "reasoning":  s.get("reasoning", ""),
                }
                for s in data.get("suggestions", [])
            ]
            top = suggestions[0]["confidence"] if suggestions else 0
            return {
                "suggestions":            suggestions,
                "current_suggestion_idx": 0,
                "retry_count":            state["retry_count"] + 1,
                "agent_log": [
                    f"LLM: {len(suggestions)} suggestions, top confidence {top:.0%}"
                ],
            }
        except Exception as e:
            logger.error(f"LLM failed: {e}")
            return {
                "suggestions": [],
                "status": "failed",
                "agent_log": [f"LLM error: {e}"],
            }

    def verify_playwright(state: HealingState) -> dict:
        logger.info("Agent -> verify_playwright")
        suggestions = state.get("suggestions", [])
        idx         = state.get("current_suggestion_idx", 0)

        if not suggestions or idx >= len(suggestions):
            return {
                "status": "failed",
                "agent_log": ["All suggestions exhausted - giving up"]
            }

        suggestion  = suggestions[idx]
        locator_str = suggestion["locator"]
        logger.info(f"  Trying: {locator_str} ({suggestion['confidence']:.0%})")

        try:
            locator = _resolve_to_playwright_locator(page, locator_str)
            action  = state["action"]

            if action == "click":
                locator.click(timeout=2000)
            elif action == "fill":
                locator.fill(state.get("value") or "", timeout=2000)
            elif action == "inner_text":
                locator.inner_text(timeout=2000)

            screenshot_url = ""
            if storage:
                try:
                    screenshot_url = storage.save_screenshot(tenant_id, locator, idx)
                except Exception as ss_err:
                    logger.debug(f"Screenshot skipped: {ss_err}")

            if cache:
                cache.set(tenant_id, state["selector"], locator_str)

            logger.info(f"  SUCCESS: {locator_str}")
            return {
                "healed_locator": locator_str,
                "status":         "healed",
                "confidence":     suggestion["confidence"],
                "reasoning":      suggestion["reasoning"],
                "screenshot_url": screenshot_url,
                "agent_log": [
                    f"Verified: {locator_str} ({suggestion['confidence']:.0%})"
                ],
            }

        except Exception as e:
            logger.warning(f"  FAILED: {locator_str} - {str(e)[:60]}")
            return {
                "current_suggestion_idx": idx + 1,
                "agent_log": [f"Attempt {idx + 1} failed: {locator_str}"],
            }

    def escalate(state: HealingState) -> dict:
        logger.warning("Agent -> escalate (switching to raw DOM)")
        tools     = mcp.get_tools()
        html_tool = next(t for t in tools if t.name == "browser_get_html")
        snippet   = html_tool.invoke({
            "selector_hint": state["selector"],
            "intent_hint":   state["intent"],
        })
        return {
            "dom_context":  snippet,
            "context_type": "DOM (escalated)",
            "agent_log": [f"Escalated to raw DOM ({len(snippet)} chars)"],
        }

    def route_after_cache(state: HealingState) -> Literal["__end__", "capture_context"]:
        return END if state["status"] == "healed" else "capture_context"

    def route_after_verify(
        state: HealingState,
    ) -> Literal["__end__", "verify_playwright", "escalate"]:
        if state["status"] in ("healed", "failed"):
            return END
        idx         = state.get("current_suggestion_idx", 0)
        suggestions = state.get("suggestions", [])
        if idx < len(suggestions):
            return "verify_playwright"
        context_type = state.get("context_type", "")
        retry_count  = state.get("retry_count", 0)
        if context_type == "A11y" and retry_count < 2:
            return "escalate"
        return END

    def route_after_escalate(state: HealingState) -> Literal["llm_reason"]:
        return "llm_reason"

    g = StateGraph(HealingState)
    g.add_node("cache_check",       cache_check)
    g.add_node("capture_context",   capture_context)
    g.add_node("rag_query",         rag_query)
    g.add_node("llm_reason",        llm_reason)
    g.add_node("verify_playwright", verify_playwright)
    g.add_node("escalate",          escalate)

    g.set_entry_point("cache_check")

    g.add_conditional_edges("cache_check", route_after_cache, {
        END: END, "capture_context": "capture_context",
    })
    g.add_edge("capture_context", "rag_query")
    g.add_edge("rag_query",       "llm_reason")
    g.add_edge("llm_reason",      "verify_playwright")
    g.add_conditional_edges("verify_playwright", route_after_verify, {
        END: END, "verify_playwright": "verify_playwright", "escalate": "escalate",
    })
    g.add_conditional_edges("escalate", route_after_escalate, {
        "llm_reason": "llm_reason",
    })

    return g.compile()


class SelfHealingPage:
    def __init__(
        self,
        page,
        memory,
        tenant_id: str = "default",
        cache_provider=None,
        storage_provider=None,
        event_store=None,
    ):
        self._page       = page
        self._memory     = memory
        self.tenant_id   = tenant_id
        self.cache       = cache_provider
        self.storage     = storage_provider
        self.event_store = event_store
        self.healing_events: List[HealingEvent] = []

        self._agent = build_healing_agent(
            memory=memory,
            page=page,
            cache=cache_provider,
            storage=storage_provider,
            tenant_id=tenant_id,
        )

    def __getattr__(self, name):
        """Proxy all other attributes to the real Playwright Page.
        This means page.locator(), page.evaluate(), page.screenshot() etc. all work normally.
        """
        return getattr(self._page, name)

    @property
    def page(self):
        return self._page

    def goto(self, url: str, **kwargs):
        self._page.goto(url, **kwargs)

    def click(self, selector: str, intent: str = "", **kwargs):
        self._run(action="click", selector=selector, intent=intent, **kwargs)

    def fill(self, selector: str, value: str, intent: str = "", **kwargs):
        self._run(action="fill", selector=selector, intent=intent, value=value, **kwargs)

    def inner_text(self, selector: str, intent: str = "") -> str:
        return self._run(action="inner_text", selector=selector, intent=intent)

    def _run(self, action: str, selector: str, intent: str,
             value: str = None, **kwargs):
        try:
            locator = _resolve_to_playwright_locator(self._page, selector)
            if action == "click":
                locator.click(timeout=4000, **kwargs)
            elif action == "fill":
                locator.fill(value or "", timeout=4000, **kwargs)
            elif action == "inner_text":
                return locator.inner_text(timeout=4000, **kwargs)
            return
        except PlaywrightError as e:
            original_error = e

        logger.warning(f"LOCATOR FAILED: {selector} - invoking healing agent")

        initial_state = {
            "selector":               selector,
            "intent":                 intent or "N/A",
            "action":                 action,
            "value":                  value,
            "tenant_id":              self.tenant_id,
            "context_type":           "",
            "dom_context":            "",
            "rag_context":            "",
            "suggestions":            [],
            "current_suggestion_idx": 0,
            "healed_locator":         "",
            "status":                 "running",
            "retry_count":            0,
            "screenshot_url":         "",
            "reasoning":              "",
            "confidence":             0.0,
            "agent_log":              [],
        }

        final_state = self._agent.invoke(initial_state)

        for log_line in final_state.get("agent_log", []):
            logger.info(f"  [agent] {log_line}")

        healed = final_state.get("status") == "healed"

        event = self._record_event(
            original       = selector,
            intent         = intent,
            action         = action,
            healed         = final_state.get("healed_locator", ""),
            confidence     = final_state.get("confidence", 0.0),
            source         = "ai" if healed else "none",
            status         = "healed" if healed else "failed",
            reasoning      = final_state.get("reasoning", ""),
            screenshot_url = final_state.get("screenshot_url", ""),
            dom_context    = final_state.get("dom_context", ""),
            rag_context    = final_state.get("rag_context", ""),
        )

        if healed:
            self._memory.save_to_vector_memory(event)
            logger.info(f"HEALED: {selector} -> {final_state['healed_locator']}")
            if action == "inner_text":
                try:
                    locator = _resolve_to_playwright_locator(
                        self._page, final_state["healed_locator"]
                    )
                    return locator.inner_text(timeout=2000)
                except Exception:
                    return ""
            return

        raise original_error

    def _build_code_fix(self, action: str, healed: str) -> str:
        if not healed:
            return ""
        s = healed.strip()
        playwright_prefixes = (
            "getByRole(", "getByTestId(", "getByLabel(",
            "getByPlaceholder(", "getByText(",
        )
        if any(s.startswith(p) for p in playwright_prefixes):
            return f"page.{s}.{action}()"
        elif s.startswith("//") or s.startswith("(//"):
            return f"page.locator('xpath={s}').{action}()"
        else:
            return f"page.locator('{s}').{action}()"

    def _record_event(
        self, original, intent, action, healed, confidence, source, status,
        reasoning="", screenshot_url="", dom_context="", rag_context=""
    ) -> HealingEvent:
        event = HealingEvent(
            timestamp        = datetime.now().isoformat(),
            original_locator = original,
            intent           = intent or "N/A",
            action           = action,
            healed_locator   = healed,
            confidence       = confidence,
            source           = source,
            status           = status,
            reasoning        = reasoning,
            screenshot_path  = screenshot_url,
            dom_context      = dom_context,
            rag_context      = rag_context,
        )
        self.healing_events.append(event)

        event_dict = {
            "tenant_id":        self.tenant_id,
            "timestamp":        event.timestamp,
            "original_locator": event.original_locator,
            "intent":           event.intent,
            "action":           event.action,
            "healed_locator":   event.healed_locator,
            "confidence":       event.confidence,
            "source":           event.source,
            "status":           event.status,
            "reasoning":        event.reasoning,
            "screenshot_url":   screenshot_url,
            "code_fix":         self._build_code_fix(action, healed),
            "dom_context":      (
                dom_context[:500] + "..." if len(dom_context) > 500 else dom_context
            ),
            "rag_context": rag_context,
        }

        if self.event_store:
            try:
                self.event_store.save_event(event_dict)
            except Exception as e:
                logger.error(f"EventStore save failed: {e}")

        try:
            import requests
            requests.post(
                "http://localhost:8000/api/heal",
                json=event_dict,
                timeout=1,
            )
        except Exception:
            pass

        return event