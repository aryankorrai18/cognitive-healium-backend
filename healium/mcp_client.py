"""
healium/mcp_client.py

Python-native MCP-style browser tools for Cognitive Healium.
"""

import re
import logging
from bs4 import BeautifulSoup, Comment
from langchain_core.tools import tool

logger = logging.getLogger("healium")


class MCPBrowserClient:
    def __init__(self, page):
        self._page = page

    def get_tools(self) -> list:
        page = self._page

        @tool
        def browser_snapshot() -> str:
            """Get the accessibility tree of the current page."""
            try:
                snapshot = page.accessibility.snapshot()
                if not snapshot:
                    return "Accessibility tree is empty - use browser_get_html instead."
                text = _flatten_snapshot(snapshot, indent=0)
                return text[:3000] + "\n... (truncated)" if len(text) > 3000 else text
            except Exception as e:
                return f"Snapshot failed: {e}"

        @tool
        def browser_get_html(selector_hint: str = "",
                             intent_hint: str = "") -> str:
            """Get cleaned HTML snippet from the live page DOM."""
            try:
                full_html = page.content()
                cleaned   = _clean_html(full_html)
                region    = _find_region(cleaned, selector_hint, intent_hint)
                return region[:4000] + "\n<!-- truncated -->" \
                    if len(region) > 4000 else region
            except Exception as e:
                return f"HTML capture failed: {e}"

        @tool
        def browser_find_by_text(text: str) -> str:
            """Find elements on the page containing specific text."""
            try:
                elements = page.locator(f"text={text}").all()
                results  = []
                for el in elements[:5]:
                    try:
                        info = el.evaluate("""el => JSON.stringify({
                            tag: el.tagName.toLowerCase(),
                            id: el.id || null,
                            class: el.className ? el.className.substring(0, 80) : null,
                            name: el.getAttribute('name'),
                            type: el.getAttribute('type'),
                            placeholder: el.getAttribute('placeholder'),
                            testid: el.getAttribute('data-testid'),
                            role: el.getAttribute('role')
                        })""")
                        results.append(info)
                    except Exception:
                        pass
                return "\n".join(results) if results \
                    else f"No elements found containing '{text}'"
            except Exception as e:
                return f"Error: {e}"

        @tool
        def browser_verify_selector(selector: str) -> str:
            """Check if a selector exists on the current page."""
            try:
                locator = _resolve_to_playwright_locator(page, selector)
                count   = locator.count()
                if count == 0:
                    return f"NOT FOUND: '{selector}' matches 0 elements"
                text = ""
                try:
                    text = locator.first.inner_text(timeout=1000)[:100]
                except Exception:
                    pass
                return (
                    f"EXISTS: '{selector}' matches {count} element(s). "
                    f"First element text: '{text}'"
                )
            except Exception as e:
                return f"ERROR checking '{selector}': {e}"

        @tool
        def browser_get_url() -> str:
            """Get the current page URL and title."""
            try:
                return f"URL: {page.url} | Title: {page.title()}"
            except Exception as e:
                return f"Error: {e}"

        return [
            browser_snapshot,
            browser_get_html,
            browser_find_by_text,
            browser_verify_selector,
            browser_get_url,
        ]


def _flatten_snapshot(node: dict, indent: int) -> str:
    if not node:
        return ""
    role  = node.get("role", "")
    name  = node.get("name", "")
    value = node.get("value", "")

    if role in ("none", "presentation", "generic") and not name:
        parts = [_flatten_snapshot(c, indent) for c in node.get("children", [])]
        return "\n".join(p for p in parts if p)

    line = "  " * indent + f"[{role}]"
    if name:
        line += f' "{name}"'
    if value:
        line += f' value="{value}"'

    children   = node.get("children", [])
    child_text = "\n".join(
        l for l in (_flatten_snapshot(c, indent + 1) for c in children) if l
    )
    return (line + ("\n" + child_text if child_text else "")).strip()


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all([
        "script", "style", "link", "meta", "noscript", "svg", "path",
        "img", "iframe", "video", "canvas", "footer", "header", "nav"
    ]):
        tag.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    for tag in soup.find_all(True):
        if tag.has_attr("style"):
            del tag["style"]
        for attr in [a for a in list(tag.attrs)
                     if a.startswith("data-") and a not in ("data-testid", "data-test")]:
            del tag[attr]
    cleaned = str(soup)
    cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned.strip()


def _find_region(html: str, failed_selector: str, intent: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    if intent:
        keywords = [w.lower() for w in intent.split() if len(w) > 3]
        for keyword in keywords:
            for element in soup.find_all(string=re.compile(keyword, re.I)):
                parent = element.find_parent()
                if parent:
                    ancestor = parent
                    for _ in range(3):
                        if ancestor.parent:
                            ancestor = ancestor.parent
                    return str(ancestor)

    fragments = re.findall(r"[\w-]+", failed_selector)
    for fragment in fragments:
        if len(fragment) < 4:
            continue
        for el in soup.find_all(attrs={"id": True}):
            if fragment in el["id"].lower():
                ancestor = el
                for _ in range(2):
                    if ancestor.parent:
                        ancestor = ancestor.parent
                return str(ancestor)

    body = soup.find("body")
    return str(body) if body else html


def _resolve_to_playwright_locator(page, selector: str):
    s = selector.strip()

    if s.startswith("getByTestId("):
        match = re.search(r"['\"](.+?)['\"]", s)
        return page.get_by_test_id(match.group(1) if match else s)

    elif s.startswith("getByRole("):
        role_match = re.match(
            r"getByRole\(\s*['\"](\w+)['\"]"
            r"(?:\s*,\s*(?:name=)?['\"](.+?)['\"])?\s*\)", s
        )
        if role_match:
            role = role_match.group(1)
            name = role_match.group(2)
            if name:
                return page.get_by_role(role, name=name)
            else:
                logger.warning(
                    f"getByRole('{role}') has no name - may match multiple elements"
                )
                return page.get_by_role(role)
        logger.warning(f"Unparseable getByRole: {s}, falling back to locator()")
        return page.locator(s)

    elif s.startswith("getByLabel("):
        match = re.search(r"['\"](.+?)['\"]", s)
        return page.get_by_label(match.group(1) if match else s)

    elif s.startswith("getByPlaceholder("):
        match = re.search(r"['\"](.+?)['\"]", s)
        return page.get_by_placeholder(match.group(1) if match else s)

    elif s.startswith("getByText("):
        match = re.search(r"['\"](.+?)['\"]", s)
        return page.get_by_text(match.group(1) if match else s)

    elif s.startswith("//") or s.startswith("(//"):
        return page.locator(f"xpath={s}")

    else:
        return page.locator(s)
