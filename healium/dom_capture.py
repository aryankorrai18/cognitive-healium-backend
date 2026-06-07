import re
import logging
from bs4 import BeautifulSoup, Comment

logger = logging.getLogger("healium")


class DOMCapture:
    def get_accessibility_snapshot(self, page, intent: str = "",
                                   max_chars: int = 3000) -> str:
        try:
            snapshot = page.accessibility.snapshot()
            if not snapshot:
                return ""
            text = self._flatten_snapshot(snapshot, indent=0)
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... (truncated)"
            return text
        except Exception as e:
            logger.warning(f"Accessibility snapshot failed: {e}")
            return ""

    def _flatten_snapshot(self, node: dict, indent: int) -> str:
        if not node:
            return ""
        role  = node.get("role", "")
        name  = node.get("name", "")
        value = node.get("value", "")

        if role in ("none", "presentation", "generic") and not name:
            parts = [self._flatten_snapshot(c, indent) for c in node.get("children", [])]
            return "\n".join(p for p in parts if p)

        line = "  " * indent + f"[{role}]"
        if name:
            line += f' "{name}"'
        if value:
            line += f' value="{value}"'

        children   = node.get("children", [])
        child_text = "\n".join(
            l for l in (self._flatten_snapshot(c, indent + 1) for c in children) if l
        )
        return (line + ("\n" + child_text if child_text else "")).strip()

    def get_relevant_snippet(self, page, failed_selector: str,
                             intent: str = "", max_chars: int = 4000) -> str:
        full_html = page.content()
        cleaned   = self._clean_html(full_html)
        snippet   = self._find_region(cleaned, failed_selector, intent)
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "\n<!-- ... truncated ... -->"
        return snippet

    def _clean_html(self, html: str) -> str:
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

    def _find_region(self, html: str, failed_selector: str, intent: str) -> str:
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
