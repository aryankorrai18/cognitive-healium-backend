import os
import time
import logging
import subprocess
import webbrowser
import socket
import argparse
import shutil
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from playwright.sync_api import sync_playwright, Error as PlaywrightError

from healium.healing_agent import SelfHealingPage
from healium.memory import HealiumMemory
from healium.providers import get_providers
from healium.selenium_wrapper import SelfHealingDriver

console = Console()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("healium")
logger.setLevel(logging.INFO)

TARGET_URL = "https://www.amazon.in/"

BREAK_THE_UI_JS = """
    const searchInput = document.getElementById('twotabsearchtextbox');
    if (searchInput) {
        searchInput.removeAttribute('id');
        searchInput.classList.add('search-bar-v3');
    }
"""


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def run_baseline_test(page) -> bool:
    page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=60000)
    time.sleep(3)
    try:
        page.fill("#twotabsearchtextbox", "Wireless Headphones", timeout=10000)
        return True
    except Exception:
        return False


def run_sdk_test(healing_page: SelfHealingPage) -> bool:
    healing_page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=60000)
    time.sleep(3)
    try:
        healing_page.page.evaluate(BREAK_THE_UI_JS)
    except Exception:
        pass
    healing_page.fill(
        "#twotabsearchtextbox",
        "Wireless Headphones",
        intent="Amazon search bar input"
    )
    return True


def generate_report(healing_events: list, output_path: str = "healing_report.html"):
    rows = ""
    for e in healing_events:
        rows += f"""
        <tr class="{'healed' if e.status == 'healed' else 'failed'}">
            <td><code>{e.original_locator}</code></td>
            <td><code>{e.healed_locator or 'NONE'}</code></td>
            <td>{e.confidence:.0%}</td>
            <td>{e.source}</td>
            <td style="font-size:12px">{e.reasoning[:120] if e.reasoning else ''}...</td>
        </tr>"""

    healed_count = sum(1 for e in healing_events if e.status == "healed")
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Healium Report</title>
  <style>
    body  {{ font-family: system-ui; background: #0f0f1a; color: #e0e0e0; padding: 2rem; }}
    h1    {{ color: #00e676; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    th    {{ background: #1a1a2e; padding: 12px; text-align: left; font-size:13px; }}
    td    {{ padding: 10px; border-bottom: 1px solid #2a2a4a; font-size: 13px; }}
    .healed td:first-child {{ border-left: 4px solid #00e676; }}
    .failed td:first-child  {{ border-left: 4px solid #ff5252; }}
    code {{ color: #448aff; }}
    .summary {{ background:#1a1a2e; padding:16px; border-radius:8px; margin-bottom:1rem; }}
  </style>
</head>
<body>
<h1>Cognitive Healium - Agentic Test Report</h1>
<div class="summary">
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
  Total events: {len(healing_events)} &nbsp;|&nbsp;
  Healed: <span style="color:#00e676">{healed_count}</span> &nbsp;|&nbsp;
  Failed: <span style="color:#ff5252">{len(healing_events) - healed_count}</span>
</div>
<table>
  <tr>
    <th>Original Locator</th>
    <th>Healed To</th>
    <th>Confidence</th>
    <th>Source</th>
    <th>AI Reasoning</th>
  </tr>
  {rows}
</table>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(output_path)
    webbrowser.open(f"file://{abs_path}")
    console.print(f"Report: {output_path}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh", action="store_true",
        help="Clear ChromaDB memory before run (recommended for demos)"
    )
    args, _ = parser.parse_known_args()

    if args.fresh:
        shutil.rmtree("data/", ignore_errors=True)
        console.print("[yellow]ChromaDB cleared - starting fresh[/yellow]")

    if not is_port_in_use(8000):
        console.print("[dim]Starting dashboard server...[/dim]")
        subprocess.Popen(
            ["python", "-m", "uvicorn", "server:app", "--port", "8000"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(2)
        webbrowser.open("http://localhost:8000")
    else:
        console.print("[dim]Dashboard server already running.[/dim]")

    console.print(Panel.fit(
        "[bold cyan]Cognitive Healium[/bold cyan]\n"
        "[dim]Agentic Self-Healing Test Automation - LangGraph + Groq + ChromaDB[/dim]\n\n"
        "Target: [bold]amazon.in[/bold]  |  "
        "LLM: [bold]Groq LLaMA 3.3 70B[/bold]  |  "
        "Memory: [bold]ChromaDB[/bold]\n"
        "[dim]Dashboard -> http://localhost:8000[/dim]",
        border_style="cyan",
        title="[bold white]v2.0.0[/bold white]"
    ))

    cache, storage, event_store = get_providers()
    tenant_id = os.getenv("TENANT_ID", "team_virtusa")
    all_events = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        )

        # PHASE 1
        console.print(
            "\n[bold green]=== PHASE 1: DAY 1 - QA Engineer writes test ===[/bold green]"
        )
        page_v1 = context.new_page()
        try:
            passed = run_baseline_test(page_v1)
            if passed:
                console.print("[bold green]PASS[/bold green] Search bar found on Amazon.")
            else:
                console.print("[yellow]Amazon may have blocked automation[/yellow]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
        page_v1.close()
        time.sleep(2)

        # PHASE 2
        console.print(
            "\n[bold red]=== PHASE 2: DAY 5 - Developer renames the element ===[/bold red]"
        )
        page_v2 = context.new_page()
        page_v2.goto(TARGET_URL, wait_until="domcontentloaded", timeout=60000)
        time.sleep(3)
        page_v2.evaluate(BREAK_THE_UI_JS)
        try:
            page_v2.fill("#twotabsearchtextbox", "Wireless Headphones", timeout=5000)
            console.print("[yellow]Unexpected pass - Amazon may have changed their DOM[/yellow]")
        except PlaywrightError:
            console.print(
                "[bold red]FAIL[/bold red] CI/CD TEST FAILED! "
                "Locator '#twotabsearchtextbox' not found.\n"
                "[dim]Pipeline is red.[/dim]"
            )
        page_v2.close()
        time.sleep(2)

        # PHASE 3
        console.print(
            "\n[bold cyan]=== PHASE 3: DAY 5 - Cognitive Healium activated ===[/bold cyan]"
        )
        memory       = HealiumMemory(tenant_id=tenant_id)
        page_v3      = context.new_page()
        healing_page = SelfHealingPage(
            page             = page_v3,
            memory           = memory,
            tenant_id        = tenant_id,
            cache_provider   = cache,
            storage_provider = storage,
            event_store      = event_store,
        )

        try:
            passed = run_sdk_test(healing_page)
            if passed:
                console.print(
                    "\n[bold green]CI/CD TEST PASSED![/bold green] "
                    "LangGraph agent healed the locator automatically."
                )
        except Exception as e:
            console.print(f"\n[bold red]Healing Failed:[/bold red] {e}")

        console.print("\n[bold blue]Agent Decision Log:[/bold blue]")
        for event in healing_page.healing_events:
            color = "green" if event.status == "healed" else "red"
            console.print(
                f"  [{color}]{event.status.upper()}[/{color}] "
                f"{event.original_locator} -> "
                f"{event.healed_locator or 'NONE'} "
                f"(conf: {event.confidence:.0%}, src: {event.source})"
            )
            if event.reasoning:
                console.print(f"    [dim italic]{event.reasoning}[/dim italic]")

        all_events.extend(healing_page.healing_events)
        page_v3.close()
        browser.close()

    # PHASE 4: Selenium cross-framework
    console.print(
        "\n[bold magenta]=== PHASE 4: Selenium - Shared RAG Memory ===[/bold magenta]"
    )
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service

        opts = Options()
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--start-maximized")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])

        sel_driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=opts,
        )

        healing_driver = SelfHealingDriver(
            driver           = sel_driver,
            memory           = memory,      # same ChromaDB as Phase 3!
            tenant_id        = tenant_id,
            cache_provider   = cache,
            storage_provider = storage,
            event_store      = event_store,
        )

        healing_driver.get(TARGET_URL)
        time.sleep(3)
        healing_driver.execute_script(BREAK_THE_UI_JS)

        healing_driver.fill(
            By.ID,
            "twotabsearchtextbox",
            "Selenium Healed!",
            intent="Amazon search bar input"
        )

        console.print(
            "[bold green]SELENIUM HEALED![/bold green] "
            "Cross-framework healing via shared ChromaDB memory."
        )

        for event in healing_driver.healing_events:
            color = "green" if event.status == "healed" else "red"
            console.print(
                f"  [{color}]{event.status.upper()}[/{color}] "
                f"{event.original_locator} -> {event.healed_locator} "
                f"(conf: {event.confidence:.0%}, src: {event.source})"
            )

        all_events.extend(healing_driver.healing_events)
        sel_driver.quit()

    except ImportError as ie:
        console.print(f"[yellow]Skipping Phase 4: {ie}[/yellow]")
        console.print("[dim]Install with: pip install selenium webdriver-manager[/dim]")
    except Exception as se:
        console.print(f"[yellow]Phase 4 attempted but failed: {se}[/yellow]")

    generate_report(all_events)
    console.print("\n[bold blue]Dashboard -> http://localhost:8000[/bold blue]")


if __name__ == "__main__":
    main()
