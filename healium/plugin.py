"""
healium/plugin.py

pytest plugin for Cognitive Healium.
Auto-registers when the cognitive-healium package is installed
via the pytest11 entry point in pyproject.toml — no conftest needed.

CLI flags:
    --healium-enabled       activate self-healing
    --healium-tenant=NAME   isolate ChromaDB collection (default: pytest-session)
    --healium-fresh         wipe ChromaDB before the run
"""

import shutil
import logging
import pytest

logger = logging.getLogger("healium")


def pytest_addoption(parser):
    g = parser.getgroup("healium", "Cognitive Healium self-healing")
    g.addoption(
        "--healium-enabled", action="store_true", default=False,
        help="Enable Cognitive Healium self-healing for all tests"
    )
    g.addoption(
        "--healium-tenant", default="pytest-session",
        help="ChromaDB collection name / tenant ID (default: pytest-session)"
    )
    g.addoption(
        "--healium-fresh", action="store_true", default=False,
        help="Clear ChromaDB memory before the run"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "healium: mark test as using Cognitive Healium self-healing"
    )


@pytest.fixture(scope="session")
def _healium_enabled(request) -> bool:
    return request.config.getoption("--healium-enabled")


@pytest.fixture(scope="session")
def _healium_tenant(request) -> str:
    return request.config.getoption("--healium-tenant")


@pytest.fixture(scope="session")
def healium_providers(_healium_enabled):
    if not _healium_enabled:
        return None, None, None
    from healium.providers import get_providers
    return get_providers()


@pytest.fixture(scope="session")
def healium_memory(request, _healium_enabled, _healium_tenant):
    if not _healium_enabled:
        return None

    if request.config.getoption("--healium-fresh"):
        shutil.rmtree("data/", ignore_errors=True)
        logger.info("Healium: ChromaDB cleared (--healium-fresh)")

    from healium.memory import HealiumMemory
    mem = HealiumMemory(tenant_id=_healium_tenant)
    logger.info(f"Healium: memory ready (tenant: {_healium_tenant})")
    return mem


@pytest.fixture
def healing_page(page, _healium_enabled, healium_memory, healium_providers):
    if not _healium_enabled or healium_memory is None:
        yield page
        return

    from healium.healing_agent import SelfHealingPage
    cache, storage, event_store = healium_providers

    hp = SelfHealingPage(
        page             = page,
        memory           = healium_memory,
        tenant_id        = healium_memory.tenant_id,
        cache_provider   = cache,
        storage_provider = storage,
        event_store      = event_store,
    )
    yield hp

    if hp.healing_events:
        healed = sum(1 for e in hp.healing_events if e.status == "healed")
        logger.info(
            f"Healium [playwright | {hp.tenant_id}]: "
            f"{healed}/{len(hp.healing_events)} locators auto-healed"
        )


@pytest.fixture
def healing_driver_factory(_healium_enabled, healium_memory, healium_providers):
    """Factory fixture — wraps a Selenium WebDriver with self-healing.

    Usage:
        def test_something(healing_driver_factory):
            driver = webdriver.Chrome(options=opts)
            h = healing_driver_factory(driver)
            h.fill(By.ID, "broken-id", "text", intent="search input")
    """
    if not _healium_enabled or healium_memory is None:
        def passthrough(driver):
            return driver
        return passthrough

    from healium.selenium_wrapper import SelfHealingDriver
    cache, storage, event_store = healium_providers

    def factory(driver):
        return SelfHealingDriver(
            driver           = driver,
            memory           = healium_memory,
            tenant_id        = healium_memory.tenant_id,
            cache_provider   = cache,
            storage_provider = storage,
            event_store      = event_store,
        )
    return factory
