"""Shared browser configuration for the Dyna project.

Single source of truth for:
- Browser identity and conservative anti-bot compatibility settings
- Playwright launch / context kwargs
- BrowserGym-compatible parameter dicts
- Homepage fetching and accessibility checking
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from playwright.sync_api import Page, sync_playwright

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stealth configuration
# ---------------------------------------------------------------------------

STEALTH_INIT_SCRIPTS: List[str] = [
    'Object.defineProperty(navigator, "webdriver", {get: () => undefined})',
    "Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})",
    "Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']})",
    "window.chrome = { runtime: {} }",
    """const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
            Promise.resolve({ state: Notification.permission }) :
            originalQuery(parameters)
    )""",
]

# Context-wide headers are attached to every request, including cross-origin
# fetch/XHR calls. Keep only CORS-safelisted, request-type-independent fields
# here. Chromium must generate Accept, Accept-Encoding, Sec-CH-UA, Sec-Fetch-*
# and Upgrade-Insecure-Requests for each individual request; hard-coding the
# values of a top-level navigation breaks APIs and subresources.
STEALTH_HEADERS: Dict[str, str] = {
    "Accept-Language": "en-US,en;q=0.9",
}

# Playwright 1.44 bundles Chromium 125 in the project's dyna environment.
# Override the default HeadlessChrome UA while leaving browser-controlled
# client hints untouched. Update this together with the Playwright runtime.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

# BrowserGym owns Chromium's launch args. Keep the standalone pipeline on the
# same launch behavior; document-start init scripts provide the shared
# webdriver compatibility treatment in both environments.
CHROMIUM_LAUNCH_ARGS: List[str] = []

# A page can satisfy ``DOMContentLoaded`` while its useful UI is still being
# rendered by JavaScript.  Wait for a short, bounded quiet window before an
# observation is handed to either the dataset pipeline or the runtime agent.
PAGE_READY_MIN_WAIT_MS = 3000
PAGE_READY_IDLE_POLL_MS = 300
PAGE_READY_LOADING_POLL_MS = 1000
PAGE_READY_STABLE_ROUNDS = 3
PAGE_READY_NETWORK_QUIET_MS = 1000
PAGE_READY_FALLBACK_MS = 5000
PAGE_READY_TIMEOUT_MS = 15000

IMPORTANT_RESOURCE_TYPES = {"document", "script", "xhr", "fetch"}


@dataclass
class PageNetworkTracker:
    """Tracks requests that can materially change the page seen by the agent."""

    pending: Set[Any]
    last_activity: float


def install_page_readiness_tracking(context: Any) -> PageNetworkTracker:
    """Install one context-wide request tracker before the first navigation."""
    existing = getattr(context, "_dyna_page_network_tracker", None)
    if existing is not None:
        return existing

    tracker = PageNetworkTracker(pending=set(), last_activity=time.monotonic())
    setattr(context, "_dyna_page_network_tracker", tracker)

    def site_key(hostname: str) -> str:
        parts = [part for part in hostname.lower().split(".") if part]
        if len(parts) <= 2:
            return ".".join(parts)
        return ".".join(parts[-2:])

    def is_important(request: Any) -> bool:
        try:
            if request.resource_type == "document":
                return True
            if request.resource_type not in IMPORTANT_RESOURCE_TYPES:
                return False
            request_host = urlparse(request.url).hostname or ""
            frame_host = urlparse(request.frame.url).hostname or ""
            return bool(request_host and frame_host) and site_key(request_host) == site_key(frame_host)
        except Exception:
            return False

    def on_request(request: Any) -> None:
        if is_important(request):
            tracker.pending.add(request)
            tracker.last_activity = time.monotonic()

    def on_request_done(request: Any) -> None:
        if request in tracker.pending:
            tracker.pending.discard(request)
            tracker.last_activity = time.monotonic()

    context.on("request", on_request)
    context.on("requestfinished", on_request_done)
    context.on("requestfailed", on_request_done)
    return tracker

# Block-detection signals for accessibility checks
BLOCK_SIGNALS: List[str] = [
    "access denied",
    "403 forbidden",
    "just a moment",       # Cloudflare challenge
    "checking your browser",
    "verify you are human",
    "captcha",
    "are you a robot",
    "blocked",
    "not available in your region",
    "err_empty_response",
    "page not found",
    "404",
    "don't have permission",
    "unable to process",
    "request denied",
]

# ---------------------------------------------------------------------------
# BrowserGym-compatible parameter dicts
# ---------------------------------------------------------------------------


def get_pw_chromium_kwargs() -> Dict[str, Any]:
    """Playwright chromium.launch() kwargs for BrowserGym's pw_chromium_kwargs.

    Note: BrowserGym already passes its own 'args' to chromium.launch(),
    so we cannot include 'args' here (would cause duplicate keyword error).
    The --disable-blink-features=AutomationControlled flag is handled via
    STEALTH_INIT_SCRIPTS instead (navigator.webdriver spoofing etc.).
    """
    return {}


def get_pw_context_kwargs() -> Dict[str, Any]:
    """Shared context identity for Pipeline Playwright and BrowserGym.

    Do not add navigation- or resource-specific headers here: Playwright
    applies ``extra_http_headers`` to every request in the context.
    """
    return {
        "user_agent": USER_AGENT,
        "extra_http_headers": STEALTH_HEADERS,
    }


def wait_for_page_ready(
    page: Page,
    *,
    timeout_ms: int = PAGE_READY_TIMEOUT_MS,
    min_wait_ms: int = PAGE_READY_MIN_WAIT_MS,
    idle_poll_ms: int = PAGE_READY_IDLE_POLL_MS,
    loading_poll_ms: int = PAGE_READY_LOADING_POLL_MS,
    stable_rounds: int = PAGE_READY_STABLE_ROUNDS,
    network_quiet_ms: int = PAGE_READY_NETWORK_QUIET_MS,
    fallback_ms: int = PAGE_READY_FALLBACK_MS,
) -> bool:
    """Wait until the useful page state is stable, without waiting forever.

    Modern sites often continue ads, analytics, or streaming requests after
    their usable content is ready, so ``networkidle`` is not a safe readiness
    definition.  Instead, require a minimum render window and several identical
    snapshots of the user-visible document, while ensuring that document,
    script, XHR, and fetch requests have finished and stayed quiet. Prefer
    ``readyState=complete``; after a bounded fallback window, accept a stable
    interactive document.

    Returns True when stability was observed and False when the hard timeout
    was reached.  Timeout is deliberately non-fatal: callers should continue
    with the best page state available.
    """
    timeout_ms = max(0, int(timeout_ms))
    if timeout_ms == 0:
        return False
    min_wait_ms = max(0, min(int(min_wait_ms), timeout_ms))
    idle_poll_ms = max(50, int(idle_poll_ms))
    loading_poll_ms = max(idle_poll_ms, int(loading_poll_ms))
    stable_rounds = max(1, int(stable_rounds))
    network_quiet_ms = max(0, int(network_quiet_ms))
    fallback_ms = max(min_wait_ms, min(int(fallback_ms), timeout_ms))
    context = getattr(page, "context", None)
    tracker = getattr(context, "_dyna_page_network_tracker", None)

    started = time.monotonic()
    try:
        page.wait_for_load_state(
            "domcontentloaded",
            timeout=max(1, min(timeout_ms, 3000)),
        )
    except Exception:
        # A persistent frame/resource must not prevent a usable observation.
        pass

    elapsed_ms = int((time.monotonic() - started) * 1000)
    initial_wait_ms = min(min_wait_ms, max(0, timeout_ms - elapsed_ms))
    if initial_wait_ms:
        page.wait_for_timeout(initial_wait_ms)

    previous: Optional[tuple] = None
    unchanged = 0
    elapsed_ms = int((time.monotonic() - started) * 1000)
    max_checks = max(1, (max(0, timeout_ms - elapsed_ms) // idle_poll_ms) + 1)

    for check_index in range(max_checks):
        try:
            state = page.evaluate(
                """() => {
                    const body = document.body;
                    const text = (body?.innerText || '')
                        .replace(/\\s+/g, ' ')
                        .trim()
                        .slice(0, 20000);
                    return {
                        readyState: document.readyState,
                        href: location.href,
                        title: document.title,
                        text: text,
                        interactiveCount: body
                            ? body.querySelectorAll('a,button,input,select,textarea,[role="button"],[role="link"]').length
                            : 0,
                    };
                }"""
            )
        except Exception:
            state = None

        if state:
            signature = (
                state.get("href", ""),
                state.get("title", ""),
                state.get("text", ""),
                state.get("interactiveCount", 0),
            )
            unchanged = unchanged + 1 if signature == previous else 1
            previous = signature

            elapsed_ms = int((time.monotonic() - started) * 1000)
            document_usable = bool(state.get("text")) and state.get("readyState") in {
                "interactive",
                "complete",
            }
            load_complete = state.get("readyState") == "complete"
            fallback_ready = elapsed_ms >= fallback_ms
            still_loading = not load_complete or bool(
                tracker is not None and tracker.pending
            )
            network_ready = tracker is None or (
                not tracker.pending
                and (time.monotonic() - tracker.last_activity) * 1000
                >= network_quiet_ms
            )
            if (
                document_usable
                and unchanged >= stable_rounds
                and network_ready
                and (load_complete or fallback_ready)
            ):
                logger.debug(
                    "Page ready after %.2fs: %s",
                    time.monotonic() - started,
                    state.get("href", ""),
                )
                return True

        if check_index < max_checks - 1:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            remaining_ms = max(0, timeout_ms - elapsed_ms)
            if remaining_ms == 0:
                break
            next_poll_ms = loading_poll_ms if state and still_loading else idle_poll_ms
            page.wait_for_timeout(min(next_poll_ms, remaining_ms))

    logger.warning(
        "Page readiness timed out after %.2fs; continuing with current state: %s",
        time.monotonic() - started,
        getattr(page, "url", ""),
    )
    return False


# ---------------------------------------------------------------------------
# Homepage fetching
# ---------------------------------------------------------------------------


@dataclass
class HomepageResult:
    """Result of a homepage fetch attempt."""
    accessible: bool
    description: str
    block_reason: str = ""


@dataclass
class RuntimeAccessResult:
    """Result of runtime-style navigation stability check."""
    accessible: bool
    stable: bool
    attempts: int
    success_count: int
    fail_count: int
    reasons: List[str]
    final_url_samples: List[str]


def fetch_homepage(
    url: str,
    timeout_ms: int = 45000,
    max_retries: int = 3,
) -> HomepageResult:
    """Open a homepage with a stealth headless browser and check accessibility."""
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            return _fetch_once(url, timeout_ms)
        except Exception as e:
            last_error = str(e)[:200]
            logger.warning(
                "Homepage fetch attempt %d/%d failed for %s: %s",
                attempt, max_retries, url, last_error,
            )
            if attempt < max_retries:
                time.sleep(2)

    return HomepageResult(
        accessible=False,
        description="",
        block_reason=f"Failed to load after {max_retries} attempts: {last_error}",
    )


def _fetch_once(url: str, timeout_ms: int) -> HomepageResult:
    """Single attempt to fetch and check a homepage."""
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=CHROMIUM_LAUNCH_ARGS,
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            locale="en-US",
            **get_pw_context_kwargs(),
        )
        install_page_readiness_tracking(context)
        for script in STEALTH_INIT_SCRIPTS:
            context.add_init_script(script)
        page = context.new_page()
        try:
            # Match the runner: interactive sites often keep ad/analytics
            # resources alive long after the usable DOM is ready.
            resp = page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=timeout_ms,
            )
            wait_for_page_ready(page)
        except Exception as e:
            browser.close()
            raise

        try:
            # Check HTTP status
            status = resp.status if resp else 0
            if status in (401, 403, 404, 451, 500, 502, 503):
                browser.close()
                return HomepageResult(
                    accessible=False,
                    description="",
                    block_reason=f"HTTP {status}",
                )

            # Check page content for block signals
            body_text = page.inner_text("body").lower()
            for signal in BLOCK_SIGNALS:
                if signal in body_text and len(body_text) < 2000:
                    browser.close()
                    return HomepageResult(
                        accessible=False,
                        description="",
                        block_reason=f"Block signal detected: '{signal}'",
                    )

            description = extract_page_description(page, body_text)
        finally:
            browser.close()

        return HomepageResult(accessible=True, description=description)


def extract_page_description(page: Page, body_text_lower: str = "") -> str:
    """Extract structured description from a loaded Playwright page."""
    parts: list[str] = []

    # 1. Navigation tabs / links
    nav_texts: list[str] = []
    for el in page.query_selector_all("nav a, nav button, [role=tab]"):
        txt = (el.inner_text() or "").strip()
        if txt and len(txt) < 60:
            nav_texts.append(txt)
    if nav_texts:
        parts.append("Navigation tabs: " + ", ".join(nav_texts[:15]))

    # 2. Input fields
    inputs: list[str] = []
    for el in page.query_selector_all("input, select, textarea"):
        placeholder = el.get_attribute("placeholder") or ""
        label = el.get_attribute("aria-label") or ""
        name = el.get_attribute("name") or ""
        desc = placeholder or label or name
        if desc:
            inputs.append(desc)
    if inputs:
        parts.append("Form fields: " + ", ".join(inputs[:15]))

    # 3. Buttons
    buttons: list[str] = []
    for el in page.query_selector_all("button"):
        txt = (el.inner_text() or "").strip()
        if txt and len(txt) < 40 and txt.lower() not in ("close", "x", "dismiss"):
            buttons.append(txt)
    if buttons:
        parts.append("Buttons: " + ", ".join(buttons[:10]))

    # 4. First visible text lines (page context)
    body_text = page.inner_text("body")
    lines = [l.strip() for l in body_text.split("\n") if l.strip()][:25]
    parts.append("Visible text (first 25 lines):\n" + "\n".join(lines))

    return "\n".join(parts) if parts else "Empty or unreadable homepage."


def probe_runtime_access(
    url: str,
    attempts: int = 3,
    timeout_ms: int = 30000,
) -> RuntimeAccessResult:
    """Probe website accessibility using runtime-like navigation semantics.

    This mimics runner_lib/task_env.py setup behavior (goto with domcontentloaded)
    and repeats attempts to estimate stability across runs.
    """
    reasons: List[str] = []
    final_url_samples: List[str] = []
    success_count = 0

    for _ in range(max(1, attempts)):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=CHROMIUM_LAUNCH_ARGS)
                context = browser.new_context(
                    viewport={"width": 1280, "height": 720},
                    locale="en-US",
                    **get_pw_context_kwargs(),
                )
                install_page_readiness_tracking(context)
                for script in STEALTH_INIT_SCRIPTS:
                    context.add_init_script(script)
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                wait_for_page_ready(page)
                final_url_samples.append(page.url)
                success_count += 1
                browser.close()
        except Exception as e:
            reasons.append(f"{type(e).__name__}: {str(e)[:300]}")

    fail_count = max(1, attempts) - success_count
    accessible = success_count > 0
    stable = success_count == max(1, attempts)
    return RuntimeAccessResult(
        accessible=accessible,
        stable=stable,
        attempts=max(1, attempts),
        success_count=success_count,
        fail_count=fail_count,
        reasons=reasons,
        final_url_samples=final_url_samples,
    )
