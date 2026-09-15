from src.agent_lib.task_env import GenericBrowserTask
from src.utils.browser import (
    CHROMIUM_LAUNCH_ARGS,
    STEALTH_HEADERS,
    STEALTH_INIT_SCRIPTS,
    USER_AGENT,
    get_pw_context_kwargs,
    install_page_readiness_tracking,
    wait_for_page_ready,
)


NAVIGATION_OR_BROWSER_MANAGED_HEADERS = {
    "accept",
    "accept-encoding",
    "sec-ch-ua",
    "sec-ch-ua-mobile",
    "sec-ch-ua-platform",
    "sec-fetch-dest",
    "sec-fetch-mode",
    "sec-fetch-site",
    "sec-fetch-user",
    "upgrade-insecure-requests",
}


def test_context_headers_are_safe_for_fetch_and_subresources():
    configured = {name.lower() for name in STEALTH_HEADERS}
    assert configured.isdisjoint(NAVIGATION_OR_BROWSER_MANAGED_HEADERS)
    assert STEALTH_HEADERS == {"Accept-Language": "en-US,en;q=0.9"}


def test_shared_context_keeps_normal_browser_identity():
    kwargs = get_pw_context_kwargs()
    assert kwargs["user_agent"] == USER_AGENT
    assert kwargs["extra_http_headers"] == STEALTH_HEADERS
    assert "Chrome/125" in USER_AGENT
    assert "HeadlessChrome" not in USER_AGENT
    assert CHROMIUM_LAUNCH_ARGS == []


def test_runner_installs_stealth_scripts_before_initial_navigation():
    calls = []

    class FakeContext:
        def on(self, event, callback):
            calls.append(("event", event, callback))

        def add_init_script(self, script):
            calls.append(("script", script))

    class FakePage:
        context = FakeContext()

        def goto(self, url, **kwargs):
            calls.append(("goto", url, kwargs))

    task = GenericBrowserTask(seed=0)
    task.start_url = "https://example.com/"
    task.setup(FakePage())

    script_calls = [call[1] for call in calls if call[0] == "script"]
    assert script_calls == STEALTH_INIT_SCRIPTS
    assert calls[-1][0:2] == ("goto", "https://example.com/")


def test_request_tracker_follows_only_page_material_requests():
    handlers = {}

    class FakeContext:
        def on(self, event, callback):
            handlers[event] = callback

    class FakeFrame:
        url = "https://www.example.com/search"

    class FakeRequest:
        def __init__(self, resource_type, url):
            self.resource_type = resource_type
            self.url = url
            self.frame = FakeFrame()

    context = FakeContext()
    tracker = install_page_readiness_tracking(context)
    xhr = FakeRequest("xhr", "https://api.example.com/results")
    image = FakeRequest("image", "https://www.example.com/image.jpg")
    third_party_ad = FakeRequest("fetch", "https://ads.example.net/bid")

    handlers["request"](xhr)
    handlers["request"](image)
    handlers["request"](third_party_ad)
    assert tracker.pending == {xhr}

    handlers["requestfinished"](xhr)
    assert tracker.pending == set()


def test_page_readiness_waits_for_repeated_stable_complete_state():
    calls = []

    class FakePage:
        def wait_for_load_state(self, state, timeout):
            calls.append(("load", state, timeout))

        def wait_for_timeout(self, timeout):
            calls.append(("wait", timeout))

        def evaluate(self, _script):
            calls.append(("evaluate",))
            return {
                "readyState": "complete",
                "href": "https://example.com/",
                "title": "Example",
                "text": "Ready content",
                "interactiveCount": 2,
            }

    ready = wait_for_page_ready(
        FakePage(),
        timeout_ms=1000,
        min_wait_ms=100,
        idle_poll_ms=100,
        loading_poll_ms=200,
        stable_rounds=3,
        fallback_ms=200,
    )

    assert ready is True
    assert calls[0][0:2] == ("load", "domcontentloaded")
    assert calls.count(("evaluate",)) == 3
    assert ("wait", 100) in calls


def test_page_readiness_times_out_without_stability_and_does_not_raise():
    counter = 0

    class ChangingPage:
        def wait_for_load_state(self, state, timeout):
            raise RuntimeError("page never reports DOMContentLoaded")

        def wait_for_timeout(self, timeout):
            pass

        def evaluate(self, _script):
            nonlocal counter
            counter += 1
            return {
                "readyState": "interactive",
                "href": "https://example.com/",
                "title": "Example",
                "text": f"Changing content {counter}",
                "interactiveCount": counter,
            }

    ready = wait_for_page_ready(
        ChangingPage(),
        timeout_ms=300,
        min_wait_ms=100,
        idle_poll_ms=100,
        loading_poll_ms=200,
        stable_rounds=2,
        fallback_ms=200,
    )

    assert ready is False
    assert counter == 4


def test_page_readiness_does_not_accept_stable_dom_with_pending_xhr():
    handlers = {}

    class FakeContext:
        def on(self, event, callback):
            handlers[event] = callback

    class FakeFrame:
        url = "https://www.example.com/search"

    class FakeRequest:
        resource_type = "xhr"
        url = "https://api.example.com/results"
        frame = FakeFrame()

    context = FakeContext()
    install_page_readiness_tracking(context)
    handlers["request"](FakeRequest())

    class FakePage:
        def __init__(self):
            self.context = context

        def wait_for_load_state(self, state, timeout):
            pass

        def wait_for_timeout(self, timeout):
            pass

        def evaluate(self, _script):
            return {
                "readyState": "complete",
                "href": "https://example.com/search",
                "title": "Search",
                "text": "Search is loading",
                "interactiveCount": 1,
            }

    ready = wait_for_page_ready(
        FakePage(),
        timeout_ms=300,
        min_wait_ms=100,
        idle_poll_ms=100,
        loading_poll_ms=200,
        stable_rounds=2,
        network_quiet_ms=0,
    )

    assert ready is False


def test_page_readiness_uses_longer_poll_while_page_is_loading():
    waits = []

    class FakePage:
        context = object()

        def wait_for_load_state(self, state, timeout):
            pass

        def wait_for_timeout(self, timeout):
            waits.append(timeout)

        def evaluate(self, _script):
            return {
                "readyState": "interactive",
                "href": "https://example.com/",
                "title": "Loading",
                "text": "Partial content",
                "interactiveCount": 1,
            }

    wait_for_page_ready(
        FakePage(),
        timeout_ms=400,
        min_wait_ms=100,
        idle_poll_ms=50,
        loading_poll_ms=200,
        stable_rounds=2,
        fallback_ms=400,
    )

    assert waits[0] == 100
    assert 200 in waits[1:]
