import playwright.sync_api
from browsergym.core.task import AbstractBrowserTask

from ..utils.browser import STEALTH_INIT_SCRIPTS, install_page_readiness_tracking


class GenericBrowserTask(AbstractBrowserTask):
    goal: str = ""
    start_url: str = ""

    def __init__(self, seed: int):
        super().__init__(seed)
        self.viewport = {"width": 1280, "height": 720}
        self.slow_mo = 0
        self.timeout = 15000

    @classmethod
    def get_task_id(cls):
        return "dyna-generic-task"

    def setup(self, page: playwright.sync_api.Page):
        # Install the shared compatibility scripts before the first navigation.
        # Adding them after env.reset() is too late for the initial document.
        install_page_readiness_tracking(page.context)
        for script in STEALTH_INIT_SCRIPTS:
            page.context.add_init_script(script)

        url = self.start_url
        if "bestbuy.com" in url and "intl=" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}intl=nosplash"

        # Retry logic for transient navigation failures
        max_retries = 2
        for attempt in range(max_retries):
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                return self.goal, {}
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                else:
                    raise RuntimeError(f"Failed to navigate to {url} after {max_retries} attempts: {str(e)}")

    def teardown(self) -> None:
        return None

    def validate(self, page: playwright.sync_api.Page, chat_messages: list):
        init_msg = "Hi! I am your UI assistant"
        for msg in reversed(chat_messages):
            role = msg.get("role")
            text = msg.get("message", "")
            if role == "infeasible":
                return 0.0, True, "agent marked infeasible", {}
            if role == "assistant" and not text.startswith(init_msg):
                return 1.0, True, f"assistant responded: {text}", {}
        return 0.0, False, "", {}
