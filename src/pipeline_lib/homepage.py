"""Homepage fetching — delegates to utils.browser.

All browser/stealth config and fetching logic lives in utils.browser.
This module re-exports for backward compatibility.
"""

from utils.browser import (
	HomepageResult,
	RuntimeAccessResult,
	extract_page_description,
	fetch_homepage,
	probe_runtime_access,
)

__all__ = [
	"HomepageResult",
	"RuntimeAccessResult",
	"fetch_homepage",
	"probe_runtime_access",
	"extract_page_description",
]
