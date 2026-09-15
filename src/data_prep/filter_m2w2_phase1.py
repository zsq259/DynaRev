"""Phase 1: Filter Mind2Web-2 tasks by website accessibility.

For each task, extract all explicitly required websites from the task description,
then check accessibility using the project's fetch_homepage (Playwright-based).
Reuses cached results from website_check.json where available.
"""

import csv
import json
import logging
import time
from pathlib import Path

from ..pipeline_lib.homepage import fetch_homepage
from ..utils.browser import probe_runtime_access
from ..utils.client import get_client
from ..utils.llm import llm_json_call
from ..utils.site_health import SiteHealthStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "gpt-4o"

EXTRACT_PROMPT = """\
You are an expert at analyzing task instructions.

Given a task description, extract ALL websites that the task EXPLICITLY requires visiting.

Rules:
- Only include websites that the task description explicitly names or clearly implies
  (e.g., "on Amazon", "from Coursera", "using Yelp", "on Sephora US")
- Include the full homepage URL (e.g., "https://www.amazon.com/")
- If the task mentions a specific sub-site, still use the main domain
  (e.g., "AKC Marketplace" → "https://www.akc.org/")
- If the task does NOT specify any particular website and the user can freely choose
  which site to use, return an empty list
- Do NOT infer websites that aren't mentioned — only extract what's explicitly stated
- Common mappings: "Reddit" → "https://www.reddit.com/",
  "Google Maps" → "https://www.google.com/maps/",
  "Google Play Store" → "https://play.google.com/",
  "IMDb" → "https://www.imdb.com/",
  "OpenTable" → "https://www.opentable.com/"

Output JSON:
{
  "websites": ["https://...", ...],
  "notes": "brief explanation of extraction"
}
"""


def extract_websites(client, task: dict) -> dict:
    user_msg = (
        f"Task ID: {task['task_id']}\n"
        f"Description: {task['task_description']}"
    )
    return llm_json_call(
        client, EXTRACT_PROMPT, user_msg, model=MODEL, temperature=0.0
    )


def normalize_url(url: str) -> str:
    """Normalize URL for cache comparison."""
    url = url.rstrip("/")
    return url


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    # Read all M2W2 tasks
    with open(PROJECT_ROOT / "data/sources/mind2web2/test_set.csv") as f:
        tasks = list(csv.DictReader(f))
    logger.info("Loaded %d tasks", len(tasks))

    # Load existing accessibility cache
    cache_path = PROJECT_ROOT / "data/sources/online-mind2web/website_check.json"
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        logger.info("Loaded %d cached website checks", len(cache))

    # Also load recheck results
    recheck_path = PROJECT_ROOT / "data/sources/online-mind2web/website_recheck.json"
    if recheck_path.exists():
        with open(recheck_path) as f:
            recheck = json.load(f)
        # Merge recheck results (newer) into cache
        for url, info in recheck.items():
            cache[url] = info
        logger.info("Merged %d recheck results", len(recheck))

    client = get_client(MODEL)

    # --- Step 1: Extract websites from all tasks ---
    logger.info("--- Step 1: Extracting websites from task descriptions ---")
    task_websites = {}  # task_id -> list of URLs
    all_urls = set()

    for i, task in enumerate(tasks):
        logger.info("[%d/%d] Extracting: %s", i + 1, len(tasks), task["task_id"])
        try:
            result = extract_websites(client, task)
            urls = result.get("websites", [])
            notes = result.get("notes", "")
        except Exception as e:
            logger.error("Failed: %s: %s", task["task_id"], e)
            urls = []
            notes = f"ERROR: {e}"
            time.sleep(2)

        task_websites[task["task_id"]] = {
            "websites": urls,
            "notes": notes,
        }
        all_urls.update(urls)

        if (i + 1) % 10 == 0:
            time.sleep(1)

    logger.info("Extracted %d unique URLs across all tasks", len(all_urls))

    # --- Step 2: Check accessibility ---
    logger.info("--- Step 2: Checking website accessibility ---")
    accessibility = {}  # url -> {accessible, block_reason}
    site_health = SiteHealthStore()
    site_health.register_urls(sorted(all_urls), source="m2w2_phase1")

    # Build cache from site health store first, then fallback to old cache files.
    health_cache = {}
    for url in sorted(all_urls):
        h = site_health.get_health(url)
        if not h:
            continue
        if h.get("runtime_accessible") is None:
            continue
        health_cache[url] = {
            "accessible": bool(h.get("runtime_accessible")),
            "block_reason": str(h.get("last_error", "") or ""),
            "stable": bool(h.get("runtime_stable", False)),
            "runtime_success_count": int(h.get("runtime_success_count", 0) or 0),
            "runtime_attempts": int(h.get("runtime_attempts", 0) or 0),
            "source": "site_health",
        }

    # First, match against cache
    unchecked = []
    for url in sorted(all_urls):
        if url in health_cache:
            accessibility[url] = health_cache[url]
            continue
        norm = normalize_url(url)
        found = False
        for cached_url, info in cache.items():
            if normalize_url(cached_url) == norm:
                accessibility[url] = {
                    "accessible": info["accessible"],
                    "block_reason": info.get("block_reason", ""),
                    "source": "cache",
                }
                found = True
                break
        if not found:
            unchecked.append(url)

    logger.info(
        "Cache hits: %d, need to check: %d",
        len(accessibility), len(unchecked),
    )

    # Check unchecked URLs with both homepage check and runtime-style stability probe
    for i, url in enumerate(unchecked):
        logger.info("[%d/%d] Testing: %s", i + 1, len(unchecked), url)
        start = time.time()
        result = fetch_homepage(url, timeout_ms=45000, max_retries=3)
        runtime_probe = probe_runtime_access(url, attempts=3, timeout_ms=30000)
        elapsed = time.time() - start
        status = "ACCESSIBLE" if runtime_probe.accessible else "BLOCKED"
        stability = "STABLE" if runtime_probe.stable else "UNSTABLE"
        logger.info(
            "  %s/%s (%.1fs) homepage=%s runtime=%d/%d",
            status,
            stability,
            elapsed,
            "OK" if result.accessible else "BLOCKED",
            runtime_probe.success_count,
            runtime_probe.attempts,
        )
        accessibility[url] = {
            # Use runtime probe as source of truth for benchmark executability.
            "accessible": runtime_probe.accessible,
            "stable": runtime_probe.stable,
            "runtime_success_count": runtime_probe.success_count,
            "runtime_attempts": runtime_probe.attempts,
            "runtime_reasons": runtime_probe.reasons,
            "runtime_final_urls": runtime_probe.final_url_samples,
            "homepage_accessible": result.accessible,
            "homepage_block_reason": result.block_reason,
            "block_reason": (
                ""
                if runtime_probe.accessible
                else (
                    runtime_probe.reasons[0]
                    if runtime_probe.reasons
                    else (result.block_reason or "runtime navigation failed")
                )
            ),
            "source": "live_check",
        }
        site_health.update_from_probe(url, accessibility[url], source="m2w2_phase1_live_check")

    site_health.save()
    site_health.export_feasible()

    # --- Step 3: Determine task-level accessibility ---
    logger.info("--- Step 3: Determining task-level results ---")
    results = []
    for task in tasks:
        tid = task["task_id"]
        info = task_websites[tid]
        urls = info["websites"]

        if not urls:
            # No specific website required → passes Phase 1
            status = "pass_no_website"
            blocked_sites = []
        else:
            blocked_sites = []
            for url in urls:
                a = accessibility.get(url)
                if a and not a["accessible"]:
                    blocked_sites.append({
                        "url": url,
                        "reason": a["block_reason"],
                        "stable": bool(a.get("stable", False)),
                        "runtime_success_ratio": f"{a.get('runtime_success_count', 0)}/{a.get('runtime_attempts', 0)}",
                    })
            status = "blocked" if blocked_sites else "pass"

        results.append({
            "task_id": tid,
            "domain": task["domain"],
            "subdomain": task["subdomain"],
            "websites": urls,
            "extraction_notes": info["notes"],
            "status": status,
            "blocked_sites": blocked_sites,
            "task_description": task["task_description"],
        })

    # --- Save & Summary ---
    out_path = PROJECT_ROOT / "data/sources/mind2web2/L2_filtered/m2w2_phase1_accessible.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved to %s", out_path)

    passed = [r for r in results if r["status"] in ("pass", "pass_no_website")]
    blocked = [r for r in results if r["status"] == "blocked"]
    pass_with_site = [r for r in results if r["status"] == "pass"]
    pass_no_site = [r for r in results if r["status"] == "pass_no_website"]

    print("\n" + "=" * 60)
    print("PHASE 1 RESULTS: Website Accessibility")
    print("=" * 60)
    print(f"Total tasks:              {len(results)}")
    print(f"PASS (website OK):        {len(pass_with_site)}")
    print(f"PASS (no specific site):  {len(pass_no_site)}")
    print(f"BLOCKED:                  {len(blocked)}")
    print()

    if blocked:
        print("BLOCKED tasks:")
        for r in blocked:
            sites = ", ".join(s["url"] for s in r["blocked_sites"])
            reasons = "; ".join(s["reason"][:40] for s in r["blocked_sites"])
            print(f"  {r['task_id']:40s} {sites}")
            print(f"  {'':40s} reason: {reasons}")
        print()

    print(f"PASSED tasks ({len(passed)}):")
    for r in passed:
        tag = "no-site" if r["status"] == "pass_no_website" else ", ".join(r["websites"])
        print(f"  {r['task_id']:40s} [{tag}]")


if __name__ == "__main__":
    main()
