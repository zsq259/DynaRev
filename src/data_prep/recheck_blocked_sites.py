"""Re-check previously blocked websites using the updated fetch_homepage with stealth measures."""

import json
import time
from pathlib import Path

from ..pipeline_lib.homepage import fetch_homepage, probe_runtime_access
from ..utils.site_health import SiteHealthStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEBSITE_CHECK = PROJECT_ROOT / "data" / "sources" / "online-mind2web" / "website_check.json"


def main():
    with open(WEBSITE_CHECK) as f:
        checks = json.load(f)

    # Collect previously blocked websites
    blocked = {url: info for url, info in checks.items() if not info["accessible"]}
    print(f"Previously blocked websites: {len(blocked)}")
    print("=" * 70)

    results = {}
    site_health = SiteHealthStore()
    for i, (url, old_info) in enumerate(sorted(blocked.items()), 1):
        print(f"\n[{i}/{len(blocked)}] Testing: {url}")
        print(f"  Old reason: {old_info['block_reason'][:80]}")

        start = time.time()
        result = fetch_homepage(url, timeout_ms=45000, max_retries=2)
        runtime_probe = probe_runtime_access(url, attempts=3, timeout_ms=30000)
        elapsed = time.time() - start

        status = "ACCESSIBLE" if runtime_probe.accessible else "STILL BLOCKED"
        print(f"  Result: {status} ({elapsed:.1f}s)")
        if not runtime_probe.accessible:
            reason = runtime_probe.reasons[0] if runtime_probe.reasons else result.block_reason
            print(f"  New reason: {reason[:80]}")

        results[url] = {
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
                    else result.block_reason
                )
            ),
            "old_block_reason": old_info["block_reason"],
        }
        site_health.register_urls([url], source="blocked_recheck")
        site_health.update_from_probe(url, results[url], source="blocked_recheck")

    # Summary
    now_accessible = [url for url, r in results.items() if r["accessible"]]
    still_blocked = [url for url, r in results.items() if not r["accessible"]]

    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(now_accessible)} now accessible, {len(still_blocked)} still blocked")

    if now_accessible:
        print(f"\nNewly accessible ({len(now_accessible)}):")
        for url in sorted(now_accessible):
            print(f"  + {url}")

    if still_blocked:
        print(f"\nStill blocked ({len(still_blocked)}):")
        for url in sorted(still_blocked):
            reason = results[url]["block_reason"][:60]
            print(f"  - {url}  [{reason}]")

    # Save results
    out_path = PROJECT_ROOT / "data" / "sources" / "online-mind2web" / "website_recheck.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    site_health.save()
    site_health.export_feasible()


if __name__ == "__main__":
    main()
