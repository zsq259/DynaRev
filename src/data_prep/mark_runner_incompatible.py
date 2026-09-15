"""Mark a site as incompatible with the full benchmark browser runner."""

from __future__ import annotations

import argparse

from ..utils.site_health import SiteHealthStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--task-id", default="")
    parser.add_argument("--site-health-dir", default="data/websites")
    args = parser.parse_args()

    store = SiteHealthStore(args.site_health_dir)
    store.mark_runner_incompatible(
        args.url,
        reason=args.reason,
        source="runner_incompatibility_review",
        task_id=args.task_id,
    )
    store.save()
    feasible = store.export_feasible()
    print(f"marked runner-incompatible: {args.url}")
    print(f"feasible sites: {feasible['count']}")


if __name__ == "__main__":
    main()
