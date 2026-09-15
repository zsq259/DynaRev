import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_url(url: str) -> str:
    """Normalize URL for stable keying across scripts and runs."""
    raw = (url or "").strip()
    if not raw:
        return ""

    if "://" not in raw:
        raw = f"https://{raw}"

    p = urlparse(raw)
    scheme = p.scheme.lower() if p.scheme else "https"
    netloc = p.netloc.lower()
    path = p.path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", "", ""))


class SiteHealthStore:
    """Persistent site inventory + health + event log store."""

    def __init__(self, base_dir: str | Path = "data/websites") -> None:
        self.base_dir = Path(base_dir)
        self.inventory_path = self.base_dir / "site_inventory.json"
        self.health_path = self.base_dir / "site_health.json"
        self.events_path = self.base_dir / "site_health_log.jsonl"
        self.feasible_path = self.base_dir / "site_feasible.json"

        self.inventory: Dict[str, Any] = _read_json(
            self.inventory_path,
            {"version": 1, "updated_at": "", "sites": {}},
        )
        self.health: Dict[str, Any] = _read_json(
            self.health_path,
            {"version": 1, "updated_at": "", "sites": {}},
        )

    def _append_event(self, event: Dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def save(self) -> None:
        now = _utc_now_iso()
        self.inventory["updated_at"] = now
        self.health["updated_at"] = now
        _write_json(self.inventory_path, self.inventory)
        _write_json(self.health_path, self.health)

    def register_urls(self, urls: List[str], source: str = "", task_id: str = "") -> List[str]:
        now = _utc_now_iso()
        out: List[str] = []
        for u in urls:
            nu = normalize_url(u)
            if not nu:
                continue
            out.append(nu)

            inv = self.inventory["sites"].setdefault(
                nu,
                {
                    "url": nu,
                    "first_seen_at": now,
                    "last_seen_at": now,
                    "sources": [],
                    "task_ids": [],
                },
            )
            inv["last_seen_at"] = now
            if source and source not in inv["sources"]:
                inv["sources"].append(source)
            if task_id and task_id not in inv["task_ids"]:
                inv["task_ids"].append(task_id)

            self.health["sites"].setdefault(
                nu,
                {
                    "url": nu,
                    "last_updated": now,
                    "runtime_accessible": None,
                    "runtime_stable": None,
                    "runtime_success_count": 0,
                    "runtime_attempts": 0,
                    "homepage_accessible": None,
                    "homepage_block_reason": "",
                    "run_access_ok_count": 0,
                    "run_access_fail_count": 0,
                    "last_run_status": "",
                    "last_error": "",
                    "runner_compatible": None,
                    "runner_incompatibility_reason": "",
                    "runner_checked_at": "",
                },
            )
            row = self.health["sites"][nu]
            row.setdefault("runner_compatible", None)
            row.setdefault("runner_incompatibility_reason", "")
            row.setdefault("runner_checked_at", "")
        return out

    def get_health(self, url: str) -> Optional[Dict[str, Any]]:
        return self.health["sites"].get(normalize_url(url))

    def is_runner_incompatible(self, url: str) -> bool:
        row = self.get_health(url)
        return bool(row and row.get("runner_compatible") is False)

    def mark_runner_incompatible(
        self,
        url: str,
        reason: str,
        source: str = "manual_review",
        task_id: str = "",
    ) -> None:
        """Persist a runner-specific exclusion without conflating it with HTTP access."""
        now = _utc_now_iso()
        nu = normalize_url(url)
        if not nu:
            return
        self.register_urls([nu], source=source, task_id=task_id)
        row = self.health["sites"][nu]
        row["runner_compatible"] = False
        row["runner_incompatibility_reason"] = reason[:1000]
        row["runner_checked_at"] = now
        row["last_updated"] = now
        row["last_error"] = reason[:500]
        self._append_event(
            {
                "timestamp": now,
                "type": "runner_compatibility_update",
                "url": nu,
                "source": source,
                "task_id": task_id,
                "payload": {
                    "runner_compatible": False,
                    "reason": reason[:1000],
                },
            }
        )

    def mark_runner_compatible(
        self,
        url: str,
        source: str = "runner",
        task_id: str = "",
    ) -> None:
        """Record that the full benchmark runner completed browser initialization."""
        now = _utc_now_iso()
        nu = normalize_url(url)
        if not nu:
            return
        self.register_urls([nu], source=source, task_id=task_id)
        row = self.health["sites"][nu]
        row["runner_compatible"] = True
        row["runner_incompatibility_reason"] = ""
        row["runner_checked_at"] = now
        row["last_updated"] = now

    def update_from_probe(
        self,
        url: str,
        probe_payload: Dict[str, Any],
        source: str = "probe",
    ) -> None:
        now = _utc_now_iso()
        nu = normalize_url(url)
        if not nu:
            return
        self.register_urls([nu], source=source)

        row = self.health["sites"][nu]
        attempts = int(probe_payload.get("runtime_attempts", 0) or 0)
        success_count = int(probe_payload.get("runtime_success_count", 0) or 0)
        row["runtime_accessible"] = bool(probe_payload.get("accessible", False))
        row["runtime_stable"] = bool(probe_payload.get("stable", False))
        row["runtime_attempts"] = attempts
        row["runtime_success_count"] = success_count
        row["homepage_accessible"] = probe_payload.get("homepage_accessible")
        row["homepage_block_reason"] = str(probe_payload.get("homepage_block_reason", "") or "")
        row["last_updated"] = now
        reasons = probe_payload.get("runtime_reasons", []) or []
        if reasons:
            row["last_error"] = str(reasons[0])[:500]

        self._append_event(
            {
                "timestamp": now,
                "type": "probe_update",
                "url": nu,
                "source": source,
                "payload": {
                    "accessible": row["runtime_accessible"],
                    "stable": row["runtime_stable"],
                    "runtime_success_count": success_count,
                    "runtime_attempts": attempts,
                    "runtime_reasons": probe_payload.get("runtime_reasons", []) or [],
                    "runtime_final_urls": probe_payload.get("runtime_final_urls", []) or [],
                    "homepage_accessible": row["homepage_accessible"],
                    "homepage_block_reason": row["homepage_block_reason"],
                },
            }
        )

    def update_from_run_result(self, run_result: Dict[str, Any], source: str = "runner") -> None:
        now = _utc_now_iso()
        url = normalize_url(str(run_result.get("website", "") or ""))
        if not url:
            return

        task_id = str(run_result.get("instance_id", "") or "")
        self.register_urls([url], source=source, task_id=task_id)

        row = self.health["sites"][url]
        stopped_reason = str(run_result.get("stopped_reason", "") or "")
        completion_status = str(run_result.get("completion_status", "") or stopped_reason)
        error_category = str(run_result.get("error_category", "") or "")
        steps = int(run_result.get("steps", 0) or 0)

        access_ok = bool(
            (steps > 0)
            or (stopped_reason in {"env_done", "stuck", "max_steps"})
        ) and error_category != "environment"

        if access_ok:
            row["run_access_ok_count"] = int(row.get("run_access_ok_count", 0) or 0) + 1
            if row.get("runtime_accessible") is None:
                row["runtime_accessible"] = True
            if row.get("runner_compatible") is None:
                row["runner_compatible"] = True
                row["runner_checked_at"] = now
        else:
            row["run_access_fail_count"] = int(row.get("run_access_fail_count", 0) or 0) + 1
            if error_category == "environment":
                row["runtime_accessible"] = False

        row["last_run_status"] = completion_status
        row["last_updated"] = now

        top_error = str(run_result.get("error_message", "") or "")
        if top_error:
            row["last_error"] = top_error[:500]

        self._append_event(
            {
                "timestamp": now,
                "type": "run_update",
                "url": url,
                "source": source,
                "task_id": task_id,
                "payload": {
                    "stopped_reason": stopped_reason,
                    "completion_status": completion_status,
                    "error_category": error_category,
                    "steps": steps,
                    "done_called": bool(run_result.get("done_called", False)),
                    "error_message": top_error,
                },
            }
        )

    def export_feasible(self) -> Dict[str, Any]:
        now = _utc_now_iso()
        feasible: List[str] = []
        for url, row in self.health["sites"].items():
            if row.get("runner_compatible") is False:
                continue
            rt_accessible = row.get("runtime_accessible") is True
            rt_stable = row.get("runtime_stable") is True
            run_ok = int(row.get("run_access_ok_count", 0) or 0)
            run_fail = int(row.get("run_access_fail_count", 0) or 0)
            has_positive_run_signal = run_ok > 0 and run_ok >= run_fail
            if (rt_accessible and (rt_stable or has_positive_run_signal)) or has_positive_run_signal:
                feasible.append(url)

        payload = {
            "version": 1,
            "generated_at": now,
            "count": len(feasible),
            "urls": sorted(feasible),
        }
        _write_json(self.feasible_path, payload)
        return payload
