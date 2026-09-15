"""
Unified interface for reading execution logs.

Provides abstraction over the log file structure, allowing evaluators to
access events, AXTrees, and summaries without knowing implementation details.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class LogReader:
    """Reads task execution logs from a run directory."""

    def __init__(self, run_dir: Path, task_id: str):
        """
        Args:
            run_dir: Root directory of the run (e.g., logs/runs/my_run)
            task_id: Task instance ID (e.g., dyna_001)
        """
        self.run_dir = Path(run_dir)
        self.task_id = task_id
        # New layout: run_dir/<task_id>/; legacy: run_dir/
        task_dir = self.run_dir / task_id
        self.task_dir = task_dir if task_dir.is_dir() else self.run_dir
        self._events_cache: Optional[List[Dict[str, Any]]] = None
        self._summary_cache: Optional[Dict[str, Any]] = None

    def get_events(self) -> List[Dict[str, Any]]:
        """Read and return all steps from events.jsonl.

        Returns:
            List of step event dictionaries, one per line from events.jsonl
        """
        if self._events_cache is not None:
            return self._events_cache

        events_file = self.task_dir / "events.jsonl"
        if not events_file.exists():
            # Legacy fallback
            events_file = self.run_dir / f"{self.task_id}.events.jsonl"
        if not events_file.exists():
            return []

        events = []
        with open(events_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

        self._events_cache = events
        return events

    def get_axtree(self, step: int) -> str:
        """Read the AXTree for a specific step.

        Args:
            step: Step number (1-based)

        Returns:
            AXTree content as string, or empty string if file not found
        """
        axtree_file = self.task_dir / f"step_{step:03d}.axtree"
        if axtree_file.exists():
            with open(axtree_file, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def get_summary(self) -> Dict[str, Any]:
        """Read the summary.json for this task.

        Returns:
            Summary dictionary, or empty dict if not found
        """
        if self._summary_cache is not None:
            return self._summary_cache

        summary_file = self.task_dir / "summary.json"
        if not summary_file.exists():
            # Legacy fallback
            summary_file = self.run_dir / f"{self.task_id}.summary.json"
        if not summary_file.exists():
            return {}

        with open(summary_file, "r", encoding="utf-8") as f:
            self._summary_cache = json.load(f)
        return self._summary_cache

    def save_evaluation(self, evaluation: Dict[str, Any]) -> None:
        """Save evaluation results back to summary.json.

        Args:
            evaluation: Evaluation result dictionary to merge into summary
        """
        summary = self.get_summary()
        summary["evaluation"] = evaluation

        summary_file = self.task_dir / "summary.json"
        if not summary_file.exists():
            # Legacy fallback
            summary_file = self.run_dir / f"{self.task_id}.summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Invalidate cache
        self._summary_cache = summary
