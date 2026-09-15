#!/usr/bin/env python3
"""Label m2w2 tasks with one of the 9 canonical primary_scene values.

Reads m2w2 pipeline reports, classifies each task into a primary_scene via
deepseek-v4-flash, and writes a CSV in the same format as om2w_scene_labels_v2.csv
so build_replacement_pairs.py can consume both side-by-side.

Usage:
    source 1.sh && conda run -n dyna python -m src.data_prep.label_m2w2_scenes
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from ..utils.client import get_client
from ..utils.llm import llm_json_call

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

REPORT_PATHS = [
    PROJECT_ROOT / "logs/pipeline/m2w2_seed_20260421/report.json",
    PROJECT_ROOT / "logs/pipeline/m2w2_batch2_20260428/report.json",
]
OUTPUT_CSV = PROJECT_ROOT / "data/sources/m2w2_scene_labels_v1.csv"

MODEL = "deepseek-v4-flash"

SCENES = [
    "automotive_vehicle",
    "housing_real_estate",
    "travel_booking",
    "health_medical_info",
    "logistics_admin",
    "ecommerce_shopping",
    "local_services_directory",
    "public_service_government",
    "media_information_content",
]

CLASSIFY_SCENE_SYSTEM = (
    "You are classifying web tasks into one of nine fixed primary_scene labels. "
    "Pick exactly one label per task. Output strict JSON only."
)

CLASSIFY_SCENE_USER = """\
Available primary_scene labels (pick exactly one):
- automotive_vehicle: cars, motorcycles, dealership, vehicle specs/comparison
- housing_real_estate: home buying/renting, listings, mortgages, neighborhood
- travel_booking: flights, hotels, trains, car rental, trip planning
- health_medical_info: clinical info, symptoms, conditions, drugs, providers
- logistics_admin: shipping, tracking, business filings, account/admin tools
- ecommerce_shopping: products, retail, cart, deals, comparison shopping
- local_services_directory: yellow-pages style local pros (plumber, salon, vet)
- public_service_government: government info/services, official agency content
- media_information_content: news, articles, blogs, podcasts, videos, social, research papers, encyclopedia, reference content

Classify each of the following tasks. Return JSON:
{{"classifications": [{{"task_id": "...", "primary_scene": "..."}}]}}

Tasks:
{tasks}
"""


def load_tasks() -> List[Dict[str, Any]]:
    """Load m2w2 tasks (task_id, website, instruction) from reports, dedup by task_id."""
    seen = set()
    tasks: List[Dict[str, Any]] = []
    for path in REPORT_PATHS:
        if not path.exists():
            log.warning("Report not found: %s", path)
            continue
        data = json.loads(path.read_text())
        for t in data:
            tid = t["task_id"]
            if tid in seen:
                continue
            seen.add(tid)
            tasks.append(
                {
                    "task_id": tid,
                    "website": t.get("website", "") or "",
                    "instruction": (t.get("original_instruction", "") or "").strip(),
                }
            )
    return tasks


CHUNK_SIZE = 10


def classify_chunk(client, chunk: List[Dict[str, Any]]) -> Dict[str, str]:
    """Send a chunk to the LLM and return {task_id: primary_scene}."""
    lines = []
    for t in chunk:
        instr = t["instruction"].replace("\n", " ")
        if len(instr) > 600:
            instr = instr[:600] + "..."
        site = t["website"] or "(no website)"
        lines.append(f'- task_id="{t["task_id"]}" site={site}\n  instruction: {instr}')
    user = CLASSIFY_SCENE_USER.format(tasks="\n".join(lines))

    result = llm_json_call(
        client,
        system=CLASSIFY_SCENE_SYSTEM,
        user=user,
        model=MODEL,
        max_tokens=4096,
    )

    out: Dict[str, str] = {}
    for item in result.get("classifications", []):
        scene = item.get("primary_scene", "")
        if scene not in SCENES:
            log.warning("Unknown scene %r for %s; skipping", scene, item.get("task_id"))
            continue
        out[item["task_id"]] = scene
    return out


def main():
    client = get_client(MODEL)
    tasks = load_tasks()
    log.info("Loaded %d unique m2w2 tasks", len(tasks))

    labels: Dict[str, str] = {}
    for i in range(0, len(tasks), CHUNK_SIZE):
        chunk = tasks[i : i + CHUNK_SIZE]
        log.info("Classifying chunk %d-%d (%d tasks)...", i, i + len(chunk) - 1, len(chunk))
        labels.update(classify_chunk(client, chunk))

    missing = [t["task_id"] for t in tasks if t["task_id"] not in labels]
    if missing:
        log.warning("Missing labels for %d tasks: %s", len(missing), missing[:10])

    # Write CSV in the same shape as om2w_scene_labels_v2.csv
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["task_id", "website", "confirmed_task", "level",
             "reference_length", "primary_scene", "matched_rules"]
        )
        for t in tasks:
            scene = labels.get(t["task_id"], "")
            w.writerow([t["task_id"], t["website"], t["instruction"], "", "", scene, ""])

    by_scene: Dict[str, int] = {}
    for s in labels.values():
        by_scene[s] = by_scene.get(s, 0) + 1
    log.info("Per-scene counts:")
    for s in SCENES:
        log.info("  %-30s %d", s, by_scene.get(s, 0))
    log.info("Written %d rows to %s", len(tasks), OUTPUT_CSV)


if __name__ == "__main__":
    main()
