#!/usr/bin/env python3
"""Build constraint replacement pair library from pipeline reports.

Reads Step 1 constraints from report.json files, classifies them as CORE vs
PREFERENCE via LLM, then pairs PREFERENCE constraints within each scene to
create a replacement pair library for Override injection.

Usage:
    conda run -n dyna python -m src.data_prep.build_replacement_pairs
"""

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List

from openai import OpenAI

from ..utils.client import get_client
from ..utils.llm import llm_json_call
from ..pipeline_lib.prompts import (
    CLASSIFY_SYSTEM,
    CLASSIFY_USER,
    PAIRING_SYSTEM,
    PAIRING_USER,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

REPORT_PATHS = [
    PROJECT_ROOT / "logs/pipeline/full_172_v1/report.json",
    PROJECT_ROOT / "logs/pipeline/ecommerce_v1/report.json",
    PROJECT_ROOT / "logs/pipeline/m2w2_seed_20260421/report.json",
    PROJECT_ROOT / "logs/pipeline/m2w2_batch2_20260428/report.json",
]
SCENE_CSVS = [
    PROJECT_ROOT / "data/sources/om2w_scene_labels_v2.csv",
    PROJECT_ROOT / "data/sources/m2w2_scene_labels_v1.csv",
]
OUTPUT_PATH = PROJECT_ROOT / "data/pipeline/replacement_pairs.json"
CLASSIFY_CACHE_PATH = PROJECT_ROOT / "data/pipeline/replacement_pairs.classify.json"
PAIR_CACHE_PATH = PROJECT_ROOT / "data/pipeline/replacement_pairs.pair.json"

MODEL = "deepseek-v4-flash"


# ── Step 1: Extract constraints grouped by scene ─────────────────────────────


def load_scene_map() -> Dict[str, str]:
    """Load task_id -> primary_scene mapping, merged from all SCENE_CSVS."""
    scenes: Dict[str, str] = {}
    for path in SCENE_CSVS:
        if not path.exists():
            log.warning("Scene CSV not found: %s", path)
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                scenes[row["task_id"]] = row["primary_scene"]
    return scenes


def extract_constraints_by_scene(
    scene_map: Dict[str, str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract all constraints from reports, grouped by scene.

    Each entry: {task_id, constraint_id, text}
    Deduplicates by task_id across reports.
    """
    seen_tasks = set()
    by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for path in REPORT_PATHS:
        if not path.exists():
            log.warning("Report not found: %s", path)
            continue
        data = json.loads(path.read_text())
        for task in data:
            tid = task["task_id"]
            if tid in seen_tasks:
                continue
            seen_tasks.add(tid)
            scene = scene_map.get(tid, "unknown")
            for c in task.get("step1_constraints", []):
                by_scene[scene].append(
                    {
                        "task_id": tid,
                        "constraint_id": f"{tid[:8]}_{c['id']}",
                        "text": c["description"],
                    }
                )

    return dict(by_scene)


# ── Step 2: LLM classification (CORE vs PREFERENCE) ─────────────────────────


CLASSIFY_CHUNK_SIZE = 40  # max constraints per LLM call


def _load_json_or_empty(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write via temp file + rename so a crash mid-write can't corrupt the cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def classify_constraints(
    client: OpenAI,
    scene: str,
    constraints: List[Dict[str, Any]],
    cache: Dict[str, str],
    save_cache: Callable[[], None],
) -> Dict[str, str]:
    """Classify constraints for a scene. Returns {constraint_id: label}.

    Splits into chunks; reads/writes to shared cache so failures don't lose
    completed chunks. cache is mutated in place; save_cache() persists it.
    """
    labels = {cid: cache[cid] for c in constraints if (cid := c["constraint_id"]) in cache}

    todo = [c for c in constraints if c["constraint_id"] not in cache]
    for i in range(0, len(todo), CLASSIFY_CHUNK_SIZE):
        chunk = todo[i : i + CLASSIFY_CHUNK_SIZE]
        lines = [f'- id="{c["constraint_id"]}": {c["text"]}' for c in chunk]
        constraint_text = "\n".join(lines)

        result = llm_json_call(
            client,
            system=CLASSIFY_SYSTEM,
            user=CLASSIFY_USER.format(scene=scene, constraints=constraint_text),
            model=MODEL,
            max_tokens=4096,
        )

        for item in result.get("classifications", []):
            labels[item["id"]] = item["label"]
            cache[item["id"]] = item["label"]
        save_cache()
    return labels


# ── Step 3: LLM pairing (PREFERENCE only) ────────────────────────────────────


PAIR_CHUNK_SIZE = 15  # smaller for deepseek-v4-flash; pair prompt is heavier


def _chunk_key(chunk: List[Dict[str, Any]]) -> str:
    """Content-addressed key so cache survives chunk-size or input-order changes."""
    import hashlib
    ids = sorted(c["constraint_id"] for c in chunk)
    return hashlib.sha1("\n".join(ids).encode()).hexdigest()[:12]


def pair_preferences(
    client: OpenAI,
    scene: str,
    pref_constraints: List[Dict[str, Any]],
    scene_cache: Dict[str, List[Dict[str, Any]]],
    save_cache: Callable[[], None],
) -> List[Dict[str, Any]]:
    """Generate replacement pairs from PREFERENCE constraints for a scene.

    scene_cache: {chunk_key: [pair, ...]} for this scene. Mutated + persisted
    after each chunk via save_cache().
    """
    if not pref_constraints:
        return []

    all_pairs: List[Dict[str, Any]] = []
    for i in range(0, len(pref_constraints), PAIR_CHUNK_SIZE):
        chunk = pref_constraints[i : i + PAIR_CHUNK_SIZE]
        key = _chunk_key(chunk)
        if key in scene_cache:
            all_pairs.extend(scene_cache[key])
            continue

        lines = [
            f'- id="{c["constraint_id"]}" (task={c["task_id"][:12]}): {c["text"]}'
            for c in chunk
        ]
        constraint_text = "\n".join(lines)

        result = llm_json_call(
            client,
            system=PAIRING_SYSTEM,
            user=PAIRING_USER.format(scene=scene, constraints=constraint_text),
            model=MODEL,
            max_tokens=16384,
        )
        pairs = result.get("pairs", [])
        scene_cache[key] = pairs
        save_cache()
        all_pairs.extend(pairs)

    return all_pairs


# ── Step 4: Cross-scene universal pairs ───────────────────────────────────────

UNIVERSAL_DIMENSIONS = {"price_range", "rating", "sort_order", "price", "sort"}


def build_cross_scene_pairs(
    scene_pairs: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Extract universal-dimension pairs and tag applicable scenes."""
    # Collect all unique (dimension, a_text, b_text) across scenes
    seen = {}  # (dim, a, b) -> set of scenes
    for scene, pairs in scene_pairs.items():
        for p in pairs:
            dim = p.get("dimension", "").lower().replace(" ", "_")
            # Normalize to canonical universal dimensions
            canonical = None
            for ud in UNIVERSAL_DIMENSIONS:
                if ud in dim:
                    canonical = ud
                    break
            if canonical is None:
                continue
            a = p["constraint_a"]["text"]
            b = p["constraint_b"]["text"]
            key = (canonical, a, b)
            if key not in seen:
                seen[key] = {"pair": p, "scenes": set()}
            seen[key]["scenes"].add(scene)

    cross = []
    for i, ((dim, a, b), info) in enumerate(seen.items(), 1):
        p = info["pair"]
        cross.append(
            {
                "id": f"xscene_{i:03d}",
                "dimension": dim,
                "constraint_a": {"text": a},
                "constraint_b": {"text": b},
                "symmetric": p.get("symmetric", True),
                "applicable_scenes": sorted(info["scenes"]),
            }
        )
    return cross


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    client = get_client(MODEL)
    scene_map = load_scene_map()

    # Step 1: Extract
    log.info("Extracting constraints from reports...")
    by_scene = extract_constraints_by_scene(scene_map)
    for scene, constraints in sorted(by_scene.items()):
        log.info("  %-35s %3d constraints", scene, len(constraints))

    # Step 2: Classify (with persistent cache, content-addressed by constraint_id)
    classify_cache: Dict[str, str] = _load_json_or_empty(CLASSIFY_CACHE_PATH)
    if classify_cache:
        log.info("Loaded %d cached classify labels from %s",
                 len(classify_cache), CLASSIFY_CACHE_PATH.name)
    save_classify = lambda: _atomic_write_json(CLASSIFY_CACHE_PATH, classify_cache)

    log.info("Classifying constraints (CORE vs PREFERENCE)...")
    all_labels: Dict[str, str] = {}
    scene_stats = {}
    for scene in sorted(by_scene):
        constraints = by_scene[scene]
        n_cached = sum(1 for c in constraints if c["constraint_id"] in classify_cache)
        log.info("  Classifying %s (%d constraints, %d cached)...",
                 scene, len(constraints), n_cached)
        labels = classify_constraints(client, scene, constraints, classify_cache, save_classify)
        all_labels.update(labels)

        core = sum(1 for v in labels.values() if v == "CORE")
        pref = sum(1 for v in labels.values() if v == "PREFERENCE")
        scene_stats[scene] = {"core": core, "preference": pref}
        log.info("    CORE=%d  PREFERENCE=%d", core, pref)

    # Step 3: Pair PREFERENCE constraints per scene (with persistent cache)
    pair_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = _load_json_or_empty(PAIR_CACHE_PATH)
    if pair_cache:
        n_chunks = sum(len(v) for v in pair_cache.values())
        log.info("Loaded %d cached pair chunks across %d scenes from %s",
                 n_chunks, len(pair_cache), PAIR_CACHE_PATH.name)
    save_pair = lambda: _atomic_write_json(PAIR_CACHE_PATH, pair_cache)

    log.info("Generating replacement pairs per scene...")
    scene_pairs: Dict[str, List[Dict[str, Any]]] = {}
    for scene in sorted(by_scene):
        pref_constraints = [
            c
            for c in by_scene[scene]
            if all_labels.get(c["constraint_id"]) == "PREFERENCE"
        ]
        if not pref_constraints:
            log.info("  %-35s  0 PREFERENCE constraints, skipping", scene)
            scene_pairs[scene] = []
            continue

        scene_cache = pair_cache.setdefault(scene, {})
        n_cached_chunks = len(scene_cache)
        log.info(
            "  Pairing %s (%d PREFERENCE constraints, %d cached chunks)...",
            scene,
            len(pref_constraints),
            n_cached_chunks,
        )
        pairs = pair_preferences(client, scene, pref_constraints, scene_cache, save_pair)
        scene_pairs[scene] = pairs

        # Stats
        n_generated = sum(
            1
            for p in pairs
            if p.get("constraint_a", {}).get("generated")
            or p.get("constraint_b", {}).get("generated")
        )
        n_existing = len(pairs) - n_generated
        dims = set(p.get("dimension", "?") for p in pairs)
        log.info(
            "    %d pairs (%d existing-value, %d with-generated)  dims=%s",
            len(pairs),
            n_existing,
            n_generated,
            dims,
        )

    # Step 4: Cross-scene universal pairs
    log.info("Building cross-scene universal pairs...")
    cross_pairs = build_cross_scene_pairs(scene_pairs)
    log.info("  %d cross-scene pairs", len(cross_pairs))

    # Step 5: Assign IDs and write output
    output = {"version": "v1", "scenes": {}, "cross_scene_pairs": cross_pairs}
    for scene in sorted(scene_pairs):
        abbrev = "".join(w[0] for w in scene.split("_"))
        pairs_with_ids = []
        for i, p in enumerate(scene_pairs[scene], 1):
            p["id"] = f"{abbrev}_{i:03d}"
            pairs_with_ids.append(p)
        output["scenes"][scene] = {"pairs": pairs_with_ids}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    log.info("Written to %s", OUTPUT_PATH)

    # Summary
    total_pairs = sum(len(v["pairs"]) for v in output["scenes"].values())
    log.info(
        "DONE: %d scene pairs + %d cross-scene pairs across %d scenes",
        total_pairs,
        len(cross_pairs),
        len(output["scenes"]),
    )


if __name__ == "__main__":
    main()
