"""Step 3a: plan constraint-level override replacements."""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from utils.llm import llm_json_call
from .models import (
    ConstraintResult,
    I0Result,
    OverrideItem,
    OverridePlanResult,
    SwitchingCondition,
)


SCENE_ALIASES = {
    "furniture_shopping": "ecommerce_shopping",
    "forum_search": "health_medical_info",
}

RULE_BASED_COSINE_THRESHOLD = 0.62


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _iter_pairs(replacement_pairs: Dict[str, Any], scene: str) -> List[Dict[str, Any]]:
    scene_pairs = replacement_pairs.get("scenes", {}).get(scene, {}).get("pairs", [])
    cross_pairs = []
    for pair in replacement_pairs.get("cross_scene_pairs", []):
        scenes = pair.get("applicable_scenes", [])
        if not scenes or scene in scenes:
            cross_pairs.append(pair)
    return scene_pairs + cross_pairs


def _text_vector(text: str) -> Counter[str]:
    tokens = _normalize(text).split()
    return Counter(tokens)


def _cosine_similarity(a: str, b: str) -> float:
    va = _text_vector(a)
    vb = _text_vector(b)
    if not va or not vb:
        return 0.0
    dot = sum(va[t] * vb[t] for t in set(va) & set(vb))
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _pair_similarity(constraint_text: str, pair: Dict[str, Any]) -> float:
    a = pair.get("constraint_a", {}).get("text", "")
    b = pair.get("constraint_b", {}).get("text", "")
    return max(_cosine_similarity(constraint_text, a), _cosine_similarity(constraint_text, b))


def _scene_similarity(
    constraints: ConstraintResult,
    scene_pairs: List[Dict[str, Any]],
) -> float:
    if not scene_pairs or not constraints.constraints:
        return 0.0
    best_scores: List[float] = []
    for c in constraints.constraints:
        score = max((_pair_similarity(c.description, p) for p in scene_pairs), default=0.0)
        best_scores.append(score)
    return sum(best_scores) / len(best_scores)


def _choose_candidate_scenes(
    constraints: ConstraintResult,
    replacement_pairs: Dict[str, Any],
    scene: str,
) -> List[str]:
    scene_map = replacement_pairs.get("scenes", {})
    if not scene_map:
        return []

    ranked = sorted(
        scene_map.keys(),
        key=lambda s: _scene_similarity(constraints, scene_map.get(s, {}).get("pairs", [])),
        reverse=True,
    )

    ordered: List[str] = []
    preferred = [scene, SCENE_ALIASES.get(scene, "")]
    for s in preferred + ranked:
        if s and s in scene_map and s not in ordered:
            ordered.append(s)
    return ordered[:3]


def _match_pair_for_constraint(
    constraint_text: str,
    pairs: List[Dict[str, Any]],
) -> Optional[Tuple[str, float]]:
    target = _normalize(constraint_text)
    best: Optional[Tuple[str, float]] = None
    for pair in pairs:
        a = _normalize(pair.get("constraint_a", {}).get("text", ""))
        b = _normalize(pair.get("constraint_b", {}).get("text", ""))
        if target and (target == a or target == b):
            pair_id = pair.get("id")
            if pair_id:
                return (pair_id, 1.0)

        score = _pair_similarity(constraint_text, pair)
        if score >= RULE_BASED_COSINE_THRESHOLD:
            pair_id = pair.get("id")
            if pair_id and (best is None or score > best[1]):
                best = (pair_id, score)
    return best


def _build_candidate_pool(
    constraints: ConstraintResult,
    pairs: List[Dict[str, Any]],
) -> List[Tuple[int, str, float]]:
    candidates: List[Tuple[int, str, float]] = []
    for c in constraints.constraints:
        matched = _match_pair_for_constraint(c.description, pairs)
        if matched:
            pair_id, score = matched
            candidates.append((c.id, pair_id, score))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def _resolve_override_item(
    constraints: ConstraintResult,
    pairs: List[Dict[str, Any]],
    target_constraint_id: int,
    replacement_pair_id: str,
) -> Optional[OverrideItem]:
    id_to_text = {c.id: c.description for c in constraints.constraints}
    target_text = id_to_text.get(target_constraint_id, "")
    if not target_text:
        return None

    pair = next((p for p in pairs if p.get("id") == replacement_pair_id), None)
    if not pair:
        return None

    a = pair.get("constraint_a", {}).get("text", "")
    b = pair.get("constraint_b", {}).get("text", "")
    if target_text == a:
        replacement_text = b
    elif target_text == b:
        replacement_text = a
    else:
        # Keep consistent with previous behavior, but still require a non-empty replacement.
        replacement_text = b or a

    if not replacement_text:
        return None

    return OverrideItem(
        target_constraint_id=target_constraint_id,
        replacement_pair_id=replacement_pair_id,
        target_constraint_text=target_text,
        replacement_constraint_text=replacement_text,
        dimension=pair.get("dimension", ""),
    )


def _llm_select_override(
    constraints: ConstraintResult,
    pairs: List[Dict[str, Any]],
    client: OpenAI,
    model: str,
) -> Optional[Tuple[int, str]]:
    constraint_list = [
        {"id": c.id, "description": c.description}
        for c in constraints.constraints
    ]
    pair_list = []
    for p in pairs[:80]:
        pair_list.append({
            "id": p.get("id", ""),
            "a": p.get("constraint_a", {}).get("text", ""),
            "b": p.get("constraint_b", {}).get("text", ""),
        })

    system = (
        "You are selecting one preference-override replacement for a GUI task. "
        "Pick at most one pair that naturally replaces exactly one constraint."
    )
    user = (
        "Select one replacement plan.\n"
        "Rules:\n"
        "- Choose at most one constraint\n"
        "- The selected pair must match that constraint semantically\n"
        "- If no safe choice exists, return enable_override=false\n\n"
        f"Constraints:\n{json.dumps(constraint_list, ensure_ascii=False, indent=2)}\n\n"
        f"Replacement pairs:\n{json.dumps(pair_list, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON:\n"
        "{\n"
        "  \"enable_override\": true|false,\n"
        "  \"target_constraint_id\": <int or null>,\n"
        "  \"replacement_pair_id\": \"...\" or null\n"
        "}"
    )

    payload = llm_json_call(
        client=client,
        system=system,
        user=user,
        model=model,
        max_tokens=1000,
    )

    if not payload.get("enable_override", False):
        return None

    target_id = payload.get("target_constraint_id")
    pair_id = payload.get("replacement_pair_id")
    if isinstance(target_id, int) and isinstance(pair_id, str) and pair_id:
        return (target_id, pair_id)
    return None


def load_switching_conditions(
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the switching condition template library.

    Defaults to ``data/pipeline/switching_conditions.yaml`` at repo root. Returns
    an empty dict if the file is absent so callers can degrade to
    unconditional mode.
    """
    import yaml

    if path is None:
        # repo root / data / switching_conditions.yaml — relative to this file:
        # src/pipeline_lib/step3a_override_plan.py → ../../../data/...
        path = str(
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "switching_conditions.yaml"
        )
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def select_switching_condition(
    item: OverrideItem,
    sc_data: Dict[str, Any],
    scene: str,
) -> Optional[SwitchingCondition]:
    """Pick a switching condition template for an override item.

    Rule-based gate per docs/conditional_override_design.md §4:
    - reject if the pair dimension is in any excluded_pair_dims group
      (self-reference, identity attribute, metadata)
    - take the first template whose ``applicable_pair_dims`` and
      ``applicable_scenes`` both include the item's dimension and the
      task scene
    - render ``text_template`` with the resolved options/threshold/unit
      to produce ``condition_text`` (the predicate evaluator will
      consume)

    Returns None when no template applies — caller is responsible for
    deciding whether to fall back to unconditional or skip the task.
    """
    if not sc_data:
        return None

    excluded: set = set()
    for group in sc_data.get("excluded_pair_dims", []):
        for d in group.get("dims", []):
            excluded.add(d)
    if item.dimension in excluded:
        return None

    default_option = item.target_constraint_text or ""
    alt_option = item.replacement_constraint_text or ""

    for template in sc_data.get("templates", []):
        applicable_dims = template.get("applicable_pair_dims") or []
        if item.dimension not in applicable_dims:
            continue
        applicable_scenes = template.get("applicable_scenes") or []
        if scene and applicable_scenes and scene not in applicable_scenes:
            continue

        threshold = template.get("default_threshold")
        unit = template.get("unit", "")
        return SwitchingCondition(
            dimension=template.get("dim", ""),
            operator=template.get("operator", ""),
            threshold=float(threshold) if isinstance(threshold, (int, float)) else None,
            unit=str(unit) if unit else "",
            default_option=default_option,
            alt_option=alt_option,
            condition_text="",   # filled by Step 3b LLM
        )
    return None


def build_override_plan(
    constraints: ConstraintResult,
    i0: I0Result,
    replacement_pairs: Dict[str, Any],
    scene: str,
    strategy: str = "rule_based",
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o",
    override_mode: str = "unconditional",
    switching_conditions: Optional[Dict[str, Any]] = None,
) -> OverridePlanResult:
    """Build constraint-only override plan (no revision placement decisions).

    When ``override_mode='conditional'`` and a switching condition
    library is supplied, each resolved override item gets a matching
    ``SwitchingCondition`` attached. If no template applies to the
    chosen pair (excluded dim, no matching scene/dim combo), the plan
    falls back to ``override_mode='unconditional'`` so the task is
    still useful. Caller can detect this via the returned
    ``override_mode`` field.
    """
    del i0  # Step 3a focuses on constraint-level planning.

    candidate_scenes = _choose_candidate_scenes(constraints, replacement_pairs, scene)
    pairs: List[Dict[str, Any]] = []
    for candidate_scene in candidate_scenes:
        pairs.extend(_iter_pairs(replacement_pairs, candidate_scene))

    # Keep one copy per pair id to avoid duplicate votes across scene pools.
    dedup: Dict[str, Dict[str, Any]] = {}
    for p in pairs:
        pair_id = p.get("id")
        if isinstance(pair_id, str) and pair_id and pair_id not in dedup:
            dedup[pair_id] = p
    pairs = list(dedup.values())

    if not pairs:
        return OverridePlanResult(enable_override=False)

    candidates = _build_candidate_pool(constraints, pairs)

    def _finalize(resolved: OverrideItem) -> OverridePlanResult:
        """Attach switching_condition (if requested) and return the plan.

        When ``override_mode='conditional'`` but no template matches the
        pair, falls back to ``override_mode='unconditional'`` rather
        than rejecting the override outright.
        """
        if override_mode != "conditional":
            return OverridePlanResult(
                enable_override=True,
                overrides=[resolved],
                override_mode="unconditional",
            )
        sc = (
            select_switching_condition(resolved, switching_conditions, scene)
            if switching_conditions
            else None
        )
        if sc is None:
            return OverridePlanResult(
                enable_override=True,
                overrides=[resolved],
                override_mode="unconditional",
            )
        resolved.switching_condition = sc
        return OverridePlanResult(
            enable_override=True,
            overrides=[resolved],
            override_mode="conditional",
        )

    if strategy == "rule_based":
        if not candidates:
            return OverridePlanResult(enable_override=False)
        cid, pid, _score = candidates[0]
        resolved = _resolve_override_item(constraints, pairs, cid, pid)
        if not resolved:
            return OverridePlanResult(enable_override=False)
        return _finalize(resolved)

    if strategy == "hybrid":
        if candidates:
            cid, pid, _score = candidates[0]
            resolved = _resolve_override_item(constraints, pairs, cid, pid)
            if not resolved:
                return OverridePlanResult(enable_override=False)
            return _finalize(resolved)
        if client is None:
            return OverridePlanResult(enable_override=False)
        picked = _llm_select_override(constraints, pairs, client, model)
        if not picked:
            return OverridePlanResult(enable_override=False)
        resolved = _resolve_override_item(constraints, pairs, picked[0], picked[1])
        if not resolved:
            return OverridePlanResult(enable_override=False)
        return _finalize(resolved)

    if strategy == "llm_only":
        if client is None:
            return OverridePlanResult(enable_override=False)
        picked = _llm_select_override(constraints, pairs, client, model)
        if not picked:
            return OverridePlanResult(enable_override=False)
        resolved = _resolve_override_item(constraints, pairs, picked[0], picked[1])
        if not resolved:
            return OverridePlanResult(enable_override=False)
        return _finalize(resolved)

    raise ValueError(f"Unknown override planning strategy: {strategy}")
