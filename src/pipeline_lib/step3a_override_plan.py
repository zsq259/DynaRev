"""Step 3a: build cosine candidate pairs. Selection happens in Step 3b."""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .models import ConstraintResult, I0Result, OverrideItem, OverridePlanResult


SCENE_ALIASES = {
    "furniture_shopping": "ecommerce_shopping",
    "forum_search": "health_medical_info",
}

RULE_BASED_COSINE_THRESHOLD = 0.62
CANDIDATE_CAP = 20


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


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


def _iter_all_pairs(replacement_pairs: Dict[str, Any], scene: str) -> List[Dict[str, Any]]:
    """All pairs: labeled scene first, then other scenes, then cross-scene.

    Scene is sort order only. Nothing is dropped because of the scene label.
    """
    scene_map = replacement_pairs.get("scenes", {}) or {}
    preferred = [scene, SCENE_ALIASES.get(scene, "")]
    ordered_scenes: List[str] = []
    for s in preferred:
        if s and s in scene_map and s not in ordered_scenes:
            ordered_scenes.append(s)
    for s in scene_map:
        if s not in ordered_scenes:
            ordered_scenes.append(s)

    pairs: List[Dict[str, Any]] = []
    seen: set = set()
    for s in ordered_scenes:
        for pair in scene_map.get(s, {}).get("pairs", []) or []:
            pair_id = pair.get("id")
            if isinstance(pair_id, str) and pair_id and pair_id not in seen:
                seen.add(pair_id)
                pairs.append(pair)
    for pair in replacement_pairs.get("cross_scene_pairs", []) or []:
        pair_id = pair.get("id")
        if isinstance(pair_id, str) and pair_id and pair_id not in seen:
            seen.add(pair_id)
            pairs.append(pair)
    return pairs


def _matches_for_constraint(
    constraint_text: str,
    pairs: List[Dict[str, Any]],
) -> List[Tuple[str, float]]:
    target = _normalize(constraint_text)
    found: List[Tuple[str, float]] = []
    seen: set = set()
    for pair in pairs:
        pair_id = pair.get("id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
            continue
        a = _normalize(pair.get("constraint_a", {}).get("text", ""))
        b = _normalize(pair.get("constraint_b", {}).get("text", ""))
        if target and (target == a or target == b):
            found.append((pair_id, 1.0))
            seen.add(pair_id)
            continue
        score = _pair_similarity(constraint_text, pair)
        if score >= RULE_BASED_COSINE_THRESHOLD:
            found.append((pair_id, score))
            seen.add(pair_id)
    found.sort(key=lambda x: x[1], reverse=True)
    return found


def ordered_pair_values(item: OverrideItem) -> List[str]:
    """Library pair texts, closer-to-anchor first. Presentation only."""
    a = (item.pair_a or "").strip()
    b = (item.pair_b or "").strip()
    if a and b:
        anchor = item.target_constraint_text
        if _cosine_similarity(anchor, b) > _cosine_similarity(anchor, a):
            return [b, a]
        return [a, b]
    return [text for text in (a, b) if text]


def _build_candidate_item(
    constraints: ConstraintResult,
    pairs: List[Dict[str, Any]],
    target_constraint_id: int,
    replacement_pair_id: str,
    score: float = 0.0,
) -> Optional[OverrideItem]:
    """Attach a library pair to a seed constraint. Do not pick a direction."""
    id_to_text = {c.id: c.description for c in constraints.constraints}
    target_text = id_to_text.get(target_constraint_id, "")
    if not target_text:
        return None

    pair = next((p for p in pairs if p.get("id") == replacement_pair_id), None)
    if not pair:
        return None

    a = str(pair.get("constraint_a", {}).get("text", "") or "").strip()
    b = str(pair.get("constraint_b", {}).get("text", "") or "").strip()
    if not a or not b:
        return None

    return OverrideItem(
        target_constraint_id=target_constraint_id,
        replacement_pair_id=replacement_pair_id,
        target_constraint_text=target_text,
        replacement_constraint_text="",
        pair_a=a,
        pair_b=b,
        dimension=pair.get("dimension", ""),
        score=score,
    )


def build_override_plan(
    constraints: ConstraintResult,
    i0: I0Result,
    replacement_pairs: Dict[str, Any],
    scene: str,
    override_mode: str = "unconditional",
) -> OverridePlanResult:
    """Return cosine-passing (constraint, pair) candidates. Do not pick one."""
    del i0

    pairs = _iter_all_pairs(replacement_pairs, scene)
    if not pairs:
        return OverridePlanResult(enable_override=False, override_mode=override_mode)

    raw: List[Tuple[int, str, float]] = []
    for constraint in constraints.constraints:
        for pair_id, score in _matches_for_constraint(constraint.description, pairs):
            raw.append((constraint.id, pair_id, score))
    raw.sort(key=lambda x: x[2], reverse=True)

    candidates: List[OverrideItem] = []
    seen: set = set()
    for cid, pid, score in raw:
        key = (cid, pid)
        if key in seen:
            continue
        resolved = _build_candidate_item(constraints, pairs, cid, pid, score=score)
        if not resolved:
            continue
        seen.add(key)
        candidates.append(resolved)
        if len(candidates) >= CANDIDATE_CAP:
            break

    return OverridePlanResult(
        enable_override=bool(candidates),
        candidates=candidates,
        overrides=[],
        override_mode=override_mode,
    )
