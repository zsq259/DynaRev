"""Constraint decomposer.

Given a single atomic constraint plus the agent's full done_message, this
module produces a list of binary sub-claims that, ANDed together, cover the
constraint completely. Each sub-claim is independently resolvable by either:

- ``simple``: a predicate over done_message text + logical/factual reasoning
  (counting, distinctness, containment, format) — resolved by Mind2Web-2-style
  Simple Verifier without consulting trajectory pages.
- ``page``: a claim about a specific subject that requires page evidence —
  resolved by ``verify_claim_on_frame`` over the host AXTree frame.

The decomposer replaces the older ``done_judge`` triage step. There is no
longer a global done_says classification — agent self-admitted failures
fall out as ``simple`` sub-claims that fail (agent did not list N items),
and process-style constraints (e.g. "on sephora.com") are expressed as
``page`` sub-claims with an empty assertion.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .errors import EvaluatorError, evaluator_error

from ..utils.llm import llm_json_call

@dataclass
class SubClaim:
    """One conjunct of a constraint decomposition."""

    claim_id: int
    type: str  # "simple" | "page"
    description: str
    # type == "page" only:
    subject: Optional[str] = None
    assertion: Optional[str] = None


_SYSTEM_PROMPT = (
    "You decompose a single atomic task constraint into a list of binary "
    "sub-claims. The constraint is satisfied IFF all sub-claims are "
    "satisfied (logical AND). Each sub-claim must be independently "
    "checkable.\n\n"
    "INPUTS you will receive:\n"
    "- The complete instruction timeline: I0 and revisions in order\n"
    "- A constraint description (one atomic requirement)\n"
    "- The agent's full final answer (done_message)\n\n"
    "Interpret the current constraint in the context of the instruction timeline. "
    "Preserve its scope and applicable revisions; generate sub-claims only for "
    "the current constraint, without adding requirements. Later overrides replace "
    "earlier requirements on the same dimension; do not reintroduce superseded "
    "requirements or combine mutually exclusive conditional alternatives. "
    "Use the timeline as context, not as a checklist to add unrelated constraints.\n\n"
    "Each sub-claim has a `type`:\n\n"
    "- type='simple': resolvable by reading done_message text and applying "
    "logic. No page evidence needed. Examples of simple predicates:\n"
    "    'done_message identifies exactly 3 distinct items'\n"
    "    'the 3 series labels in done_message are pairwise distinct'\n"
    "    'Injustice: Gods Among Us Year One does not appear in the items '\n"
    "      'listed in done_message'\n"
    "    'done_message provides exactly 3 separate URLs, one per item'\n\n"
    "- type='page': requires actual page content from the agent's trajectory. "
    "Provide subject + optional assertion. The claim is that the subject "
    "has the stated property; evidence need not come from the subject's "
    "detail page. Examples of page predicates:\n"
    "    subject='DCeased', assertion='heavily features Batman/Bat-family'\n"
    "      → evidence that DCeased has this property.\n"
    "    subject='Batman: The Dark Knight', assertion='by Tom Taylor'\n"
    "      → evidence that it is by Tom Taylor.\n"
    "    subject='Injustice', assertion='page is the homepage of Injustice on "
    "      DC.com'\n"
    "    subject='sephora.com', assertion=null\n"
    "      → trajectory must show a page on sephora.com (no further check).\n\n"
    "DECOMPOSITION RULES:\n"
    "1. EXHAUSTIVE: cover every verifiable dimension of the constraint. "
    "Missing a dimension means false positives slip through.\n"
    "2. DISJOINT: don't repeat the same check.\n"
    "3. PREFER SIMPLE: when the same check can be done from done_message "
    "text alone OR via a page, prefer simple. Use page only when text is "
    "genuinely insufficient (per-item factual properties, byline, synopsis "
    "content, URL semantics, trajectory presence).\n"
    "4. PER-ITEM × LIST decomposition: for constraints of the form 'each "
    "X must Y' or 'find N X with property Y':\n"
    "    - Emit one PAGE sub-claim per item the agent listed (subject + "
    "assertion describing Y).\n"
    "    - Emit any SIMPLE sub-claims about the list as a whole (count, "
    "pairwise distinctness, exclusion).\n"
    "5. ANSWER-SECTION SCOPING: done_message is often a structured "
    "document. The agent may write multiple sections — one for each "
    "constraint or sub-question of the task. When a constraint says "
    "'each X', 'N X', 'exclude Y', or otherwise refers to a list of "
    "items, IDENTIFY THE SPECIFIC SECTION the agent designated as the "
    "answer for THIS constraint, and scope your decomposition to that "
    "section only. Do NOT include items from other sections (reference "
    "items, items mentioned for context, items answering a different "
    "constraint) as if they were part of the answer.\n"
    "    Section signals to look for in done_message:\n"
    "      - Bold headers like '**Three Batman comic books ...:**' "
    "marking what the agent considers the answer to a particular "
    "constraint.\n"
    "      - Numbered lists '1. ... 2. ... 3. ...' — typically the answer "
    "items.\n"
    "      - Explicit exclusion phrases like '(excluding X)' inside a "
    "header — these CONFIRM the agent has already excluded X from this "
    "section's items.\n"
    "      - Standalone reference URLs / context paragraphs separated "
    "from the numbered list — usually NOT part of the answer count.\n"
    "    Example (from a real task — see worked examples below):\n"
    "      done_message has:\n"
    "        '**Injustice: Gods Among Us Year One** (the prequel ...):\n"
    "         https://www.dc.com/...year-one...\n"
    "         **Three Batman comic books by Tom Taylor (different series, "
    "         excluding Injustice):**\n"
    "         1. Detective Comics #1090 ...\n"
    "         2. Batman – One Bad Day ...\n"
    "         3. Batman: The Dark Knight ...'\n"
    "      For the constraint 'Find exactly three comic books', the "
    "answer items are the THREE numbered comics — NOT four. Injustice "
    "appears in a separate section answering a different constraint "
    "('provide the link to Year One on DC.com'); it is reference, not "
    "part of the count.\n"
    "      For the constraint 'Each comic must feature Batman', the "
    "'each' refers to the THREE in the numbered list — NOT to Injustice. "
    "Decompose 3 page sub-claims (one per numbered comic), NOT 4.\n"
    "      For the constraint 'Exclude Injustice itself', the 3 numbered "
    "comics are the candidate set; agent's header explicitly says "
    "'excluding Injustice'; emit one simple sub-claim 'none of the 3 "
    "numbered Batman comics is Injustice: Gods Among Us Year One' — do "
    "NOT treat Injustice's own reference link as 'inclusion'.\n"
    "6. AGENT SAID NOTHING: if the agent did not address this constraint "
    "at all in done_message, decomposition still proceeds — emit the "
    "sub-claims the constraint would require regardless. simple sub-claims "
    "checking 'done_message says X' will fail naturally; page sub-claims "
    "for process-style requirements may still resolve from trajectory.\n\n"
    "FIELDS for type='page':\n\n"
    "subject: A VERBATIM SUBSTRING of done_message that identifies the "
    "item. The subject must appear EXACTLY AS IT APPEARS in done_message "
    "— do NOT reformat, do NOT strip punctuation (#, –, :, etc.), do NOT "
    "lowercase, do NOT hyphenate, do NOT pull a path segment out of a URL "
    "if a readable name is also available.\n\n"
    "Self-check before writing subject: can you find the subject as a "
    "contiguous substring of done_message? If no, you reformulated — "
    "pick a real verbatim substring instead.\n\n"
    "Preference among multiple valid verbatim substrings:\n"
    "  1. When agent provides BOTH a human-readable name AND a URL for "
    "an item, USE THE NAME. Names appear as display text in AXTrees; "
    "URL slugs do not.\n"
    "  2. Pick the LONGEST distinctive verbatim substring specific to "
    "the item. Avoid generic series names that match many pages.\n"
    "  3. URL substrings are only acceptable when no readable name is "
    "available (e.g. agent supplied only a link). Even then, prefer a "
    "short distinctive verbatim token over the full URL.\n\n"
    "Examples:\n"
    "  done_message: '1. **Detective Comics #1090** (ongoing series) - "
    "https://www.dc.com/comics/detective-comics-2016/detective-comics-"
    "1090'\n"
    "    ✓ subject='Detective Comics #1090'  (verbatim, with #, name "
    "preferred)\n"
    "    ✗ subject='Detective Comics 1090'   (NOT verbatim — # was "
    "dropped)\n"
    "    ✗ subject='Detective Comics'        (verbatim BUT generic; "
    "matches many DC pages)\n"
    "    ✗ subject='detective-comics-1090'   (URL slug — verbatim but "
    "doesn't grep-hit display text; name is available, prefer it)\n\n"
    "  done_message: 'Batman: The Dark Knight (2021 miniseries)'\n"
    "    ✓ subject='Batman: The Dark Knight'   (verbatim substring)\n"
    "    ✗ subject='Batman: The Dark Knight (2021 miniseries)'   "
    "(verbatim BUT includes parenthetical that varies across pages)\n"
    "    ✗ subject='Batman: The Dark Knight 2021 miniseries'   (NOT "
    "verbatim — parenthesis changed)\n\n"
    "  done_message: 'https://www.dc.com/search?q=Injustice%3A+Year+One' "
    "(no readable name attached)\n"
    "    ✓ subject='Injustice%3A+Year+One' or 'Injustice'   (verbatim "
    "URL substrings; agent gave no name)\n"
    "    ✗ subject='Injustice: Year One'   (NOT verbatim — URL-decoded)\n\n"
    "assertion: the property the subject must have. THREE cases:\n"
    "  (a) Agent wrote a specific say-so verbatim: COPY EXACTLY.\n"
    "      done_message: 'DCeased (heavily features Batman/Bat-family)'\n"
    "      → assertion='heavily features Batman/Bat-family'\n"
    "  (b) Constraint requires a per-item attribute that agent claimed only "
    "globally (or did not claim per item, but the attribute is verifiable "
    "as a property of each item): use a SHORT canonical phrase for that "
    "property.\n"
    "      Constraint: 'Each comic by Tom Taylor', agent: 'Author: Tom "
    "Taylor (one global statement)'\n"
    "      → For each item: assertion='by Tom Taylor' (canonical, not "
    "verbatim agent text — but only used because the per-item byline is "
    "the property that must be demonstrated).\n"
    "  (c) Subject just needs to exist / be visited (count, type, "
    "trajectory-presence constraints): assertion=null.\n"
    "      Constraint: 'Find 3 comic books' → for each subject, "
    "assertion=null (the page just needs to show it's a real comic).\n"
    "      Constraint: 'on sephora.com' → subject='sephora.com', "
    "assertion=null.\n\n"
    "DO NOT FABRICATE: do not synthesize per-item assertions from the "
    "constraint wording when the constraint is about the LIST AS A WHOLE "
    "(not per-item).\n"
    "  Constraint: 'Find exactly three comic books'\n"
    "  WRONG: per-item assertion='one of three comic books' — fabricated; "
    "the COUNT is a list-level property, handled by a simple sub-claim.\n"
    "  RIGHT: emit one simple sub-claim 'done_message lists exactly 3 "
    "items' + one page sub-claim per item with assertion=null.\n\n"
    "Worked examples:\n\n"
    "Constraint: 'Find exactly three comic books'\n"
    "done_message lists Batman: The Dark Knight, Batman: The Detective, "
    "DCeased\n"
    "Sub-claims:\n"
    "  {type: simple, description: 'done_message identifies exactly 3 "
    "distinct comic books'}\n"
    "  {type: page, subject: 'Batman: The Dark Knight', assertion: null, "
    "description: 'Batman: The Dark Knight is shown to be a real comic '\n"
    "    'book on the trajectory'}\n"
    "  {type: page, subject: 'Batman: The Detective', assertion: null, "
    "description: 'Batman: The Detective is shown to be a real comic '\n"
    "    'book on the trajectory'}\n"
    "  {type: page, subject: 'DCeased', assertion: null, description: "
    "'DCeased is shown to be a real comic book on the trajectory'}\n\n"
    "Constraint: 'Each comic must feature Batman'\n"
    "done_message: '... 1. Batman: The Dark Knight (2021) 2. Batman: The "
    "Detective (2022) 3. DCeased (heavily features Batman/Bat-family)'\n"
    "Sub-claims:\n"
    "  {type: page, subject: 'Batman: The Dark Knight', assertion: null, "
    "description: 'Batman: The Dark Knight features Batman (title self-"
    "evident)'}\n"
    "  {type: page, subject: 'Batman: The Detective', assertion: null, "
    "description: 'Batman: The Detective features Batman (title self-"
    "evident)'}\n"
    "  {type: page, subject: 'DCeased', assertion: 'heavily features "
    "Batman/Bat-family', description: 'DCeased page corroborates the "
    "agent statement'}\n\n"
    "Constraint: 'Each comic from a different series'\n"
    "done_message: '1. Batman: The Dark Knight (2021 miniseries), 2. "
    "Batman: The Detective (2022 miniseries), 3. DCeased (2019-2021 "
    "series)'\n"
    "Sub-claims:\n"
    "  {type: simple, description: 'the 3 series labels in done_message '\n"
    "    '(2021 miniseries / 2022 miniseries / 2019-2021 series) are '\n"
    "    'pairwise distinct'}\n"
    "  {type: page, subject: 'Batman: The Dark Knight', assertion: '2021 "
    "miniseries', description: 'Dark Knight page corroborates 2021 "
    "miniseries series label'}\n"
    "  {type: page, subject: 'Batman: The Detective', assertion: '2022 "
    "miniseries', description: 'Detective page corroborates 2022 miniseries "
    "label'}\n"
    "  {type: page, subject: 'DCeased', assertion: '2019-2021 series', "
    "description: 'DCeased page corroborates 2019-2021 series label'}\n\n"
    "Constraint: 'Exclude Injustice: Gods Among Us Year One itself'\n"
    "Sub-claims:\n"
    "  {type: simple, description: 'Injustice: Gods Among Us Year One does "
    "not appear among the items listed in done_message'}\n\n"
    "Constraint: 'Provide a link to the homepage of Year One on DC.com'\n"
    "done_message: 'Links: https://www.dc.com/search?q=Injustice%3A+Year"
    "+One'\n"
    "Sub-claims:\n"
    "  {type: page, subject: 'search?q=Injustice', assertion: 'page is the "
    "homepage of Injustice: Gods Among Us Year One on DC.com', description: "
    "'the URL agent provided resolves to the comic homepage on DC.com'}\n\n"
    "Constraint: 'Provide a link to the homepage of each comic on DC.com'\n"
    "done_message: 'Links: https://www.dc.com/search?q=Injustice...' "
    "(only one link given)\n"
    "Sub-claims:\n"
    "  {type: simple, description: 'done_message provides 3 distinct "
    "homepage URLs, one per comic book'}\n"
    "  (no page sub-claims — agent did not provide per-comic links; the "
    "simple sub-claim already fails)\n\n"
    "Output JSON only:\n"
    '{"sub_claims": [\n'
    '  {"claim_id": 1, "type": "simple", "description": "..."},\n'
    '  {"claim_id": 2, "type": "page", "description": "...", '
    '"subject": "...", "assertion": "..." | null}\n'
    "]}"
)


def extract_done_message(events: List[Dict[str, Any]]) -> str:
    """Extract the agent's final done message from a list of step events.

    The convention is the last ``send_msg_to_user("done:...")`` call. We scan
    in reverse and parse the first matching call's string argument with
    ``ast.literal_eval`` to handle escapes correctly. Returns the empty
    string when nothing usable is found.

    Prefers the ``action`` field (parsed/cleaned tool call) over
    ``raw_output`` (full model output, which may have reasoning prose
    prepended before the actual ``send_msg_to_user(...)`` call).
    """
    for ev in reversed(events):
        raw = (ev.get("action") or ev.get("raw_output") or "").strip()
        if not raw.startswith("send_msg_to_user("):
            continue
        inner = raw[len("send_msg_to_user(") :]
        if inner.endswith(")"):
            inner = inner[:-1]
        try:
            msg = ast.literal_eval(inner)
        except (SyntaxError, ValueError):
            continue
        if isinstance(msg, str):
            return msg
    return ""


def decompose(
    constraint_description: str,
    done_message: str,
    client: OpenAI,
    model: str,
    execution_date: str = "",
    instruction_timeline: str = "",
) -> List[SubClaim]:
    """Decompose a constraint into binary sub-claims.

    Raises EvaluatorError when the evaluator model cannot return a valid
    decomposition. Evaluator failures must not be scored as agent failures.
    """
    date_line = f"\nTask was executed on: {execution_date}" if execution_date else ""
    timeline_context = (
        f"Complete instruction timeline:\n{instruction_timeline}\n\n"
        if instruction_timeline else ""
    )
    user = (
        timeline_context
        + f"Constraint:\n{constraint_description}{date_line}\n\n"
        f"done_message:\n{done_message}\n\n"
        "Decompose the constraint into binary sub-claims."
    )

    try:
        payload = llm_json_call(
            client=client,
            system=_SYSTEM_PROMPT,
            user=user,
            model=model,
            max_tokens=2048,
        )
    except Exception as exc:
        raise evaluator_error("constraint decomposition failed", exc) from exc

    raw_subclaims = payload.get("sub_claims") or payload.get("subclaims") or []
    out: List[SubClaim] = []
    for i, raw in enumerate(raw_subclaims):
        if not isinstance(raw, dict):
            continue
        typ = raw.get("type", "")
        if typ not in ("simple", "page"):
            continue
        description = (raw.get("description") or "").strip()
        if not description:
            continue
        cid = raw.get("claim_id")
        try:
            cid = int(cid)
        except (TypeError, ValueError):
            cid = i + 1

        subject: Optional[str] = None
        assertion: Optional[str] = None
        if typ == "page":
            subj_raw = raw.get("subject")
            if not isinstance(subj_raw, str) or not subj_raw.strip():
                # page sub-claim without a subject is malformed; skip
                continue
            subject = subj_raw.strip()
            assertion_raw = raw.get("assertion")
            if isinstance(assertion_raw, str):
                assertion = assertion_raw.strip() or None

        out.append(
            SubClaim(
                claim_id=cid,
                type=typ,
                description=description,
                subject=subject,
                assertion=assertion,
            )
        )
    if not out:
        raise EvaluatorError("constraint decomposition returned no valid sub-claims")
    return out
