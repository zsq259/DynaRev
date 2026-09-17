"""Sub-claim resolution.

Two LLM-backed judges, each resolving one sub-claim type produced by the
decomposer:

- ``simple_verify``: a binary judgment over a predicate that can be
  resolved from done_message text + logical/factual reasoning. Modeled on
  Mind2Web-2's Simple Verifier — used for counts, distinctness, exclusion,
  format checks, and any predicate where page evidence is not required.

- ``verify_claim_on_frame``: a binary judgment over a (subject, assertion)
  page sub-claim against a single AXTree frame. Modeled on Mind2Web-2's
  URL Verifier — strict, evidence-grounded, no world-knowledge fishing.
  Returns whether the frame is the subject's host page, whether the
  assertion (if any) is supported by the page, and a short evidence
  summary used in diagnostics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from openai import OpenAI

from ..utils.llm import llm_json_call
from .errors import evaluator_error

_AXTREE_LIMIT = 20000


def _truncate_axtree(axtree: str) -> str:
    if len(axtree) <= _AXTREE_LIMIT:
        return axtree
    return axtree[:_AXTREE_LIMIT] + "...(truncated)"


def simple_verify(
    constraint_description: str,
    predicate: str,
    done_message: str,
    client: OpenAI,
    model: str = "deepseek-v4-flash",
    execution_date: str = "",
) -> Dict[str, Any]:
    """Binary verification of a predicate over done_message text.

    Resolves type='simple' sub-claims: count, distinctness, containment,
    format. Looks at done_message and applies logic / simple factual
    reasoning. Does NOT look at trajectory pages — that is the page
    verifier's job.

    Returns ``{"satisfied": bool, "reason": str, "confidence": float}``.
    """
    system_msg = (
        "You verify whether a given binary predicate is correct, given the "
        "agent's full final answer (done_message) and the original task "
        "constraint as context. The predicate is checkable by reading "
        "done_message and applying logic or simple factual reasoning. You "
        "do NOT have access to web pages — your judgment must come from "
        "the text you are given.\n\n"
        "GENERAL RULES:\n"
        "1. Examine the predicate carefully. Decide satisfied=true only "
        "when done_message demonstrably supports it; satisfied=false when "
        "it is contradicted or absent.\n"
        "2. The original task constraint is provided as context — it tells "
        "you why the predicate matters, but the predicate itself is what "
        "you must judge.\n"
        "3. Reasoning must cite specific text from done_message. Quote "
        "snippets using single quotes when relevant.\n"
        "4. Be strict about format predicates: 'lists exactly 3' means 3, "
        "not 2 or 4. 'Provides 3 distinct URLs' means 3 separate URLs, "
        "not one URL repeated.\n"
        "5. Apply fuzzy matching for minor variations (case, spacing, "
        "hyphenation, common abbreviations) — but do not stretch matches "
        "across genuinely different items.\n"
        "6. If done_message does not contain the information needed to "
        "judge the predicate, return satisfied=false (absence of evidence "
        "for an assertion is failure, not insufficiency).\n"
        "7. ANSWER-SECTION SCOPING. done_message is often a structured "
        "document with multiple sections — one per constraint or sub-"
        "question. When the predicate involves COUNT ('lists exactly N'), "
        "EXCLUSION ('X does not appear in the items'), DISTINCTNESS, or "
        "any other property OVER A LIST OF ITEMS, you MUST first identify "
        "which section of done_message the agent designated as the "
        "answer for THIS constraint. Apply the predicate ONLY to items "
        "in that section.\n"
        "    Section signals to look for:\n"
        "      - Bold headers like '**Three Batman comic books ...:**' "
        "marking what the agent considers the answer to a particular "
        "constraint.\n"
        "      - Numbered lists '1. ... 2. ... 3. ...' — typically the "
        "answer items.\n"
        "      - Explicit exclusion phrases like '(excluding X)' inside "
        "a header — these CONFIRM the agent has already excluded X from "
        "this section's items.\n"
        "      - Standalone reference URLs / context paragraphs separated "
        "from the numbered list — usually NOT part of the answer count.\n"
        "    Worked example. Suppose done_message has:\n"
        "        '**Injustice: Gods Among Us Year One** (the prequel ...):\n"
        "         https://www.dc.com/...year-one...\n"
        "         **Three Batman comic books by Tom Taylor (different "
        "         series, excluding Injustice):**\n"
        "         1. Detective Comics #1090 ...\n"
        "         2. Batman – One Bad Day ...\n"
        "         3. Batman: The Dark Knight ...'\n"
        "      Predicate 'done_message identifies exactly 3 distinct "
        "comic books': the answer section is the numbered list (3 items) "
        "→ satisfied=true. DO NOT count Injustice as a 4th item — it "
        "appears in a SEPARATE section that answers a different "
        "constraint.\n"
        "      Predicate 'Injustice: Gods Among Us Year One does not "
        "appear among the items listed': the relevant 'items' are the "
        "numbered list (the agent's answer to the find-three constraint); "
        "Injustice is NOT in that numbered list → satisfied=true. The "
        "agent's separate reference to Injustice (with its own URL, "
        "answering the link constraint) does NOT mean it appears 'among "
        "the items' for this exclusion predicate.\n"
        "      Predicate 'done_message provides exactly 3 distinct "
        "homepage URLs, one per comic book': count URLs in the THREE "
        "numbered comics' section, not URLs from the Injustice reference "
        "section.\n\n"
        "Output JSON only: "
        '{"satisfied": bool, "reason": "...", "confidence": 0.0-1.0}'
    )

    date_line = f"\nTask was executed on: {execution_date}\n" if execution_date else ""
    user_prompt = (
        f"Original constraint (context):\n{constraint_description}\n"
        f"{date_line}\n"
        f"Predicate to verify:\n{predicate}\n\n"
        f"Agent's done_message:\n{done_message}\n\n"
        "Is the predicate satisfied? Cite specific text from done_message."
    )

    try:
        payload = llm_json_call(
            client=client,
            system=system_msg,
            user=user_prompt,
            model=model,
            max_tokens=400,
        )
    except Exception as exc:
        raise evaluator_error("simple constraint verification failed", exc) from exc

    return {
        "satisfied": bool(payload.get("satisfied", False)),
        "reason": str(payload.get("reason", "")).strip(),
        "confidence": float(payload.get("confidence", 0.0)),
    }


def verify_branch_condition(
    condition_text: str,
    done_message: str,
    client: OpenAI,
    model: str = "deepseek-v4-flash",
    execution_date: str = "",
) -> Dict[str, Any]:
    """Three-state condition check for conditional-override tasks.

    Determines whether the agent's done_message reports the
    page-observable condition as holding, not holding, or addresses it
    at all. ``not_addressed`` is the third state required by conditional
    override evaluation — when the agent ignored the conditional clause,
    we cannot pick a branch and must surface the failure separately
    (see core.py Phase A).

    Returns ``{"condition_state": "true"|"false"|"not_addressed",
    "reason": str, "confidence": float}``. Evaluator failures raise instead
    of being confused with an agent that did not address the condition.
    """
    system_msg = (
        "You judge whether an agent's done_message addresses a specific "
        "page-observable condition. The condition came from a conditional "
        "override instruction (e.g. 'if X, then Y') — the agent was "
        "expected to check it on the page and report observation in the "
        "done_message.\n\n"
        "Return one of three states:\n"
        "  'true'          — done_message clearly states the condition HOLDS, "
        "with concrete values from the page.\n"
        "  'false'         — done_message clearly states the condition DOES "
        "NOT HOLD, with concrete values from the page.\n"
        "  'not_addressed' — done_message does not address the condition. "
        "Default to this when uncertain — do NOT guess.\n\n"
        "RULES:\n"
        "1. Look for explicit observation statements. Phrases like '[CONDITIONAL "
        "OBSERVATION]', 'I observed', 'the price difference is X', "
        "'I compared X and Y' are strong signals the agent addressed it.\n"
        "2. The agent's wording does not need to match the condition verbatim. "
        "'price difference is 160 RMB, below threshold' addresses 'price "
        "difference exceeds 500 RMB' → state='false'. '900 RMB diff exceeds "
        "500' → state='true'.\n"
        "3. Concrete values are required. 'I saw the prices' without numbers "
        "or comparison is not_addressed. Vague claims about thresholds without "
        "actual values are not_addressed.\n"
        "4. The agent's selection is NOT used here — only what the agent SAID "
        "the condition state was. Inconsistencies between what the agent "
        "reports and what the agent did get caught by downstream sub_claim "
        "judgment, not here.\n"
        "5. Reasoning must cite specific text from done_message.\n\n"
        "Output JSON only: "
        '{"condition_state": "true"|"false"|"not_addressed", '
        '"reason": "...", "confidence": 0.0-1.0}'
    )

    date_line = f"\nTask was executed on: {execution_date}\n" if execution_date else ""
    user_prompt = (
        f"Condition to evaluate:\n{condition_text}\n"
        f"{date_line}\n"
        f"Agent's done_message:\n{done_message}\n\n"
        "Did the agent address the condition? If yes, did the agent report "
        "it as true or false?"
    )

    try:
        payload = llm_json_call(
            client=client,
            system=system_msg,
            user=user_prompt,
            model=model,
            max_tokens=400,
        )
    except Exception as exc:
        raise evaluator_error("branch condition verification failed", exc) from exc

    state = str(payload.get("condition_state", "not_addressed")).strip().lower()
    if state not in ("true", "false", "not_addressed"):
        state = "not_addressed"

    return {
        "condition_state": state,
        "reason": str(payload.get("reason", "")).strip(),
        "confidence": float(payload.get("confidence", 0.0)),
    }


def verify_claim_on_frame(
    constraint_description: str,
    subject: str,
    agent_assertion: Optional[str],
    axtree: str,
    client: OpenAI,
    model: str = "deepseek-v4-flash",
    execution_date: str = "",
) -> Dict[str, Any]:
    """Verify a single (subject, agent_assertion) page sub-claim on a frame.

    Decides whether the frame can host a claim about ``subject`` (its
    own page, or a results view where it is a result item), and if
    ``agent_assertion`` is provided, whether this frame shows the
    subject has that property. Always returns a short
    ``evidence_summary`` (1-2 sentences) describing what the page shows
    about the subject in the context of the constraint.

    Returns ``{"is_subject_page": bool, "assertion_supported": bool|None,
    "evidence_summary": str, "confidence": float, "reason": str}``.
    ``assertion_supported`` is ``None`` when ``agent_assertion`` is
    ``None``.
    """
    has_assertion = agent_assertion is not None and agent_assertion.strip() != ""

    system_msg = (
        "You verify a page sub-claim against a GUI page snapshot (AXTree), "
        "in the context of a task constraint. Your judgment must come from "
        "the AXTree only — do NOT use world knowledge or background facts "
        "to fill gaps.\n\n"
        "A sub-claim has two parts:\n"
        "  subject — the item the page should be about (book, product, "
        "person, URL fragment, domain, etc.)\n"
        "  agent_assertion — what the page must show about the subject. "
        "May be null when the sub-claim only requires the subject to be "
        "shown, with no further attribute check.\n\n"
        "Decide three things:\n\n"
        "1. is_subject_page: Can this frame host a claim about the "
        "subject? True when the page is the subject's detail / content / "
        "profile / home page, or is otherwise centered on the subject "
        "(e.g. for domain subjects like 'sephora.com', any page on that "
        "host counts). Also true when this is a listing / search / "
        "filter / results view and the subject appears as a result item "
        "(card, row, or hit). False when the subject only appears as a "
        "navigation menu item, an unrelated link, or a passing mention.\n\n"
        "2. assertion_supported: If agent_assertion is provided, does "
        "THIS SAME frame show that the subject has that property? True "
        "when the page contains direct textual evidence (synopsis text, "
        "attribute labels, character lists, byline, price tags, series "
        "labels), OR when the subject is a result item here and the "
        "view's visible filters / query / category / chips entail the "
        "property. False when this frame does not contain such "
        "evidence — even if the assertion happens to be true in the "
        "world. Null when agent_assertion itself is null.\n\n"
        "3. evidence_summary: One or two short sentences describing what "
        "this page shows about the subject relevant to the constraint. "
        "Quote or paraphrase short snippets directly from the AXTree. "
        "If the page is not about the subject, summarize what the page "
        "actually is. This summary is used in diagnostic reasons — it "
        "should be specific enough that a reader can reconstruct why you "
        "decided as you did.\n\n"
        "STRICTNESS:\n"
        "- Apply fuzzy matching for minor variations (case, spacing, "
        "hyphenation, year format, common abbreviations) — but final "
        "decisions must rely on explicit AXTree evidence.\n"
        "- If the page is entirely irrelevant or inaccessible (error page, "
        "blank, blocked), conclude is_subject_page=false directly.\n"
        "- For null agent_assertion: is_subject_page=true is sufficient — "
        "you do not need to verify any further attribute.\n\n"
        "Examples:\n"
        "- subject='Batman: The Dark Knight', agent_assertion='2021 "
        "miniseries by Tom Taylor'\n"
        "  AXTree shows a comic detail page with the title, '2021 "
        "miniseries' label, and 'by Tom Taylor' byline →\n"
        "    is_subject_page=true, assertion_supported=true, "
        "evidence_summary='detail page identifies the comic as Batman: "
        "The Dark Knight (2021 miniseries) by Tom Taylor'.\n"
        "- subject='DCeased', agent_assertion='heavily features Batman/"
        "Bat-family'\n"
        "  AXTree shows the DCeased detail page with a synopsis about a "
        "zombie outbreak, no mention of Batman characters →\n"
        "    is_subject_page=true, assertion_supported=false, "
        "evidence_summary='detail page identifies DCeased (2019-2021 "
        "ongoing series); synopsis describes a zombie outbreak with no "
        "Batman/Bat-family characters visible'.\n"
        "- subject='American Hairless Terrier', agent_assertion='medium "
        "barking'\n"
        "  AXTree is a breed listing with Barking Level 'Medium' checked "
        "and a result card 'American Hairless Terrier' →\n"
        "    is_subject_page=true, assertion_supported=true, "
        "evidence_summary='breed is a result item on a listing filtered "
        "to medium barking'.\n"
        "- subject='sephora.com', agent_assertion=null\n"
        "  AXTree shows a page with URL containing sephora.com →\n"
        "    is_subject_page=true, assertion_supported=null, "
        "evidence_summary='page is on sephora.com (URL visible in nav)'.\n"
        "- subject='search?q=Injustice', agent_assertion='homepage of "
        "Injustice on DC.com'\n"
        "  AXTree shows a search results listing on dc.com →\n"
        "    is_subject_page=true, assertion_supported=false, "
        "evidence_summary='page is a dc.com search results listing for "
        "Injustice, not a comic homepage'.\n\n"
        'Respond with JSON only: {"is_subject_page": bool, '
        '"assertion_supported": bool | null, "evidence_summary": "...", '
        '"confidence": 0.0-1.0, "reason": "..."}'
    )

    assertion_line = (
        f"agent_assertion: {agent_assertion}\n"
        if has_assertion
        else "agent_assertion: null (only the subject's presence on the page "
        "needs to be verified)\n"
    )
    date_line = f"\nTask was executed on: {execution_date}\n" if execution_date else ""

    user_prompt = (
        f"Constraint:\n{constraint_description}\n\n"
        f"Sub-claim to verify:\n  subject: {subject}\n  {assertion_line}"
        f"{date_line}\n"
        f"Page state (AXTree):\n{_truncate_axtree(axtree)}\n\n"
        "Verify the sub-claim against the AXTree."
    )

    try:
        payload = llm_json_call(
            client=client,
            system=system_msg,
            user=user_prompt,
            model=model,
            max_tokens=400,
        )
    except Exception as exc:
        raise evaluator_error("page evidence verification failed", exc) from exc

    is_subject_page = bool(payload.get("is_subject_page", False))
    if has_assertion:
        raw_supported = payload.get("assertion_supported")
        assertion_supported = bool(raw_supported) if raw_supported is not None else False
    else:
        assertion_supported = None

    return {
        "is_subject_page": is_subject_page,
        "assertion_supported": assertion_supported,
        "evidence_summary": str(payload.get("evidence_summary", "")).strip(),
        "confidence": float(payload.get("confidence", 0.0)),
        "reason": str(payload.get("reason", "")).strip(),
    }
