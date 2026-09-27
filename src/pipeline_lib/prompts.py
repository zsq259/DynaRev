"""All LLM prompts for the 4-step pipeline, extracted from dyna_pipeline_design.md."""

# ── Step 1: Constraint Extraction ─────────────────────────────────────────────

STEP1_SYSTEM = (
    "You are an expert at analyzing task instructions. "
    "Your job is to extract the constraints that an evaluator can actually check."
)

STEP1_USER = """\
Given the following task instruction, extract all constraints and output a JSON object.

A constraint is a condition, requirement, or filter stated in the instruction that can be checked from the final answer and/or a cited page.

Output format:
```json
{{
  "original_instruction": "...",
  "constraints": [
    {{"id": 1, "description": "..."}},
    {{"id": 2, "description": "..."}}
  ]
}}
```

Rules:
- Each constraint must describe exactly one thing — do not merge multiple constraints into one
- Write each constraint as a clear, self-contained statement
- Preserve logical structure inside a single constraint when the original instruction expresses alternatives or conjunctions.
- If the original instruction contains an OR relation (e.g., "A or B"), keep it as ONE composite constraint with the OR relation explicit.
- Do not split one OR condition into two separate atomic constraints.
- Extract what the instruction asks for, at the same scope it was written. Do not invent extra requirements or promote a vague overall goal into a per-item rule.
- Skip wording that is only subjective or scene-setting (e.g. fun, nice, good). If a sentence mixes that with a concrete requirement, keep the concrete part.

Here are some examples:

**Example 1:**
Task instruction: `Show me the list of Men's Blazers, Black, Size M on Uniqlo.`
```json
{{
  "original_instruction": "Show me the list of Men's Blazers, Black, Size M on Uniqlo.",
  "constraints": [
    {{"id": 1, "description": "Product type must be Blazers"}},
    {{"id": 2, "description": "Must be from the Men's category"}},
    {{"id": 3, "description": "Color must be Black"}},
    {{"id": 4, "description": "Size must be M"}}
  ]
}}
```

**Example 2:**
Task instruction: `Find Mac external hard drives with capacity greater than 2TB, sorted by price from lowest to high, showing availability.`
```json
{{
  "original_instruction": "Find Mac external hard drives with capacity greater than 2TB, sorted by price from lowest to high, showing availability.",
  "constraints": [
    {{"id": 1, "description": "Product type must be external hard drives"}},
    {{"id": 2, "description": "Must be compatible with Mac"}},
    {{"id": 3, "description": "Capacity must be greater than 2TB"}},
    {{"id": 4, "description": "Results must be sorted by price from lowest to highest"}},
    {{"id": 5, "description": "Availability must be shown"}}
  ]
}}
```

**Example 3:**
Task instruction: `I am a DC Comics fan. Could you find three comic books from different series written by the same author as Injustice: Gods Among Us Year One that feature Batman (excluding Injustice itself)? Please provide the links to the homepage of Injustice and these comic books on DC comic's official website.`
```json
{{
  "original_instruction": "I am a DC Comics fan. Could you find three comic books from different series written by the same author as Injustice: Gods Among Us Year One that feature Batman (excluding Injustice itself)? Please provide the links to the homepage of Injustice and these comic books on DC comic's official website.",
  "constraints": [
    {{"id": 1, "description": "Comics must be found on DC.com"}},
    {{"id": 2, "description": "The comics must be written by the same author as Injustice: Gods Among Us Year One"}},
    {{"id": 3, "description": "Each of the comics kept after Year One must feature Batman"}},
    {{"id": 4, "description": "The final answer must identify exactly three comic books besides Year One"}},
    {{"id": 5, "description": "The three comic books must be from different series"}},
    {{"id": 6, "description": "Injustice: Gods Among Us Year One itself must be excluded from the three comic books"}},
    {{"id": 7, "description": "The final answer must include a DC.com page link for Injustice: Gods Among Us Year One"}},
    {{"id": 8, "description": "The final answer must include a DC.com page link for each of the three comic books"}}
  ]
}}
```

Now extract constraints from the following task instruction:

Task instruction: {task}"""


# ── Step 2: I0 Design ────────────────────────────────────────────────────────

STEP2_SYSTEM = (
    "You are an expert at analyzing website homepages and determining what information "
    "is required to initiate a search or action."
)

STEP2_USER = """\
You are given a list of task constraints and a description of a website's homepage.

Step 1: For each constraint, classify whether it is REQUIRED by the homepage to initiate \
the first search or action, or whether it can be added LATER after seeing results.

A constraint is REQUIRED only if the homepage has a form field or input that MUST be filled \
with this information before the user can proceed. \
Preferences, filters, and secondary refinements that do not block initial navigation are LATER.

Step 2: Design I0 as the minimum but naturally complete starting request.

**Principle:**
- I0 should be phrased as a natural user instruction that expresses a complete intent
- Start with REQUIRED constraints; if they alone form a natural complete request, use only those
- If REQUIRED constraints alone would result in an unnatural or vague request that obscures the user's core intent, \
  you may supplement by selecting constraints from LATER that clarify the main goal or subject matter
- Avoid over-supplementing; only add what is necessary to express the user's genuine intent
- Do NOT make I0 over-complete by including every constraint. If a constraint can naturally be refined after initial results \
  (e.g., source/site preference, ranking preference, secondary filters), prefer leaving it for later turns.
- I0 must explicitly include the core topic/subject of the task; avoid generic openings that only describe an action
- Avoid underspecified I0 such as "search for products", "look for guides", or other action-only phrasing without the task's key subject

The key criterion is whether I0 would sound like something a real user would naturally say when starting the task, \
with their core intent clearly expressed.

{previous_step4_context}

Homepage description:
{homepage_description}

Constraints:
{constraints}

Output format:
```json
{{
  "classification": [
    {{"id": 1, "required": true, "reason": "homepage has a destination field"}},
    {{"id": 2, "required": false, "reason": "this is a filter, not a homepage field"}}
  ],
  "i0": "..."
}}
```

Here are some examples:

**Example 1:**
Homepage description: Trip.com train search form (Trains tab). Departure station and arrival station inputs must be filled before Search will run. The form also has a departure date/time control, but it already has a default. High-speed vs other train types is not a required homepage field.

Constraints:
```json
{{
  "constraints": [
    {{"id": 1, "description": "Must be a high speed train ticket"}},
    {{"id": 2, "description": "Departure time must be before 23:00"}},
    {{"id": 3, "description": "Origin must be Shanghai"}},
    {{"id": 4, "description": "Destination must be Beijing"}}
  ]
}}
```

```json
{{
  "classification": [
    {{"id": 1, "required": false, "reason": "high-speed is not a blocking field on the train search form"}},
    {{"id": 2, "required": false, "reason": "departure date/time already has a default; before 23:00 is not required to run Search"}},
    {{"id": 3, "required": true, "reason": "departure-station field must be filled before Search can run"}},
    {{"id": 4, "required": true, "reason": "arrival-station field must be filled before Search can run"}}
  ],
  "i0": "Find a high speed train ticket from Shanghai to Beijing."
}}
```

**Example 2:**
Homepage description: DC.com comics catalog. Titles can be browsed without filling the search box. Character, series, quantity, and link details are later refinements.

Constraints:
```json
{{
  "constraints": [
    {{"id": 1, "description": "Comics must be found on DC.com"}},
    {{"id": 2, "description": "The comics must be written by the same author as Injustice: Gods Among Us Year One"}},
    {{"id": 3, "description": "Each of the comics kept after Year One must feature Batman"}},
    {{"id": 4, "description": "The final answer must identify exactly three comic books besides Year One"}},
    {{"id": 5, "description": "The three comic books must be from different series"}},
    {{"id": 6, "description": "Injustice: Gods Among Us Year One itself must be excluded from the three comic books"}},
    {{"id": 7, "description": "The final answer must include a DC.com page link for Injustice: Gods Among Us Year One"}},
    {{"id": 8, "description": "The final answer must include a DC.com page link for each of the three comic books"}}
  ]
}}
```

```json
{{
  "classification": [
    {{"id": 1, "required": false, "reason": "already on DC.com; not a blocking homepage field"}},
    {{"id": 2, "required": false, "reason": "catalog can be browsed without filling search; author is not a required homepage field"}},
    {{"id": 3, "required": false, "reason": "character filter after seeing the author's comics"}},
    {{"id": 4, "required": false, "reason": "quantity can be decided after results"}},
    {{"id": 5, "required": false, "reason": "series diversity is a later refinement"}},
    {{"id": 6, "required": false, "reason": "exclusion is a later refinement"}},
    {{"id": 7, "required": false, "reason": "answer-format link, not needed to start the search"}},
    {{"id": 8, "required": false, "reason": "answer-format links, not needed to start the search"}}
  ],
  "i0": "On DC.com, find Injustice: Gods Among Us Year One and other comic books by the same author."
}}
```

Now classify and design I0."""


# ── Step 3: Multi-turn Instruction Design ─────────────────────────────────────

STEP3_SYSTEM = (
    "You are an expert at designing natural multi-turn user interactions for GUI tasks. "
    "Your job is to arrange a list of constraints into a natural multi-turn instruction "
    "sequence, simulating how a real user would progressively refine their request "
    "during task execution. "
    "For each subsequent instruction, you also define a milestone "
    "that determines when it should be injected."
)

STEP3_USER = """\
Given the original task instruction, its extracted constraints, and the pre-designed I0, \
complete the multi-turn instruction sequence and define milestones.

## Part 1: Instruction Sequence

**I0:**
- Already provided — do not modify it

**Subsequent instructions:**
- Introduce remaining constraints (those NOT in I0) in the order a real user would naturally \
think of them after seeing intermediate results
- A constraint should be introduced later if the user would only think of it after seeing \
some results or making progress
- Do NOT repeat or restate constraints that are already clearly expressed in I0
- Every instruction after I0 must add new information or a new restriction that was not \
already stated; paraphrasing existing constraints is not allowed
- If a constraint is already fully specified in I0, it must not appear again in later turns \
unless the later turn adds genuinely new specificity beyond I0
- Avoid weak increments such as tone strengthening, exclusion of obvious negatives, \
or restating the same constraint with different wording — these do not count as new revisions
- Group constraints that a user would naturally say together in the same turn. \
Do NOT split constraints of the same type across multiple turns just to inflate the count
- Each Step1 constraint should be introduced exactly once by one instruction. I0 covers some constraints; subsequent instructions should cover the rest.
- For composite constraints (including OR/AND relations), treat the full logic expression as one unit and introduce it in one instruction without changing its operators.
- Do not rewrite logical type across turns (OR must remain OR; AND must remain AND).
- Avoid artificial decomposition: if only one natural follow-up remains, keep the sequence concise but still natural.

**Style:**
- All instructions should be written in natural user voice, as if the user is speaking directly
- Use phrasing like "Show me...", "Only show...", "Now set..." rather than task description style

## Part 2: Milestones

For each instruction i_k (k >= 1), define a milestone that triggers its injection.

A milestone is a **branch-robust semantic task state** reached after correctly completing \
i0 through i_{{k-1}}. It should describe user-visible task progress, not UI mechanics or \
system internals.

How to produce each milestone:
1. Summarize what prior instructions have semantically established (intent/state, not actions).
2. Phrase that as a user-understandable task state that remains valid across common branches \
  (including empty-result branches).
3. Check two conditions: (a) saying i_k at this state would not feel abrupt for the user, and (b) among stages satisfying (a), this is an earlier one that is reliably reachable by diverse agent trajectories.

Quality guidance:
- Derive the milestone only from i0..i_{{k-1}}; never leak content from i_k.
- Prefer semantic state wording over UI-surface wording.
- Avoid implementation-specific language such as "system processed/registered/executed...".
- A milestone should hold across all branches that correct prior execution can produce. When a concrete phrasing like "X is displayed" is branch-robust for this task, use it; otherwise phrase more general.
- Necessity is a baseline, not a goal — multiple stages on a correct completion path usually satisfy it. Prefer an earlier candidate over one that restates i0's full terminal condition, since the latter collapses the milestone into an i0-completion check.

Examples illustrating the principle:
- "The search has been constrained to the previously requested criteria, including cases with zero matches" — rather than "A non-empty result list is displayed".
- "The query scope defined by prior constraints is established" — rather than "The system has registered the user's criteria".
- "The sofa browsing context is active with black leather candidates visible among current results" — rather than "A filtered list of black leather sofas with three seats is displayed".
- "Eligibility criteria for child benefit are displayed" — on a static info page where the content always renders, the concrete phrasing is itself branch-robust.

Output format:
```json
{{
  "instructions": [
    {{"id": 0, "instruction": "..."}},
    {{"id": 1, "instruction": "...", "milestone": "..."}},
    {{"id": 2, "instruction": "...", "milestone": "..."}}
  ]
}}
```

Note: i0 does not have a milestone. Only i_k (k >= 1) has a milestone.

Here are some examples:

**Example 1:**
Original instruction: `Find a hairless dog breed with medium barking.`
I0: `Find me a hairless dog breed.`

Constraints:
```json
{{
  "constraints": [
    {{"id": 1, "description": "Must be a dog breed"}},
    {{"id": 2, "description": "Must be hairless"}},
    {{"id": 3, "description": "Barking level must be medium"}}
  ]
}}
```

```json
{{
  "instructions": [
    {{"id": 0, "instruction": "Find me a hairless dog breed."}},
    {{"id": 1, "instruction": "Now only show breeds with medium barking.",
     "milestone": "The search context for hairless dog breeds is established."}}
  ]
}}
```

**Example 2:**
Original instruction: `I am a DC Comics fan. Could you find three comic books from different series written by the same author as Injustice: Gods Among Us Year One that feature Batman (excluding Injustice itself)? Please provide the links to the homepage of Injustice and these comic books on DC comic's official website.`
I0: `On DC.com, find Injustice: Gods Among Us Year One and other comic books by the same author.`

Constraints:
```json
{{
  "constraints": [
    {{"id": 1, "description": "Comics must be found on DC.com"}},
    {{"id": 2, "description": "The comics must be written by the same author as Injustice: Gods Among Us Year One"}},
    {{"id": 3, "description": "Each of the comics kept after Year One must feature Batman"}},
    {{"id": 4, "description": "The final answer must identify exactly three comic books besides Year One"}},
    {{"id": 5, "description": "The three comic books must be from different series"}},
    {{"id": 6, "description": "Injustice: Gods Among Us Year One itself must be excluded from the three comic books"}},
    {{"id": 7, "description": "The final answer must include a DC.com page link for Injustice: Gods Among Us Year One"}},
    {{"id": 8, "description": "The final answer must include a DC.com page link for each of the three comic books"}}
  ]
}}
```

```json
{{
  "instructions": [
    {{"id": 0, "instruction": "On DC.com, find Injustice: Gods Among Us Year One and other comic books by the same author."}},
    {{"id": 1, "instruction": "Only keep ones that feature Batman.",
     "milestone": "The DC.com search for Injustice: Gods Among Us Year One and other comic books by the same author is established."}},
    {{"id": 2, "instruction": "I want three of them, each from a different series, and not Injustice: Gods Among Us Year One itself.",
     "milestone": "The search for comics by the Year One author is being carried out under the Batman-feature criterion."}},
    {{"id": 3, "instruction": "Give the DC.com page link for Year One and for each of those three comics.",
     "milestone": "Three comics by that author, from different series and excluding Injustice itself, are being identified under the Batman criterion."}}
  ]
}}
```

Before you answer, self-check:
- Ensure each i_k (k >= 1) milestone is a branch-robust semantic state derived only from prior instructions.
- Ensure JSON output follows the schema.

{previous_step4_context}

{retry_reflection_context}

Now design the multi-turn instruction sequence with milestones for the following:

Original instruction: {original_instruction}

Constraints:
{constraints}

I0: {i0}"""


STEP3_RETRY_REFLECTION = """\
## Retry reflection (REQUIRED before producing new design)

You have attempted this task {n_attempts} time(s) before. The full history of prior \
attempts and reviewer critiques is below. Before producing a new design, you MUST \
first reflect on the history and output a diagnosis.

{history_block}

Reflection questions (answer each honestly in the `reflection` field of your JSON output):

1. **Common pattern**: What is the common thread across the reviewer's critiques? Are they \
   pointing to the same i_k, the same milestone type (necessity / branch-neutrality / \
   content-leakage / UI-assumption), or different failures each time?

2. **Dimensions already tried**: Across prior attempts, what dimensions of the design have \
   actually changed (e.g., milestone wording only, number of turns, constraint grouping, \
   which turn a constraint lives in, I0 scope)? Be specific — look at the diffs, not \
   intentions.

3. **Stuck-in-a-loop check**: If prior attempts varied only milestone wording but the \
   critique kept pointing at the same semantic issue, you are in a text-swap loop. Say so \
   explicitly if this is the case.

4. **Next move**: Based on 1-3, what dimension should this attempt change, and why? This \
   is YOUR call — do not default to either "small edit" or "big restructure". Let the \
   pattern you identified drive the choice. It is equally valid to conclude "prior attempts \
   varied structure unnecessarily; the fix is precise milestone rewording" OR "prior \
   attempts shuffled words; the fix is a different constraint grouping".

Output this reflection as a structured field in your JSON (see schema below). Then produce \
the new design per the rules above, with the chosen dimension actually changed.

Output format for retry:
```json
{{
  "reflection": {{
    "common_pattern": "...",
    "dimensions_already_tried": ["..."],
    "in_text_swap_loop": true or false,
    "next_move": "...",
    "rationale": "..."
  }},
  "instructions": [
    {{"id": 0, "instruction": "..."}},
    {{"id": 1, "instruction": "...", "milestone": "..."}}
  ]
}}
```
"""


# ── Step 4: Quality Review ───────────────────────────────────────────────────

STEP4_SYSTEM = (
    "You are an expert at evaluating the quality of multi-turn GUI task designs. "
    "Your job is to judge whether a redesigned multi-turn task feels like a natural "
    "interaction that a real user would have in a real scenario."
)

STEP4_USER = """\
Given the original task instruction and its redesigned multi-turn version (with milestones), \
evaluate whether the redesign is acceptable.

Guidelines:

**Judge the instructions:**
- Is each instruction reasonable — could a user plausibly say this at that point in the task?

**Judge the milestones:**
For each instruction i_k (k >= 1), a milestone defines the semantic stage that triggers its injection.
- Is the milestone derived ONLY from instructions i0 through i_{{k-1}}? It must NOT depend on i_k's content.
- Is the milestone a semantic state (e.g. "search results for headphones are displayed"), \
NOT a specific UI operation (e.g. "clicked the search button")?
- Is the milestone truly necessary — is there any way to correctly complete i0 through i_{{k-1}} \
WITHOUT reaching this state? If so, the milestone is wrong.
- When the milestone state is reached, would it be natural for a user to say i_k?
- Milestones should be phrased as semantic progress states, not tied to a specific UI surface detail.
- Prefer state descriptions that remain valid across branches (including empty-result branches), as long as prior instructions were correctly applied.
- Do not require "non-empty results" unless non-emptiness is explicitly required by prior instructions.

**Important context:**
- This is a benchmark for testing how agents handle dynamic instruction changes. \
The goal is to split a single-turn task into multiple turns.
- It is expected and acceptable that constraints are introduced gradually across turns, \
even if a real user might sometimes state them all at once.
- A split is acceptable as long as each instruction is reasonable on its own — \
it does NOT need to be the most efficient way a user would interact.
- Do NOT reject because a single turn could have included more constraints. \
Gradual refinement is the intended design.

**Reject if:**
- An instruction makes no sense at that point (e.g. asking to filter before any results exist)
- The split is completely degenerate (e.g. splitting a single action into sub-word fragments)
- A milestone depends on the content of i_k rather than preceding instructions
- A milestone describes a UI operation instead of a semantic state
- A milestone is NOT necessary — a correct trajectory could bypass it

{previous_step4_context}

{override_context_block}

Output format:
```json
{{
  "passed": true/false,
  "worth_retry": null,
  "retry_from_step": null,
  "reason": "..."
}}
```

**When "passed" is false, provide two additional decisions:**

1. **worth_retry** (true/false): Should we retry this task?
   - true: The failure is fixable through redesign (e.g., milestone logic error, i0 boundary issue, instructions need restructuring)
   - false: The failure indicates the task is unsuitable for multi-turn decomposition or has infrastructure issues
2. **retry_from_step** (only when worth_retry=true): Which step should we redo?
   - "step2": The issue is in the I0 design or its clarity (unclear i0 leads to bad milestone boundaries). Redo i0 with feedback.
   - "step3": The issue is in multi-turn arrangement or milestones (bad milestone logic, poor turn boundaries). Redo instructions with feedback.

**Consistency requirement:**
- Your `reason` and `retry_from_step` must describe the same root cause.
- If the problem is that a foundational constraint belongs in I0, or that the current I0 is too vague/narrow, choose `retry_from_step: "step2"`.
- If the problem is only how later instructions or milestones are arranged, choose `retry_from_step: "step3"`.
- Do NOT write a reason that points to an I0 problem and then set `retry_from_step` to "step3".
- Keep `reason` focused on the chosen fix path; do not mention an alternative retry path unless it is the one you selected.

Example outputs:
- Acceptance: `{{"passed": true, "worth_retry": null, "retry_from_step": null, "reason": "Clear and valid design"}}`
- Fixable at Step 3: `{{"passed": false, "worth_retry": true, "retry_from_step": "step3", "reason": "Milestone depends on i_1 content; restructure instructions and milestones"}}`
- Fixable at Step 2: `{{"passed": false, "worth_retry": true, "retry_from_step": "step2", "reason": "I0 is too vague; clarify the initial request"}}`
- Unfixable: `{{"passed": false, "worth_retry": false, "retry_from_step": null, "reason": "Task naturally fits single-turn; no natural multi-turn split"}}`

Now evaluate the following:

Original instruction: {original_instruction}

Multi-turn redesign:
{redesign}"""


# ── Step 3c: Sync related constraint wording ────────────────────────────────

STEP3C_USER = """\
This is the whole task:

{instructions}

The user's requirements change during the task. A revision in a later turn may override an earlier requirement, or select according to a condition, and that decides some requirements of the final goal.

There are several constraints describing the requirements of the final goal. Each constraint is independent of the others, and together they describe the final goal.

{constraints}

Because the user's requirements changed, one constraint has already been updated correctly:
- id: {target_id}
- before: {keep}
- now: {now}

Other constraints may still contain related wording that was not updated. Find those descriptions and change them so they agree with this updated constraint.

Return JSON:
{{"edits": [{{"id": <int>, "description": "<updated sentence>"}}]}}

Example:

This is the whole task:
i0: Find a certified refurbished iPad Air in blue with 256GB storage.
i1: Pick the cheapest one.
i2: Change the storage to 512GB. Keep the blue color and the cheapest pick.

Constraints:
1. Color must be blue.
2. Storage capacity must be 512GB.
3. The selected product must be the cheapest among certified refurbished iPad Air models with 256GB storage in blue.

One constraint has already been updated correctly:
- id: 2
- before: Storage capacity must be 256GB.
- now: Storage capacity must be 512GB.

{{"edits": [{{"id": 3, "description": "The selected product must be the cheapest among certified refurbished iPad Air models with 512GB storage in blue."}}]}}

Now do this for the task above.
"""

STEP3C_CONDITIONAL_USER = """\
This is the whole task:

{instructions}

The user's requirements change during the task. A revision in a later turn may override an earlier requirement, or select according to a condition, and that decides some requirements of the final goal.

There are several constraints describing the requirements of the final goal. Each constraint is independent of the others, and together they describe the final goal.

{constraints}

Because the user's requirements changed, one constraint is decided by a condition. It has two outcomes:
- id: {target_id}
- default: {keep}
- alternative: {change_to}
- condition: {condition}

Other constraints may still restate the default outcome only. For each of those, write two sentences: one for the default outcome, one for the alternative. Do not merge them into one sentence with if.

Return JSON:
{{"splits": [{{"id": <int>, "default": "<sentence>", "alternative": "<sentence>"}}]}}

Example:

This is the whole task:
i0: Find a certified refurbished iPad Air in blue with 256GB storage.
i1: I still prefer blue. But if fewer than 1 blue option is listed, Space Gray is acceptable.
i2: Pick the cheapest one under that color rule.

Constraints:
1. Color must be blue.
2. Storage capacity must be 256GB.
3. The selected product must be the cheapest among certified refurbished iPad Air models with 256GB storage in blue.

One constraint is decided by a condition:
- id: 1
- default: Color must be blue.
- alternative: Color may be Space Gray.
- condition: fewer than 1 blue certified refurbished iPad Air with 256GB is listed.

{{"splits": [{{"id": 3, "default": "The selected product must be the cheapest among certified refurbished iPad Air models with 256GB storage in blue.", "alternative": "The selected product must be the cheapest among certified refurbished iPad Air models with 256GB storage in Space Gray."}}]}}

Now do this for the task above.
"""


# ── Step 5: Constraint Binding ──────────────────────────────────────────────

STEP5_SYSTEM = (
    "You are an expert at analyzing multi-turn task designs and mapping constraints "
    "to instructions. Your job is to determine which instruction introduces or covers "
    "each constraint, and to audit the coverage quality."
)

STEP5_USER = """\
You are given:
1. A list of constraints extracted from a task
2. The I0 (initial instruction)
3. The full multi-turn instruction sequence (including I0)

Your job is to bind each constraint to the instruction that introduces or covers it, \
and audit the quality of constraint coverage.

{override_context_block}

**Rules:**
- Each constraint must be mapped to exactly one instruction (the one that most directly \
introduces or addresses it)
- Use instruction IDs: "i0" for the initial instruction, "i1", "i2", etc. for subsequent ones
- Rate your confidence for each mapping: "high", "medium", or "low"
- Provide a brief reason for each mapping

**Audit checks:**
- Flag any constraint not clearly covered by any instruction
- Flag any constraint that seems poorly matched to its assigned instruction
- Flag if the overall coverage is incomplete (constraints missing from the instruction sequence)
- If override context is provided, allow the target constraint to be satisfied via an explicit target->replacement override instruction.

Constraints:
{constraints}

I0: {i0}

Full instruction sequence:
{instructions}

Output format:
```json
{{
  "mapping": [
    {{"constraint_id": 1, "instruction_id": "i0", "confidence": "high", "reason": "I0 explicitly mentions ..."}},
    {{"constraint_id": 2, "instruction_id": "i1", "confidence": "medium", "reason": "i1 addresses this through ..."}}
  ],
  "issues": [
    "constraint 3 is not clearly covered by any instruction"
  ],
  "passed": true,
  "retry_from_step": null
}}
```

Set "passed" to false if there are serious coverage gaps (constraints with no matching instruction, \
or multiple low-confidence mappings). Minor issues (e.g., one medium-confidence mapping) should \
still pass.

**When "passed" is false, set "retry_from_step":**
- "step2": The issue originates from I0 design — I0 is too vague or too narrow, causing \
downstream instructions to miss constraints or overlap awkwardly. Redo from I0 design.
- "step3": The issue is in how constraints were arranged across instructions — poor distribution, \
missing coverage, or redundant overlap. I0 itself is fine; redo the multi-turn arrangement.

Now analyze the constraint binding."""


# ── Replacement Pairs: Constraint Classification ─────────────────────────────

CLASSIFY_SYSTEM = (
    "You are an expert at analyzing task constraints for web-based GUI tasks. "
    "Your job is to classify each constraint as either CORE or PREFERENCE."
)

CLASSIFY_USER = """\
You are given a list of constraints from tasks in the "{scene}" scene.

Classify each constraint as:
- **CORE**: Defines the fundamental goal of the task — the product type, search subject, \
destination, platform, or action that makes the task what it is. \
If you changed this constraint, the task would become a completely different task.
- **PREFERENCE**: A filtering, sorting, or preference condition that refines results — \
color, price range, size, rating, sort order, brand preference, time range, etc. \
The user could plausibly change their mind about this during task execution.

Constraints:
{constraints}

Output format:
```json
{{
  "classifications": [
    {{"id": "...", "text": "...", "label": "CORE", "reason": "..."}},
    {{"id": "...", "text": "...", "label": "PREFERENCE", "reason": "..."}}
  ]
}}
```

Classify every constraint. Be conservative with PREFERENCE — when in doubt, mark as CORE."""


# ── Replacement Pairs: Preference Pairing ─────────────────────────────────────

PAIRING_SYSTEM = (
    "You are an expert at identifying constraint dimensions and generating "
    "meaningful replacement pairs for preference constraints in web tasks."
)

PAIRING_USER = """\
You are given PREFERENCE constraints from tasks in the "{scene}" scene. \
Group them by dimension, then create replacement pairs.

PREFERENCE constraints:
{constraints}

Rules:
1. Group constraints by dimension (e.g., "color", "price_range", "sort_order", "size", "rating").
2. Within each dimension, pair constraints that have different values. \
If a dimension has only one value, generate a plausible alternative value for that domain.
3. Mark "symmetric": true if both values are equally likely preferences (e.g., black vs blue). \
Mark "symmetric": false if one value is a more common default (e.g., "under $100" vs "under $200").
4. Each constraint should appear in at most one pair. Prefer pairing values from different tasks.

Output format:
```json
{{
  "pairs": [
    {{
      "dimension": "color",
      "constraint_a": {{"text": "Color must be black", "source_task": "task_id or null", "generated": false}},
      "constraint_b": {{"text": "Color must be blue", "source_task": "task_id or null", "generated": true}},
      "symmetric": true
    }}
  ]
}}
```

Generate all meaningful pairs. Quality over quantity — each pair should represent a \
realistic preference change a user might make."""


# ── Step 3b (Override): Instruction Generation ──────────────────────────────

STEP3_OVERRIDE_SYSTEM = (
    "You are an expert at designing natural multi-turn user interactions for GUI tasks. "
    "Arrange extracted constraints into a realistic timeline, with a milestone "
    "for each turn after I0. Prefer a row from the candidate list as a reference. "
    "If that list is empty, propose a replacement for one existing constraint. "
    "You decide which preference to keep and what it changes to. "
    "Follow the override_mode section exactly."
)

STEP3_OVERRIDE_UNCONDITIONAL_BLOCK = """\
## Override mode: unconditional

Use a candidate row if any exist; otherwise use the empty-candidate rules. \
Write one `type="override"` turn that switches the current preference (`keep`) \
to `change_to`. No if-clause. Other turns are `type="additive"`.

Candidate `pair_values` are a reference on one dimension, not gold and not a \
direction label. Write `keep` and `change_to` in the task's own wording. They \
may differ from the pair texts and from the anchor constraint.

When using a candidate, `selected` has:
- `target_constraint_id` / `replacement_pair_id` copied from that row
- `keep`: the preference that stays in force until the override
- `change_to`: the preference after the override (a different concrete value)

Good: "Actually, change the color to blue instead. Keep the other criteria the same."
Bad: "Color doesn't matter anymore." / "If it's expensive, blue is fine."
Bad: keep and change_to that are the same preference.

Example with candidates (duration 6–12 min → 12–18 min). Copy real candidate \
ids; do not reuse the example ids.

```json
{
  "enable_override": true,
  "selected": {
    "target_constraint_id": 3,
    "replacement_pair_id": "from_a_candidate_row",
    "keep": "The talk must last 6 to 12 minutes.",
    "change_to": "The talk must last 12 to 18 minutes."
  },
  "instructions": [
    {"id": 0, "instruction": "Find me a TED talk on the topic of robots that lasts 6 to 12 minutes."},
    {"id": 1, "instruction": "Actually, change the duration to 12 to 18 minutes instead. Keep the topic the same, and pick the most viewed one.",
     "milestone": "The search context for TED talks on the topic of robots with a duration preference is established.",
     "type": "override"}
  ]
}
```

Empty-candidate example (IMDb).
Good: "Actually, use 2023 instead of 2024. Keep the matching China/US theatrical dates."
Bad: "Find TV shows instead." / "Any year is fine."

```json
{
  "enable_override": true,
  "selected": {
    "target_constraint_id": 2,
    "replacement_pair_id": null,
    "keep": "The movies must have premiered in 2024.",
    "change_to": "The movies must have premiered in 2023.",
    "pair_dimension": "year"
  },
  "instructions": [
    {"id": 0, "instruction": "On IMDb, search for movies premiered in 2024 that have matching theatrical release dates in mainland China and the United States."},
    {"id": 1, "instruction": "Actually, use 2023 instead of 2024. Keep the matching China/US theatrical dates.",
     "milestone": "The IMDb search for 2024 movies with matching China/US theatrical dates is established.",
     "type": "override"}
  ]
}
```
"""

STEP3_OVERRIDE_CONDITIONAL_BLOCK = """\
## Override mode: conditional

Use a candidate row if any exist; otherwise use the empty-candidate rules. \
The default stays `keep`. `change_to` is accepted only when a \
page-observable condition holds.

A conditional override has two axes:

- Replacement dimension — the preference that changes between `keep` and \
`change_to` (what the user is willing to revise: color, ticket type, \
distance, a rating bar, how many items, and so on).
- Condition dimension (`quantity`) — what is read from the page to decide \
whether the alternative is allowed. Choose it on its own. It may be the \
same as the replacement dimension, or different from it.

Condition dimension is one of:
- price  — a listed price or a price gap (include a currency)
- rating — stars, rank, or a similar score
- time   — duration, clock time, or recency on the page
- count  — how many results match a preference on the page

Choose a quantity this website can actually show. Write the condition only \
(not the replacement pair) in this form:

  (QUANTITY) ( > or < ) (NUMBER)

Those two axes combine in two natural ways:

1. Advantage / cost — The condition compares the default outcome and the \
alternative outcome on price, rating, or time (for example, how much more \
the preferred option costs than the fallback). The replacement dimension \
is usually a categorical preference; the condition dimension is the \
tradeoff being weighed.

2. Existence — The condition is that the results required under the default \
preference are not there: one matching result, or the K results the task \
asked for. The user then relaxes the preference, or relaxes how many are \
required. `count` is the usual quantity here.

Prefer a condition a real user would say after seeing results: a tradeoff \
they would weigh, or a shortfall they would notice. The numbers should come \
from the task and the page, so the reason for changing one's mind is easy to \
follow. Conditions that almost never fire, almost always fire, or only restate \
that nothing matched, rarely sound like real speech.

Do not restate the replacement pair as the condition itself.

Fill NUMBER from the task and the page.

The revision must keep `keep` as the default, state the condition with the \
concrete number, and accept `change_to` only if the condition holds. \
`type="conditional_override"`. Also set `condition_text` on that turn: one \
predicate sentence using the same number and comparison.

Candidate `pair_values` are a reference, not gold and not a direction label. \
Write `keep` and `change_to` in the task's own wording.

Milestone: the default preference has been applied, including empty or short lists.

If you cannot form a real preference change (from a candidate or a proposal), \
or no QUANTITY is observable on this page, return `"enable_override": false`. \
Do not write a flat override.

`selected` fields:
- `target_constraint_id`
- `replacement_pair_id` copied from a candidate row, or null if proposing
- `keep` / `change_to` (required; different preferences)
- if proposing: `pair_dimension`
- `quantity`: price | rating | time | count
- `operator`: > or <
- `threshold`: the NUMBER
- `unit`: e.g. USD, or ""

Copy real candidate ids in your output; do not reuse the example ids.

Examples (keep → change_to, condition):

Advantage / cost:
- Replacement: non-red-eye ↔ red-eye. Condition: price gap.
  keep="Selected flight is non-red-eye", change_to="Selected flight is red-eye",
  quantity=price, operator=">", threshold=500, unit="RMB"
  Good: "I prefer no red-eye flights. But if non-red-eye flights are at least \
500 RMB more expensive than the cheapest red-eye, red-eye is acceptable."
- Replacement: blue ↔ Space Gray. Condition: price gap.
  keep="Color must be blue", change_to="Color must be Space Gray",
  quantity=price, operator=">", threshold=50, unit="USD"
  Good: "I still want blue. But if the cheapest blue is more than $50 above \
the cheapest Space Gray, Space Gray is fine."

Existence:
- Replacement: must feature Batman ↔ need not. Condition: count.
  keep="Comics must feature Batman", change_to="Comics need not feature Batman",
  quantity=count, operator="<", threshold=2
  Good: "I prefer keeping the Batman requirement. If fewer than two such \
comics are listed, drop the Batman filter."
- Replacement: need five → accept three. Condition: count.
  keep="Find five matching items", change_to="Three matching items are enough",
  quantity=count, operator="<", threshold=5
  Good: "I want five. If fewer than five are listed under that filter, three \
is fine."

Unnatural (users do not talk this way):
- "If there are fewer than 3 blue options, black is fine" when they only need \
one item. The task never asked for three; the 3 is invented, so the cutoff \
does not explain a real shortfall the user cares about.
- "If the cheapest blue is above $500, accept Space Gray." Crossing an \
absolute dollar amount does not say why gray becomes acceptable — blue may \
still be the better deal than gray, and a real user would compare the two \
colors (or notice blue is missing), not switch color at a round price tag.

Full timeline:

```json
{
  "enable_override": true,
  "selected": {
    "target_constraint_id": 3,
    "replacement_pair_id": "from_a_candidate_row",
    "keep": "Selected flight is non-red-eye.",
    "change_to": "Selected flight is red-eye.",
    "quantity": "price",
    "operator": ">",
    "threshold": 500,
    "unit": "RMB"
  },
  "instructions": [
    {"id": 0, "instruction": "Search Ctrip for flights from Beijing to Shanghai departing the day after tomorrow."},
    {"id": 1, "instruction": "Only keep non-red-eye options.",
     "milestone": "A Beijing–Shanghai search for the day after tomorrow is underway with non-red-eye as the default.",
     "type": "additive"},
    {"id": 2, "instruction": "I prefer no red-eye flights. But if non-red-eye flights are at least 500 RMB more expensive than the cheapest red-eye, red-eye is acceptable.",
     "milestone": "The flight search is running under the non-red-eye preference, with a priced red-eye fallback if the gap is large enough.",
     "type": "conditional_override",
     "condition_text": "The cheapest non-red-eye Beijing–Shanghai flight for the day after tomorrow is more than 500 RMB above the cheapest red-eye on Ctrip."},
    {"id": 3, "instruction": "Pick one matching flight and tell me the flight number and price.",
     "milestone": "A flight on the active red-eye branch has been identified with its number and price.",
     "type": "additive"}
  ]
}
```
"""

STEP3_OVERRIDE_USER = """\
Given the original task, constraints, I0, website, and override candidates, \
write a multi-turn instruction sequence with milestones.

**I0:** already provided — do not modify it.

**Later turns:**
- Sound like a user refining the request after seeing progress.
- Group information naturally; do not split one idea across turns.
- Keep composite AND/OR constraints intact.
- If `override_candidates` is non-empty: pick exactly one row and copy \
`target_constraint_id` / `replacement_pair_id` into `selected`. Do not invent a pair. \
Write `keep` and `change_to` yourself. `pair_values` are a reference only; \
do not copy them unless that wording already fits the task.
- If `override_candidates` is empty: pick one constraint as the preference the \
user could change (not the task subject). Write `keep` and `change_to` on \
the same dimension. `change_to` must be a different concrete value this \
website can actually show. Set `replacement_pair_id` to null. Do not change \
the subject (movies stay movies). Do not invent a price on a page with no prices.
- `"enable_override": false` if no such preference exists, or if keep and \
change_to would be the same preference.
- Do not rewrite a replacement into "any is fine" / "remove the filter" / "don't care".

{mode_block}

**Style:** natural user voice ("Show me…", "Only show…", "Now set…").

## Milestones

Each i_k (k >= 1) needs a milestone: a branch-robust semantic state after \
correctly completing i0..i_{{k-1}}. User-visible progress, not UI clicks. \
Derive it only from prior turns; never leak i_k. It must remain valid on \
empty-result branches. Prefer an earlier reachable state over restating i0's \
final goal.

Override insertion timing:
{override_timing_guidance}

{previous_step4_context}

Website: {website}

Original instruction: {original_instruction}

Constraints:
{constraints}

I0: {i0}

Override candidates (each row is one seed constraint plus a reference pair; \
`pair_values` are unlabeled; `pair_dimension` is the pair's axis, not a \
switching quantity):
{override_candidates}

Notes:
- i0 has no milestone.
- Turns after i0 need `type`.

Output JSON only.
"""
