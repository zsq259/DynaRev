# Pilot: Cleveland Museum Chinese objects

Formal4 **M2W2 seed #1** (`cleveland_museum`), domain `clevelandart.org`. Recorded in `seeds/batch0_m2w2.jsonl`. In `batches/batch0/tasks.jsonl`.

## Base task (upstream)

Identify five ancient Chinese objects from the Cleveland Museum of Art collection dated entirely to 2000 BCE or earlier; for each, name / period / date range / dimensions / location (or not on display).

## Construction

Multi-turn (not `i0+i1`). Override and conditional each occupy **one** turn; the catalog-detail request is a later additive turn. Date pair reuses `replacement_pairs.json` `mic_051` (2000 BCE or earlier ↔ 1000 BCE or later).

| variant | i0 | i1 | i2 |
|---------|----|----|----|
| additive | Chinese objects in the CMA collection | add date entirely ≤ 2000 BCE | pick five + catalog fields |
| override | Chinese objects dated 1000 BCE or later | replace date → entirely ≤ 2000 BCE | pick five + catalog fields |
| conditional | Chinese objects dated entirely ≤ 2000 BCE | if fewer than 5 listed, switch to 1000 BCE or later | pick five + catalog fields |

Milestones are preceding-goal stages, not the next revision:

| variant | m1 (inject i1) | m2 (inject i2) |
|---------|----------------|----------------|
| additive | CMA search for Chinese objects is established | Chinese-object search is under the entirely-2000-BCE-or-earlier criterion |
| override | CMA search for Chinese objects dated 1000 BCE or later is established | same as additive m2 |
| conditional | CMA search for Chinese objects dated entirely to 2000 BCE or earlier is established | working search follows the date branch (keep 2000 BCE after the default count is known, or move to 1000 BCE or later) |

`start_url`: `https://www.clevelandart.org/art/collection/search`

## Oracle note (2026-09-16, Open Access API, not the live GUI)

`department=Chinese Art & created_before=-2000` returned **15** hits. The API date filter is looser than “dated entirely to 2000 BCE or earlier” (e.g. Western Zhou “1000s BCE” and Qijia 2000–1700 BCE still appear). On the first page, several records *do* satisfy the strict cutoff (Hongshan amulet, Majiayao jars, Liangzhu bi, etc.), so the conditional’s “fewer than 5” is expected **false** (default branch), analogous to TED Robots ∩ 12–18. Recheck on the public collection search before accepting the seed.

Chinese culture / department facets exist; “entirely ≤ 2000 BCE” may still need reading object dates, not a single facet value.
