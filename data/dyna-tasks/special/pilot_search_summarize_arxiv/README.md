# Pilot: arXiv web-agent papers / Jan–Sep 2025

Formal4 **M2W2 seed #4** (`search_summarize_arxiv`), domain `arxiv.org`. Recorded in `seeds/batch0_m2w2.jsonl`. In `batches/batch0/tasks.jsonl`.

## Base task (upstream)

Find 10 papers related to web agents published on arXiv between January and September of last year (first-version submission time). For each: paper name, PDF link, first author name and affiliation, first-version submission time.

Constructed in September 2026, so “last year” is frozen as **January–September 2025**.

## Construction

Three variants share the same i0 (web-agent papers on arXiv). The date window is added later. Override introduces Jan–Sep 2024 in dialogue only, then replaces it with Jan–Sep 2025. Conditional adds 2025, then may switch to 2024 if too few. Quantity and catalog fields stay later additive. Quantity is not the override dimension.

| variant | i0 | i1 | i2 | i3 |
|---------|----|----|----|----|
| additive | web-agent papers on arXiv | add first version Jan–Sep 2025 | ten + name, PDF, first author, affiliation, submission time | — |
| override | same i0 | add first version Jan–Sep 2024 | 2025, not 2024 | same as additive i2 |
| conditional | same i0 | add first version Jan–Sep 2025 | if fewer than 5 such papers listed, switch to Jan–Sep 2024 | same as additive i2 |

Milestones are preceding-goal stages, not the next revision:

| variant | m1 (inject i1) | m2 (inject i2) | m3 (inject i3) |
|---------|----------------|----------------|----------------|
| additive | arXiv search for web-agent papers is established | that search is under first-version Jan–Sep 2025 | — |
| override | same as additive m1 | that search is under first-version Jan–Sep 2024 | that search is under Jan–Sep 2025 rather than 2024 |
| conditional | same as additive m1 | same as additive m2 | working search follows the date branch (keep 2025 after the default count is known, or move to 2024) |

`start_url`: `https://arxiv.org/search/`  
`max_steps`: 48

## Oracle note

BrowserGym probe 2026-09-17 passed on `/` and `/search/` (no Access Denied). Search lists title, authors, submitted date, and PDF; affiliation usually needs an abs page. Conditional “fewer than 5” is expected **false** (default keep 2025) if the live catalog has many web-agent papers in that window.

DeepSeek-flash obskeep5 20260917: official 0/3, mean 0.556 (additive 0.89 submitted; conditional 0.78 submitted; override max_steps). Site reachable. Remaining misses are agent.
