# Pilot: HuggingFace English VQA datasets

Formal4 **M2W2 seed #2** (`hf_vqa_dataset`), domain `huggingface.co`. Recorded in `seeds/batch0_m2w2.jsonl`. In `batches/batch0/tasks.jsonl`.

## Base task (upstream)

Find two English visual question answering datasets on HuggingFace with more than 1M rows and Croissant metadata, released by a US-university research group whose HuggingFace profile states the university affiliation; give dataset and organization profile links.

## Construction

Multi-turn. **Revision count is not capped at two.** Override and conditional occupy **one** turn (`i1`, size pair `mic_057`); later turns stay additive (Croissant, then US-university source / affiliation / two links). English VQA on HuggingFace stays in i0 across variants.

| variant | i0 | i1 | i2 | i3 |
|---------|----|----|----|----|
| additive | English VQA datasets on HuggingFace | add >1 million rows | add Croissant metadata | two US-university datasets + affiliation stated + links |
| override | English VQA datasets with >500,000 rows | replace size → >1 million, not 500,000 | add Croissant metadata | same as additive i3 |
| conditional | English VQA datasets with >1 million rows | if fewer than 2 listed, switch to >500,000 rows | add Croissant metadata | same as additive i3 |

Milestones are preceding-goal stages, not the next revision:

| variant | m1 (inject i1) | m2 (inject i2) | m3 (inject i3) |
|---------|----------------|----------------|----------------|
| additive | HuggingFace search for English VQA datasets is established | that search is under the >1-million-rows criterion | Croissant metadata is required on that search |
| override | HuggingFace search for English VQA datasets with >500,000 rows is established | same as additive m2 | same as additive m3 |
| conditional | HuggingFace search for English VQA datasets with >1 million rows is established | working search follows the size branch (keep 1M after the default count is known, or move to 500k) | Croissant is required on that working search |

`start_url`: `https://huggingface.co/datasets`  
`max_steps`: 42

## Oracle note (2026-09-16)

BrowserGym (runner `DynaBrowserEnv`) opened the datasets catalog: title `Datasets – Hugging Face`; Tasks / Size(rows) / Format facets present; no block page.

HuggingFace has a Visual Question Answering task facet, a language facet, and a row-count size slider. Croissant is typically `library:mlcroissant` under Other/libraries, not a top-level checkbox. US university is **not** a facet — it requires opening organization profiles.

API sample of the VQA tag: **12** datasets in a 1M+ size bucket (several English), so the conditional “fewer than 2” on English VQA × >1M is expected **false** (default branch), analogous to Cleveland / TED. The original’s US-university ∩ 1M ∩ Croissant slice is much thinner on the same sample (NYU/UCLA orgs appear, but the tagged sets in the first 100 were below 1M). i3 may need paging beyond the size facet; if two full matches cannot be found, treat as fail-and-drop rather than rewriting the seed around a different site.
