# Pilot: DC Comics Injustice / Batman

Formal4 **M2W2 seed #3** (`dc_comics_injustice`), domain `dc.com`. Recorded in `seeds/batch0_m2w2.jsonl`. In `batches/batch0/tasks.jsonl`.

## Base task (upstream)

Find three comic books from different series by the same author as *Injustice: Gods Among Us Year One*, featuring Batman, excluding Injustice itself; give DC.com links for Year One and those comics.

## Construction

Three variants share the same i0 (Year One + other comics by that author). Character preference is added later; override replaces Superman with Batman on a later turn rather than putting Superman into i0. Conditional adds Batman, then may drop it if too few. Quantity is not the override dimension. I0 may mention Year One; it is not a scored “search must start from this page” constraint.

Formal3 converted jsonl is not on this machine; constraints align with the seven atoms recorded in the evaluator docs (author / series / Batman / exclude Year One / links). Override i1 introduces Superman in the dialogue only; the scored character constraint is the post-override Batman requirement (otherwise AND would still require Superman).

| variant | i0 | i1 | i2 | i3 | i4 |
|---------|----|----|----|----|----|
| additive | Year One on DC.com and other comics by that author | add Batman | three, different series, not Injustice itself | DC.com links for Year One and the three | — |
| override | same i0 | add Superman | replace character → Batman, not Superman | same as additive i2 | same as additive i3 |
| conditional | same i0 | add Batman | if fewer than 2 such comics listed, drop the Batman filter | same as additive i2 | same as additive i3 |

Milestones are preceding-goal stages, not the next revision:

| variant | m1 (inject i1) | m2 (inject i2) | m3 (inject i3) | m4 (inject i4) |
|---------|----------------|----------------|----------------|----------------|
| additive | DC.com search for Year One and other comics by that author is established | that search is under the Batman-feature criterion | three different-series, non-Injustice comics are being identified under Batman | — |
| override | same as additive m1 | that search is under the Superman-feature criterion | that search is under Batman rather than Superman | three different-series, non-Injustice comics are being identified under Batman |
| conditional | same as additive m1 | that search is under the Batman-feature criterion | working search follows the character branch (keep Batman after the default count is known, or drop Batman) | three different-series, non-Injustice comics are being identified on that branch |

`start_url`: `https://www.dc.com/comics`  
`max_steps`: 42

## Oracle note (2026-09-16)

BrowserGym opened `https://www.dc.com/` and `https://www.dc.com/comics`: titles `DC.com - The Official Home of DC` / `Comics | DC`; CHARACTERS, COMICS, search; latest comics list; no block page.

Character is not a single guaranteed facet on every listing; author + Batman usually means opening a title/character/search. Formal3 eval walkthrough found three Tom Taylor Batman books (including *One Bad Day* / *Detective Comics*), so conditional “fewer than 2 besides Year One” is expected **false** (default Batman), analogous to Cleveland / TED. Recheck on the live comics catalog before accepting the seed.
