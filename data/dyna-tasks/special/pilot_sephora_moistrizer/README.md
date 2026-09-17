# Pilot: Sephora moisturizers / Squalane · Ceramide · Dimethicone

Formal4 **M2W2 seed #4 candidate** (`sephora_moistrizer`), domain `sephora.com`. Not yet in `seeds/batch0_m2w2.jsonl`. Not in the OM2W Batch0 pack.

## Base task (upstream)

Find three moisturizers on Sephora that contain Squalane and Ceramide, but do not contain Dimethicone. Provide their purchase links on Sephora, brand names and prices.

## Construction

Three variants share the same i0 (moisturizers on Sephora). Ingredient *must-contain* (Squalane and Ceramide) is added later. The exclusion is the one override/conditional turn: additive adds no-Dimethicone; override adds no-Parabens then replaces it with no-Dimethicone (`es_048`); conditional adds no-Dimethicone then may drop it. Quantity / links / brand / price stay later additive. Quantity is not the override dimension.

Override i2 introduces Parabens in the dialogue only; the scored exclusion is the post-override Dimethicone requirement (otherwise AND would still require Parabens).

| variant | i0 | i1 | i2 | i3 | i4 |
|---------|----|----|----|----|----|
| additive | moisturizers on Sephora | add Squalane and Ceramide | add no Dimethicone | three + Sephora link, brand, price | — |
| override | same i0 | same as additive i1 | add no Parabens | Dimethicone, not Parabens | same as additive i3 |
| conditional | same i0 | same as additive i1 | add no Dimethicone | if fewer than 2 such moisturizers listed, drop the Dimethicone exclusion | same as additive i3 |

Milestones are preceding-goal stages, not the next revision:

| variant | m1 (inject i1) | m2 (inject i2) | m3 (inject i3) | m4 (inject i4) |
|---------|----------------|----------------|----------------|----------------|
| additive | Sephora search for moisturizers is established | that search is under Squalane and Ceramide | that search is under Squalane, Ceramide, and no-Dimethicone | — |
| override | same as additive m1 | same as additive m2 | that search is under Squalane, Ceramide, and no-Parabens | that search is under no-Dimethicone rather than no-Parabens |
| conditional | same as additive m1 | same as additive m2 | same as additive m3 | working search follows the Dimethicone branch (keep the exclusion after the default count is known, or drop it) |

`start_url`: `https://www.sephora.com/`  
`max_steps`: 42

## Oracle note

Not probed in BrowserGym for this draft. Runner docs record Sephora Access Denied before stealth, then operable after. Ingredient constraints usually need a product page, not only a category list. Conditional “fewer than 2” is written like DC (expected **false** / default keep Dimethicone if the catalog has several matches); recheck on the live shop before accepting the seed.
