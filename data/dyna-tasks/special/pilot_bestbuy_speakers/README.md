# Pilot: BestBuy Bluetooth Speakers

**Batch0 seed #2** (`om2w_87f4c512`, see `seeds/batch0_om2w.jsonl`). Domain: `bestbuy.com` (not IGN).

Still under `special/` until quality gates pass. `tasks.jsonl` now has three drafts:

| instance_id | type | revision idea |
|-------------|------|----------------|
| `..._additive` | additive | i0 find speakers → i1 add On Sale + &lt;$50 + name one |
| `..._override` | unconditional override | i0 under $100 → i1 replace with On Sale + &lt;$50 |
| `..._conditional` | conditional override | i0 under $50 → if On Sale facet exists, also require On Sale |

On Sale facet is expected present on BestBuy (condition → alternative branch).

Oracle note (2026-09-11): homepage hits an international country gate unless `?intl=nosplash` (or equivalent cookie). Search for `bluetooth wireless speakers` works after US entry; `On Sale` and price facets like `$25 - $49.99` are present.

DO NOT use without `intl=nosplash` (or an explicit United States click) in this runner environment.
