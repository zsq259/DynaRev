# Formal4 seeds

- `batch0_om2w.jsonl`：首批 OM2W seed 4 个齐（Zelda #1 drafted、BestBuy #2 drafted、AKC #3 accepted、TED #4 accepted）
- `batch0_m2w2.jsonl`：首批 M2W2 seed 4 个齐（Cleveland #1 drafted、HF VQA #2 drafted、DC Injustice #3 drafted、arXiv web-agent papers #4 drafted）
- `om2w_candidates.jsonl` / `m2w2_candidates.jsonl`：筛选池（可选）
- `rejected.jsonl`：淘汰原因（DOJ / Mayo / Coursera / Sephora；详见 [`docs/formal4_failure_cases.md`](../../../docs/formal4_failure_cases.md) B1–B4）

字段约定见上级 `README.md`。写入 `selected` 后，再到 `batches/` 生成三条变体任务。
