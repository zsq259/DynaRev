# Formal4 seeds

- `batch0_om2w.jsonl`：首批 OM2W seed 4 个齐（Zelda #1 drafted、BestBuy #2 drafted、AKC #3 accepted、TED #4 accepted）
- `om2w_candidates.jsonl` / `m2w2_candidates.jsonl`：筛选池（可选）
- `rejected.jsonl`：淘汰原因（DOJ / Mayo；详见 [`docs/formal4_failure_cases.md`](../../../docs/formal4_failure_cases.md) B1–B2）

字段约定见上级 `README.md`。写入 `selected` 后，再到 `batches/batch0/` 生成三条变体任务。
