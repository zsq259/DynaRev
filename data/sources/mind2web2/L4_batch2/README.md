# L4 Batch 2 (2026-04-28)

第二批 M2W2 任务生成。从手挑过的两组候选清单（L2 时期的 `m2w2_pipeline_input.csv` 22 条 + `m2w2_remaining_pipeline_input.csv` 22 条，已删除）合并去重并排除 L3 seed batch 的 17 条，得到 39 条作为输入。

## 输入

`m2w2_batch2_pipeline_input.csv` — 39 条。

## Pipeline 输出

- `logs/pipeline/m2w2_batch2_20260428/tasks.jsonl` — 35 条（其中 1 条 `m2w2_blazer_allergy` 是 export bug 误带进来的，Step 5 binding 实际失败）
- `logs/pipeline/m2w2_batch2_20260428/report.json` — 实际通过 34 条；filtered 5 条（`blazer_allergy` Step5、`las_vegas_experience` Step4、`manhattan_restaurants` Step2 403、`restaurant_tuesday` Step2 403、`tokyo_filming_location` Step4）

## 正式数据集

`data/dyna-tasks/v1/m2w2_batch2_20260428.jsonl` — 34 条，已：
- 排除 blazer_allergy
- 改用 `dyna_<full_task_id>` 唯一 instance_id（避免与 L3 seed batch 同样踩 instance_id 截断碰撞）

manifest：`data/dyna-tasks/v1/m2w2_batch2_20260428.manifest.json`

## Agent 执行

`logs/runs/m2w2_batch2_s50_20260428/` — 34 条任务，max_steps=50，deepseek-reasoner，viewport-only。
