# L3 Batch 1 — Seed Batch (2026-04-21)

第一批从 M2W2 出发的端到端测试，验证 pipeline + agent + evaluator 全链路。

## 输入

`m2w2_seed_batch.csv` — 17 条手挑任务（来自 L2 phase2 早期版本，**不全在当前 L2/m2w2_filtered_final.csv 的 92 条里**，因为这次手挑发生在 phase2 prompt 调优之前）。

## Pipeline 输出

- `logs/pipeline/m2w2_seed_20260421/tasks.jsonl` — 13 条通过（4 条 filtered：`baltimore_marc_station`, `captain_america_starring`, `ra_vip_ticket`, `foster_dogs`）
- `logs/pipeline/m2w2_seed_20260421/report.json`

注：seed batch 的 instance_id 用了旧的 `dyna_<task_id[:8]>` 截断格式（`dyna_m2w2_<3char>`），这种格式有碰撞风险，从 batch2 起改为 `dyna_<full_task_id>`（参见 `src/pipeline_lib/export.py`）。

## Agent 执行

- `logs/runs/m2w2_seed_s40_20260422/` — 13 条任务，max_steps=40，deepseek-reasoner，mean score 0.267

## 评估报告

`docs/m2w2_seed_eval_report_20260422.md` 详细分析了执行结果、失败模式、milestone 触发情况。
