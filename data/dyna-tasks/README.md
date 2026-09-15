# Formal4 数据目录

目标：**60 seed × 3 动态变体 = 180 任务**（OM2W 30 + M2W2 30）。  
设计说明见 [`docs/formal4_dataset_design.md`](../../docs/formal4_dataset_design.md)。

| 路径 | 用途 |
|------|------|
| `seeds/` | 上游 seed 候选与分批冻结名单 |
| `batches/` | 通过质量门槛后的可运行任务 |
| `special/` | 手工试点（不计入正式 180） |
| `release/` | 冻结发布（尚未生成） |

## 当前状态

- `special/ign_boardgame`：首套试点留下的反例（不进 Batch0）；失败机制与日志例子见 [`docs/formal4_failure_cases.md`](../../docs/formal4_failure_cases.md)
- `special/ign_zelda_walkthrough`：Batch0 #1
- Batch0 OM2W seeds（`seeds/batch0_om2w.jsonl`，4 个齐）：
  1. IGN Zelda（`om2w_b320c68b`，drafted）— `special/ign_zelda_walkthrough`
  2. BestBuy speakers（`om2w_87f4c512`，drafted）— `special/pilot_bestbuy_speakers`
  3. AKC hairless breeds（`om2w_eb2db4b7`，accepted）— `special/pilot_akc_hairless_breed`
  4. TED robots talks（`om2w_bb5d90e6`，accepted）— `special/pilot_ted_robots`；Topics facet + 时长换桶 conditional
- `batches/batch0/tasks.jsonl`：上述 4 seed × 3 变体，共 12 条可运行任务

## Seed 记录字段（`seeds/*.jsonl`）

每行一个 JSON 对象，建议字段：

```json
{
  "seed_id": "om2w_aa4b5cb7",
  "source": "om2w",
  "source_task_id": "aa4b5cb7114fcc138ade82b4b9716d24",
  "website": "https://www.ign.com/",
  "base_task": "...",
  "status": "candidate",
  "batch": null,
  "notes": ""
}
```

`status`: `candidate` | `selected` | `drafted` | `accepted` | `rejected`。
