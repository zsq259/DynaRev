# IGN Board Game Special Set

Formal4 **第一套**手写 task family（IGN 桌游 Editors' Choice + 分数）。**不进 Batch0**（与 Zelda 同域；Batch0 同站只留 walkthrough）。

保留它，是因为首轮 DeepSeek pilot 把流水线问题跑清楚了，不是因为这套题本身要发布。失败机制（milestone 过窄、尖括号包动作导致评估 done 为空、static 都交不出卷）与日志例子见：

**[`docs/formal4_failure_cases.md`](../../../../docs/formal4_failure_cases.md)**（A1–A3）

本目录文件：

- `tasks.jsonl`：可跑任务（当前 schema）
- `design_metadata.json`：family / 变体说明（runner 不读）
- `audit.json`：当时检查清单与实验摘要（runner 不读；细节以 failure_cases 文档为准）

未完成 `audit.json` 里剩余 gate、且未按 failure_cases 修好对应问题之前，不要并入 Formal4 release。
