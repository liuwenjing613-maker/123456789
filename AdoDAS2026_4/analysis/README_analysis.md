# AdoDAS A1 数据集与校准分析

本目录按 `lwj/AdoDAS_A1_dataset_calibration_analysis_for_cursor.md` 实现：标签分布、预测分布、阈值稳定性、shrink 校准、session 质量、特征漂移、融合与总报告。

## 快速开始

1. 编辑 `analysis/configs/analysis_a1.yaml`：填写 `baseline` / `dynamic` 的 checkpoint、校准 JSON，或先导出 logits。
2. 生成预测 CSV（若尚无）：

```bash
cd /home/adodas/AdoDAS2026_4
python analysis/scripts/prepare_prediction_csvs.py --config analysis/configs/analysis_a1.yaml
```

3. 一键分析：

```bash
python analysis/scripts/run_all_analysis.py --config analysis/configs/analysis_a1.yaml
```

输出：`analysis/outputs/**` 与 `analysis/reports/a1_dataset_calibration_report.md`。

## 说明

- 大权重文件（`.pt`）不纳入本仓库时，请在 YAML 中指向本机绝对路径。
- `prepare_prediction_csvs.py` 会调用 `tools/export_a1_logits.py`（需 GPU/数据路径正确），耗时与显存取决于 `preload` 与 `batch_size`。
