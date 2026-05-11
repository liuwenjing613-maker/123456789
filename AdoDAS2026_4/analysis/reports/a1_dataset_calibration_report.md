# A1 数据集与校准分析报告
（本报告由 `generate_calibration_report.py` 根据分析输出自动生成。）
## 总览
本次分析包含：
1. 标签分布分析
2. 预测概率分布分析
3. threshold 稳定性分析
4. 校准 bias 分析
5. session 质量分析
6. feature drift 分析
7. ensemble 分析

## 关键结论
### 1. train / val 中 D/A/S 正例率差异
- **D**: train = 0.2155, val = 0.2150, diff = 0.0005
- **A**: train = 0.2824, val = 0.2833, diff = 0.0010
- **S**: train = 0.1286, val = 0.1300, diff = 0.0014

### 2. threshold 稳定性（bootstrap 最优阈值）
- **D**: std = 0.0498, p05~p95 = 0.1750~0.3500
- **A**: std = 0.0202, p05~p95 = 0.2400~0.2852
- **S**: std = 0.0382, p05~p95 = 0.1590~0.2900

### 3. 校准风险（full calibration 与 pred_pos）
- shrink=1.0（full）与 shrink=0.4（示例轻量）时各类 `pred_pos_0.5` 与 `target_pos_rate` 对比见 `shrink_calibration_summary.csv`。
- **baseline**: full 平均 |pred_pos−target| ≈ 0.2432；shrink=0.4 时 ≈ 0.0472
- **dynamic**: full 平均 |pred_pos−target| ≈ 0.1738；shrink=0.4 时 ≈ 0.1460
- 若 full 校准下 pred_pos 相对 target 明显偏大/偏小，则存在“过度偏移”风险。
- val **calibrated** 预测：`prob_summary.csv` 中 `pred_pos@0.5` 列可对照标签正例率。

### 4. 动态特征 / 特征组漂移
- test 相对 train 漂移（按 feature_group 平均 test_drift_score）较高的组：
  - vad: 0.0866
  - vad_agg: 0.0634
  - global_motion: 0.0332
  - body_pose: 0.0198
  - headpose_geom: 0.0142
- **body_pose / global_motion** 均值：global_motion=0.0332, body_pose=0.0198

### 5. Ensemble 与推荐权重
- 推荐 baseline 权重 **0.5**，dynamic **0.5**（pred_calib=raw）。
- 说明：prefer baseline-heavy w among {0.5–0.8} using macro_F1 - 0.35*mean|pred_pos-val_rate|; picked w=0.5
- 备注：Global best F1 was at w=0.0 (macro F1=0.4258); chosen w=0.5 per stability / pred_pos rules (md §10.5).
- 融合测试集提交示例：`/home/adodas/AdoDAS2026_4/analysis/outputs/ensemble_analysis/test_submission_ensemble_w050.csv`

### 6. 推荐提交策略（摘要）
- test **baseline_test_raw** 上 D 的 pred_pos@0.5 与 train 正例率差距较大，需检查校准/偏差（§14.5）。
- test **baseline_test_raw** 上 A 的 pred_pos@0.5 与 train 正例率差距较大，需检查校准/偏差（§14.5）。
- test **baseline_test_raw** 上 S 的 pred_pos@0.5 与 train 正例率差距较大，需检查校准/偏差（§14.5）。
- ensemble 脚本推荐权重 baseline=0.5（见 `recommended_ensemble.json`）。

## 输出索引
- 根目录：`/home/adodas/AdoDAS2026_4/analysis/outputs`
- 子目录：`label_analysis/`、`prediction_analysis/`、`threshold_analysis/`、`calibration_analysis/`、`session_quality/`、`feature_drift/`、`ensemble_analysis/`。
