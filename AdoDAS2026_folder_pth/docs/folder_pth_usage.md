# AdoDAS2026_folder_pth 使用说明

## kfold=0 单模型训练（默认）

```bash
cd /home/adodas/AdoDAS2026_folder_pth
python train.py --task a1 --config tasks/a1/default.yaml --epochs 30 --batch_size 32 --num_workers 8
```

训练结束后 `runs/<run>/checkpoints/` 应包含：

- `best_raw_f1.pt` / `best_auc.pt` / `best_safe_submit.pt`
- `checkpoint_meta.json`
- `selection_summary.json` / `selection_summary.csv`
- `calibration/best_raw_f1.bias.json`、`calibration/best_auc.bias.json`、`calibration/best_safe_submit.bias.json`
- 可选 `last.pt`（`save_last_checkpoint: 1`）

主 checkpoint 按 `safe_submit_f1 = max(raw_f1, shrink0.3_f1, shrink0.5_f1)` 选取（tie → AUROC → raw_f1）。早停指标同为 `safe_submit_f1`。

## 训练前特征审计（推荐）

若 Val 曾报 `No valid sessions in batch`，通常是 manifest 中有参与者但磁盘缺少 `sequence.npz`。训练前可扫描：

```bash
python tools/audit_grouped_manifest.py \
  --manifest /path/to/val.csv \
  --config tasks/a1/default.yaml \
  --split val \
  --path_split train

# K-fold 内部 val 示例
python tools/audit_grouped_manifest.py \
  --manifest outputs_folder_pth/a1/kfold3_baseline/manifests/kfold_3/fold_0/val.csv \
  --config outputs_folder_pth/a1/kfold3_baseline/configs/fold_0.yaml \
  --split val
```

数据集会在初始化时自动过滤「零可加载 session」的参与者并打 WARNING；无特征列表会写到 `*.no_features.csv`。

## 推理（默认 raw，无 bias）

```bash
python infer.py \
  --task a1 \
  --checkpoint outputs/a1/runs/<run>/checkpoints/best_safe_submit.pt \
  --config tasks/a1/default.yaml \
  --split test_hidden \
  --output submissions/a1_best_safe_submit_raw.csv \
  --a1_bias_mode none \
  --dump_pred_stats
```

## 推理（auto + sidecar shrink，与 val 选点一致）

```bash
python infer.py \
  --task a1 \
  --checkpoint outputs/a1/runs/<run>/checkpoints/best_safe_submit.pt \
  --config tasks/a1/default.yaml \
  --split test_hidden \
  --output submissions/a1_best_safe_submit_calibrated.csv \
  --a1_bias_mode auto \
  --a1_use_sidecar_shrink \
  --dump_pred_stats
```

bias sidecar 路径：`checkpoints/calibration/<stem>.bias.json`（含 `safe_submit_mode` / `safe_submit_shrink`）。

## kfold=3 / kfold=5

```bash
python tools/run_a1_kfold.py --config tasks/a1/default.yaml --kfold 3 \
  --work_dir outputs_folder_pth/a1/kfold3_baseline --epochs 30 \
  --checkpoint_name best_safe_submit.pt

python tools/run_a1_kfold.py --config tasks/a1/default.yaml --kfold 5 \
  --work_dir outputs_folder_pth/a1/kfold5_baseline --epochs 30 \
  --checkpoint_name best_safe_submit.pt
```

## K-fold 断点续跑

`run_a1_kfold.py` 支持按折跳过已完成步骤（默认 `--skip_completed`），以及分阶段执行：

| `--phase` | 作用 |
|-----------|------|
| `all`（默认） | train+OOF → 合并 OOF → official val/test → ensemble |
| `train_oof` | 仅各折训练 + fold 内 val OOF |
| `merge_oof` | 从 `oof/fold_*_val_pred.csv` 写出 `oof/oof_predictions.csv` |
| `ensemble` | official val + test_hidden 推理 + 概率平均 ensemble |

常用参数：

- `--start_fold N`：只**新训练** `fold_N` 及之后；`fold_0..N-1` 须已有 checkpoint+OOF，否则报错。
- `--no-skip_completed`：强制重训/重推理（慎用，会覆盖 fold 输出）。
- `--remake_manifests`：强制重新生成 kfold manifest。
- 进度文件：`work_dir/kfold_progress.json`（默认 `--write_progress`）。

### 示例：fold_0/1 已完成，从 fold_2 续跑并跑完 ensemble

```bash
cd /home/adodas/AdoDAS2026_folder_pth
CUDA_VISIBLE_DEVICES=6 python tools/run_a1_kfold.py \
  --kfold 3 \
  --work_dir outputs_folder_pth/a1/kfold3_baseline \
  --config tasks/a1/default.yaml \
  --manifest_dir /home/adodas/dataset/manifests \
  --checkpoint_name best_safe_submit.pt \
  --start_fold 2 \
  --phase all \
  --skip_completed
```

日志中应出现 `SKIP fold_0 train_oof`、`SKIP fold_1 train_oof`，仅训练 `fold_2`。

### 示例：三折均已训完，只补 ensemble

```bash
python tools/run_a1_kfold.py --kfold 3 \
  --work_dir outputs_folder_pth/a1/kfold3_baseline \
  --manifest_dir /home/adodas/dataset/manifests \
  --phase merge_oof --skip_completed

python tools/run_a1_kfold.py --kfold 3 \
  --work_dir outputs_folder_pth/a1/kfold3_baseline \
  --manifest_dir /home/adodas/dataset/manifests \
  --phase ensemble --skip_completed
```

最终提交：`outputs_folder_pth/a1/kfold3_baseline/ensemble/test_ensemble_raw.csv`

ensemble 阶段对**官方** `val.csv` / `test_hidden.csv` 推理时会显式 `--path_split val` / `test_hidden`（折内配置里的 `val_sequence_path_split=train` 仅用于 K-fold 内部 val）。

ensemble 结束后默认对 official val 计算 **mean F1 / macro AUROC**（与训练 val 相同指标），输出：

- 每折：`ensemble/fold_i_official_val.csv.metrics.json`
- 集成：`ensemble/official_val_ensemble_raw.csv.metrics.json`
- 汇总：`ensemble/official_val_metrics_summary.json`，并写入 `final_report.md`

单独评估任意预测 CSV：

```bash
python tools/evaluate_a1_official_val.py \
  --pred_csv outputs_folder_pth/a1/kfold3_baseline/ensemble/official_val_ensemble_raw.csv \
  --manifest /home/adodas/dataset/manifests/val.csv
```

## pred_stats 检查

```bash
python tools/summarize_a1_submission.py --csv submissions/xxx.csv
```

参考报警（非硬规则）：D pred_pos>0.40、A>0.50、S>0.30。
