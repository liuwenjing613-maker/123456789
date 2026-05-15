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

## pred_stats 检查

```bash
python tools/summarize_a1_submission.py --csv submissions/xxx.csv
```

参考报警（非硬规则）：D pred_pos>0.40、A>0.50、S>0.30。
