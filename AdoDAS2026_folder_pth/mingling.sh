#单独评估命令
python tools/evaluate_a1_official_val.py \
  --pred_csv outputs_folder_pth/a1/kfold3_baseline/ensemble/official_val_ensemble_raw.csv \
  --manifest /home/adodas/dataset/manifests/val.csv

#或对已有预测只补指标（不重跑 infer）：
python tools/run_a1_kfold.py --kfold 3 \
  --work_dir outputs_folder_pth/a1/kfold3_baseline \
  --manifest_dir /home/adodas/dataset/manifests \
  --phase ensemble --skip_completed