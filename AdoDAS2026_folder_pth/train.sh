cd /home/adodas/AdoDAS2026_folder_pth
'''CUDA_VISIBLE_DEVICES=1 python tools/run_a1_kfold.py \
  --config tasks/a1/default.yaml --kfold 3 \
  --work_dir outputs_folder_pth/a1/kfold3_baseline \
  --checkpoint_name best_safe_submit.pt --epochs 30
'''

CUDA_VISIBLE_DEVICES=2 python tools/run_a1_kfold.py \
  --config tasks/a1/default.yaml --kfold 0 \
  --work_dir   /home/adodas/AdoDAS2026_folder_pth/outputs/a1/runs \
  --checkpoint_name best_safe_submit.pt --epochs 30