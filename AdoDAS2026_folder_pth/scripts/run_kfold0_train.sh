#!/usr/bin/env bash
# Pick emptiest GPU and run kfold=0 training in AdoDAS2026_folder_pth
set -euo pipefail
cd /home/adodas/AdoDAS2026_folder_pth

GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES="${GPU}"
echo "Using GPU ${CUDA_VISIBLE_DEVICES}"

python train.py \
  --task a1 \
  --config tasks/a1/default.yaml \
  --epochs 30 \
  --batch_size 32 \
  --num_workers 8 \
  2>&1 | tee logs_kfold0_train_$(date +%Y%m%d_%H%M%S).log
