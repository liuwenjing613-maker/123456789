cd /home/adodas/AdoDAS2026_folder_pth

RUN=/home/adodas/AdoDAS2026_folder_pth/outputs_folder_pth/a1/kfold3_baseline/fold_0/outputs/a1/runs/a1__grouped__mtcn__a-base-mel_mfcc+vad+egemaps__a-ssl-wav2vec2-chinese-xlsr__v-base-headpose+facebeh+qc+vadagg__v-ssl-dinov2-large__mask-andcore__pw_biascalib__seed42__20260515_162910

# kfold 的 manifest_dir 只有 train/val；test_hidden 必须用官方 manifest
CUDA_VISIBLE_DEVICES=1 python infer.py \
  --task a1 \
  --checkpoint "$RUN/checkpoints/best_safe_submit.pt" \
  --config "$RUN/config_used.yaml" \
  --manifest /home/adodas/dataset/manifests/test_hidden.csv \
  --split test_hidden \
  --output submissions/kfold3_fold0_best_safe_submit_shrink05.csv \
  --a1_bias_mode auto \
  --a1_use_sidecar_shrink \
  --dump_pred_stats