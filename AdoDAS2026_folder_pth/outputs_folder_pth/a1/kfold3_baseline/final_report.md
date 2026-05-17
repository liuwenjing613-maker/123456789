# K-fold 3 report

- checkpoint for OOF/ensemble: `best_safe_submit.pt`
- oof: `outputs_folder_pth/a1/kfold3_baseline/oof/oof_predictions.csv`
- val ensemble: `outputs_folder_pth/a1/kfold3_baseline/ensemble/official_val_ensemble_raw.csv`
- test ensemble: `outputs_folder_pth/a1/kfold3_baseline/ensemble/test_ensemble_raw.csv`

## Official val metrics (threshold=0.5)

| model | mean_f1 | macro_auroc | D_f1 | A_f1 | S_f1 |
|-------|---------|-------------|------|------|------|
| ensemble | 0.4261 | 0.7035 | 0.4350 | 0.4933 | 0.3500 |
| fold_0 | 0.4148 | 0.6896 | 0.4260 | 0.5064 | 0.3119 |
| fold_1 | 0.3863 | 0.6839 | 0.4230 | 0.4428 | 0.2931 |
| fold_2 | 0.4094 | 0.7010 | 0.4516 | 0.4640 | 0.3125 |

Details: `outputs_folder_pth/a1/kfold3_baseline/ensemble/official_val_metrics_summary.json`

Recommended submissions:
1. test_ensemble_raw.csv
2. (optional) shrink0.3 after fitting OOF bias offline
