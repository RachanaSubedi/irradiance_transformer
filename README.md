# Irradiance Transformer Project

Professional structure for the V2 14-feature irradiance imputation pipeline.

## Workflow

Run only the step you need:

```bash
python scripts/01_build_master.py
python scripts/02_build_pretrain_dataset.py
python scripts/03_pretrain.py
python scripts/04_finetune.py
python scripts/05_analyze_results.py
```

Each script checks whether its output already exists and skips expensive work unless you pass `--force`.

## Main saved outputs

- `master_v2.parquet` — processed complete-station master dataframe
- `X_pretrain_v2.npy`, `y_pretrain_v2.npy`, `meta_pretrain_v2.csv` — pretraining dataset
- `pretrain_best_model_v2.pt` — pretrained Transformer
- `finetune_frozen_46_78_synth_v2.pt` — fine-tuned model
- `station_46_78_full_year_GHI_v2.csv` — final imputed full-year output

## Feature order for V2

```text
0  anchor1_CSI
1  anchor1_mask
2  anchor2_CSI
3  anchor2_mask
4  NSRDB_CSI_target
5  C13_norm_target
6  NSRDB_CSI_anchor1
7  C13_norm_anchor1
8  NSRDB_CSI_anchor2
9  C13_norm_anchor2
10 hour_sin
11 hour_cos
12 doy_sin
13 doy_cos
```
