# Selected TALENT Datasets

This folder is organized by experiment axis.

## Sample-size scaling

### `sample_scale/A_1000_3000`

- `pc1_1109rows_21feat_binclass`
- `kc1_2109rows_21feat_binclass`

### `sample_scale/B_3000_10000`

- `sylvine_5124rows_20feat_binclass`
- `ringnorm_7400rows_20feat_binclass`

### `sample_scale/C_10000_30000`

- `jm1_10885rows_21feat_binclass`
- `default_of_credit_card_clients_30000rows_23feat_binclass`

## Feature-size scaling

### `feature_scale/F1_6_20`

- `mfeat-morphological_2000rows_6feat_multiclass`

### `feature_scale/F2_20_100`

- `mfeat-zernike_2000rows_47feat_multiclass`
- `mfeat-karhunen_2000rows_64feat_multiclass`
- `mfeat-fourier_2000rows_76feat_multiclass`

### `feature_scale/F3_100_300`

- `mfeat-factors_2000rows_216feat_multiclass`
- `mfeat-pixel_2000rows_240feat_multiclass`

## Naming rule

Each dataset folder uses:

```text
<dataset_name>_<rows>rows_<features>feat_<task_type>
```

This makes the dataset easy to identify without opening `info.json`.
