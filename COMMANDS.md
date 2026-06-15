# Command Cheatsheet

本文件集中保存本项目常用命令。默认在 PowerShell 中运行，并先进入项目根目录：

```powershell
cd "D:\Codex项目\人工智能2大作业"
```

## 本地目录约定

```text
.local_envs\      本地 Conda 环境，不上传 GitHub
.local_models\    本地模型权重，不上传 GitHub
.local_external\  完整 TALENT/TabArena 等外部资源，不上传 GitHub
data\selected_talent\  已筛选实验数据，上传 GitHub
```

## Python 与环境检查

检查 TabPFN 环境：

```powershell
& ".\.local_envs\tabpfn\python.exe" -c "import tabpfn, torch, sklearn, pandas, numpy; print('tabpfn ok')"
```

检查 TabICL 环境：

```powershell
& ".\.local_envs\tabicl\python.exe" -c "import tabicl, torch, sklearn, numpy; print('tabicl ok')"
```

检查 XGBoost 环境：

```powershell
& ".\.local_envs\xgboost\python.exe" -c "import xgboost, sklearn, pandas, numpy; print('xgboost ok', xgboost.__version__)"
```

检查 LightGBM 环境：

```powershell
& ".\.local_envs\lightgbm\python.exe" -c "import lightgbm, sklearn, pandas, numpy; print('lightgbm ok', lightgbm.__version__)"
```

检查本机模型环境、权重和数据：

```powershell
python scripts\check_local_assets.py
```

## 查看 TALENT `.npy` 数据

查看 `mfeat-morphological` 的训练特征：

```powershell
python data\inspect_npy.py data\selected_talent\feature_scale\F1_6_20\mfeat-morphological_2000rows_6feat_multiclass\N_train.npy
```

查看 `mfeat-morphological` 的训练标签：

```powershell
python data\inspect_npy.py data\selected_talent\feature_scale\F1_6_20\mfeat-morphological_2000rows_6feat_multiclass\y_train.npy
```

默认最多直接展示 5000 行。超过 5000 行时，脚本会询问是否输出完整数据：

```text
Print the complete array? Enter y or n:
```

如果想把阈值改成 1000 行：

```powershell
python data\inspect_npy.py data\selected_talent\feature_scale\F1_6_20\mfeat-morphological_2000rows_6feat_multiclass\N_train.npy --max-rows 1000
```

## 当前单次实验

### 样本数扩展性：pc1

`pc1_1109rows_21feat_binclass` 属于样本数扩展性 A 组，路径是：

```text
data\selected_talent\sample_scale\A_1000_3000\pc1_1109rows_21feat_binclass
```

运行 TabICL 在 `pc1` 上的实验：

```powershell
& ".\.local_envs\tabicl\python.exe" scripts\run_tabicl_pc1.py
```

运行 TabPFN 在 `pc1` 上的实验：

```powershell
& ".\.local_envs\tabpfn\python.exe" scripts\run_tabpfn_pc1.py
```

运行 LightGBM 在 `pc1` 上的实验：

```powershell
& ".\.local_envs\lightgbm\python.exe" scripts\run_lightgbm_pc1.py
```

运行 XGBoost 在 `pc1` 上的实验：

```powershell
& ".\.local_envs\xgboost\python.exe" scripts\run_xgboost_pc1.py
```

### 特征数扩展性：mfeat 示例

运行 TabICL 在 `mfeat-morphological` 上的实验：

```powershell
& ".\.local_envs\tabicl\python.exe" scripts\run_tabicl_mfeat_morphological.py
```

如果只使用 `train`，不合并 `val`：

```powershell
& ".\.local_envs\tabicl\python.exe" scripts\run_tabicl_mfeat_morphological.py --no-use-val-in-train
```

运行 TabPFN 在 `mfeat-morphological` 上的实验：

```powershell
& ".\.local_envs\tabpfn\python.exe" scripts\run_tabpfn_mfeat_morphological.py
```

运行 LightGBM 在 `mfeat-morphological` 上的实验：

```powershell
& ".\.local_envs\lightgbm\python.exe" scripts\run_lightgbm_mfeat_morphological.py
```

运行 XGBoost 在 `mfeat-morphological` 上的实验：

```powershell
& ".\.local_envs\xgboost\python.exe" scripts\run_xgboost_mfeat_morphological.py
```

运行 TabICL 在 `mfeat-pixel` 上的实验：

```powershell
& ".\.local_envs\tabicl\python.exe" scripts\run_tabicl_mfeat_pixel.py
```

运行 TabPFN 在 `mfeat-pixel` 上的实验：

```powershell
& ".\.local_envs\tabpfn\python.exe" scripts\run_tabpfn_mfeat_pixel.py
```

运行 LightGBM 在 `mfeat-pixel` 上的实验：

```powershell
& ".\.local_envs\lightgbm\python.exe" scripts\run_lightgbm_mfeat_pixel.py
```

运行 XGBoost 在 `mfeat-pixel` 上的实验：

```powershell
& ".\.local_envs\xgboost\python.exe" scripts\run_xgboost_mfeat_pixel.py
```

## 结果位置

正式实验结果保存到：

```text
results\raw\{model}\{experiment_axis}\{scale_group}\{dataset_name}\{run_id}\
```

每个结果目录包含：

```text
metrics.csv
run_config.json
predictions.csv
confusion_matrix.csv
run.log
```

## 原有通用实验脚本

运行通用 benchmark：

```powershell
python scripts\run_benchmark.py --datasets all --models all --sample-sizes 1000,3000,10000
```

样本数扩展性实验：

```powershell
python scripts\run_benchmark.py --datasets pc1,kc1,sylvine,ringnorm,jm1,default_of_credit_card_clients --models all --sample-sizes 1000,3000,10000,30000
```

特征数扩展性实验：

```powershell
python scripts\run_benchmark.py --datasets mfeat-morphological,mfeat-zernike,mfeat-karhunen,mfeat-fourier,mfeat-factors,mfeat-pixel --models all --sample-sizes 2000
```

缺失值鲁棒性实验：

```powershell
python scripts\run_missing_value_experiment.py
```

retrieval-based in-context learning 实验：

```powershell
python scripts\run_retrieval_icl_experiment.py
```

生成图表：

```powershell
python scripts\analyze_results.py
```

合并两名成员结果：

```powershell
python scripts\merge_results.py
```

离线冒烟测试：

```powershell
python scripts\run_smoke_test.py
```

## Git 常用命令

查看当前改动：

```powershell
git status --short
```

查看远端仓库地址：

```powershell
git remote -v
```

提交前查看具体改动：

```powershell
git diff
```
