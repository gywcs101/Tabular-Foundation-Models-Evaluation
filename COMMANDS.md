# Command Cheatsheet

本文件集中保存本项目常用命令。默认在 PowerShell 中运行，并先进入项目根目录：

```powershell
cd "D:\Codex项目\人工智能2大作业"
```

## Python 与环境检查

检查 Anaconda Python：

```powershell
& "E:\Software_Download\Anaconda\python.exe" --version
```

检查 TabPFN 环境：

```powershell
& "E:\Software_Download\Anaconda\envs\tabpfn\python.exe" -c "import tabpfn, torch, sklearn, pandas, numpy; print('tabpfn ok')"
```

检查 TabICL 环境：

```powershell
& "E:\Software_Download\Anaconda\envs\tabicl\python.exe" -c "import tabicl, torch, sklearn, numpy; print('tabicl ok')"
```

检查 XGBoost 环境：

```powershell
& "E:\Software_Download\Anaconda\envs\xgboost\python.exe" -c "import xgboost, sklearn, pandas, numpy; print('xgboost ok', xgboost.__version__)"
```

检查 LightGBM 环境：

```powershell
& "E:\Software_Download\Anaconda\envs\lightgbm\python.exe" -c "import lightgbm, sklearn, pandas, numpy; print('lightgbm ok', lightgbm.__version__)"
```

检查本机模型环境、TALENT 数据和 TabArena 元数据：

```powershell
& "E:\Software_Download\Anaconda\python.exe" scripts\check_local_assets.py
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

运行 TabICL 在 `mfeat-morphological` 上的实验，默认使用 `train + val` 作为上下文/训练样本，`test` 作为测试集：

```powershell
& "E:\Software_Download\Anaconda\envs\tabicl\python.exe" scripts\run_tabicl_mfeat_morphological.py
```

如果只使用 `train`，不合并 `val`：

```powershell
& "E:\Software_Download\Anaconda\envs\tabicl\python.exe" scripts\run_tabicl_mfeat_morphological.py --no-use-val-in-train
```

运行 TabPFN 在 `mfeat-morphological` 上的实验，默认使用本地权重：

```powershell
& "E:\Software_Download\Anaconda\envs\tabpfn\python.exe" scripts\run_tabpfn_mfeat_morphological.py
```

如果想手动指定 TabPFN 权重路径：

```powershell
& "E:\Software_Download\Anaconda\envs\tabpfn\python.exe" scripts\run_tabpfn_mfeat_morphological.py --model-path "E:\Software_Download\TabPFN_models\tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
```

运行 LightGBM 在 `mfeat-morphological` 上的实验：

```powershell
& "E:\Software_Download\Anaconda\envs\lightgbm\python.exe" scripts\run_lightgbm_mfeat_morphological.py
```

运行 XGBoost 在 `mfeat-morphological` 上的实验：

```powershell
& "E:\Software_Download\Anaconda\envs\xgboost\python.exe" scripts\run_xgboost_mfeat_morphological.py
```

运行 TabICL 在 `mfeat-pixel` 上的实验，默认使用 `train + val` 作为上下文/训练样本，`test` 作为测试集：

```powershell
& "E:\Software_Download\Anaconda\envs\tabicl\python.exe" scripts\run_tabicl_mfeat_pixel.py
```

运行 TabPFN 在 `mfeat-pixel` 上的实验，默认使用本地权重：

```powershell
& "E:\Software_Download\Anaconda\envs\tabpfn\python.exe" scripts\run_tabpfn_mfeat_pixel.py
```

运行 LightGBM 在 `mfeat-pixel` 上的实验：

```powershell
& "E:\Software_Download\Anaconda\envs\lightgbm\python.exe" scripts\run_lightgbm_mfeat_pixel.py
```

运行 XGBoost 在 `mfeat-pixel` 上的实验：

```powershell
& "E:\Software_Download\Anaconda\envs\xgboost\python.exe" scripts\run_xgboost_mfeat_pixel.py
```

旧试跑结果已经清空。正式实验脚本更新后，结果将保存到：

```text
results\raw\tabpfn\feature_scale\F1_6_20\mfeat-morphological_2000rows_6feat_multiclass\<run_id>\
results\raw\tabicl\feature_scale\F1_6_20\mfeat-morphological_2000rows_6feat_multiclass\<run_id>\
results\raw\lightgbm\feature_scale\F1_6_20\mfeat-morphological_2000rows_6feat_multiclass\<run_id>\
results\raw\xgboost\feature_scale\F1_6_20\mfeat-morphological_2000rows_6feat_multiclass\<run_id>\
results\raw\tabpfn\feature_scale\F3_100_300\mfeat-pixel_2000rows_240feat_multiclass\<run_id>\
results\raw\tabicl\feature_scale\F3_100_300\mfeat-pixel_2000rows_240feat_multiclass\<run_id>\
results\raw\lightgbm\feature_scale\F3_100_300\mfeat-pixel_2000rows_240feat_multiclass\<run_id>\
results\raw\xgboost\feature_scale\F3_100_300\mfeat-pixel_2000rows_240feat_multiclass\<run_id>\
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
