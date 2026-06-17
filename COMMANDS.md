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

## 主实验：一键跑完

用自己的名字区分结果。比如你的名字/ID 是 `gywcs101`：

```powershell
python scripts\controllers\run_all_selected_talent.py --runner-name gywcs101
```

同学可以换成自己的名字/ID：

```powershell
python scripts\controllers\run_all_selected_talent.py --runner-name teammate_name
```

只预览将要执行哪些实验，不真正运行：

```powershell
python scripts\controllers\run_all_selected_talent.py --runner-name gywcs101 --dry-run
```

只跑某几个模型：

```powershell
python scripts\controllers\run_all_selected_talent.py --runner-name gywcs101 --models tabicl,tabpfn
```

主实验结果保存到：

```text
results\raw\{runner_name}\{model}\{experiment_axis}\{scale_group}\{dataset_name}\{run_id}\
```

例如：

```text
results\raw\gywcs101\tabicl\sample_scale\A_1000_3000\pc1_1109rows_21feat_binclass\run_...\ 
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

## 单次实验调试

这些命令用于调试单个模型/单个数据集。主实验不需要一个个手动跑。

运行 TabICL 在 `pc1` 上的实验：

```powershell
& ".\.local_envs\tabicl\python.exe" scripts\experiments\single\run_tabicl_pc1.py --runner-name gywcs101
```

运行 TabPFN 在 `pc1` 上的实验：

```powershell
& ".\.local_envs\tabpfn\python.exe" scripts\experiments\single\run_tabpfn_pc1.py --runner-name gywcs101
```

运行 LightGBM 在 `pc1` 上的实验：

```powershell
& ".\.local_envs\lightgbm\python.exe" scripts\experiments\single\run_lightgbm_pc1.py --runner-name gywcs101
```

运行 XGBoost 在 `pc1` 上的实验：

```powershell
& ".\.local_envs\xgboost\python.exe" scripts\experiments\single\run_xgboost_pc1.py --runner-name gywcs101
```

运行 TabICL 在 `mfeat-morphological` 上的实验：

```powershell
& ".\.local_envs\tabicl\python.exe" scripts\experiments\single\run_tabicl_mfeat_morphological.py --runner-name gywcs101
```

运行 TabPFN 在 `mfeat-morphological` 上的实验：

```powershell
& ".\.local_envs\tabpfn\python.exe" scripts\experiments\single\run_tabpfn_mfeat_morphological.py --runner-name gywcs101
```

运行 LightGBM 在 `mfeat-morphological` 上的实验：

```powershell
& ".\.local_envs\lightgbm\python.exe" scripts\experiments\single\run_lightgbm_mfeat_morphological.py --runner-name gywcs101
```

运行 XGBoost 在 `mfeat-morphological` 上的实验：

```powershell
& ".\.local_envs\xgboost\python.exe" scripts\experiments\single\run_xgboost_mfeat_morphological.py --runner-name gywcs101
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

## 结果文件

每个结果目录包含：

```text
metrics.csv
run_config.json
predictions.csv
confusion_matrix.csv
run.log
```

## 结果分析与绘图命令

游戏本结果还没跑完时，也可以先用现有轻薄本结果生成第一版分析表和图。等游戏本结果放入 `results/raw/` 后，重新运行同一条命令即可刷新。

一键完整分析：

```powershell
python scripts\analysis\run_full_analysis.py
```

只检查结果是否完整：

```powershell
python scripts\analysis\check_analysis_inputs.py
```

只合并结果并生成最终 CSV 表：

```powershell
python scripts\analysis\build_final_tables.py
```

单独生成预测性能图：

```powershell
python scripts\analysis\plot_performance.py
```

单独生成扩展性和时间图：

```powershell
python scripts\analysis\plot_scalability.py
```

单独生成峰值内存图：

```powershell
python scripts\analysis\plot_memory.py
```

单独生成置信度图：

```powershell
python scripts\analysis\plot_confidence.py
```

单独生成混淆矩阵图：

```powershell
python scripts\analysis\plot_confusion_matrices.py
```

主要输出位置：

```text
results/final/
results/figures/
```

正式图表会同时保存为 `.png` 和 `.svg`。

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
