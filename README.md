# Tabular Foundation Models Project

本项目用于完成“人工智能大模型”期末作业 Project 2：Tabular Foundation Models。

## 当前目标

先搭建一套可复现实验流程：

1. 从 `data/selected_talent/` 读取已经筛选好的 TALENT 分类数据集。
2. 运行 TabPFN v2、TabICL、LightGBM、XGBoost。
3. 记录 accuracy、balanced accuracy、macro F1、置信度、训练耗时、推理耗时、墙钟时间、峰值内存和训练吞吐量。
4. 分开做样本数扩展性、特征数扩展性、缺失值鲁棒性和 retrieval-based in-context learning 加分实验。
5. 生成报告和 PPT 可用的结果表与图表。

## 本地目录约定

为了让两名成员使用同一份脚本，项目采用项目内相对路径：

```text
.local_envs/       本地 Conda 环境，不上传 GitHub
.local_models/     本地模型权重，不上传 GitHub
.local_external/   原始 TALENT/TabArena 等大型外部资源，不上传 GitHub
data/selected_talent/ 已筛选数据集，上传 GitHub
```

推荐环境路径：

```text
.local_envs/tabpfn
.local_envs/tabicl
.local_envs/xgboost
.local_envs/lightgbm
```

推荐权重路径：

```text
.local_models/tabicl/tabicl-classifier-v2-20260212.ckpt
.local_models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt
```

## 依赖安装

本项目推荐用“每个模型一个环境”的方式协作，避免 TabPFN、TabICL、LightGBM/XGBoost 的依赖互相影响。

示例：

```powershell
python -m pip install -r requirements-tabpfn.txt
python -m pip install -r requirements-tabicl.txt
python -m pip install -r requirements-lightgbm.txt
python -m pip install -r requirements-xgboost.txt
```

如果确认依赖兼容，也可以一次安装全部依赖：

```powershell
python -m pip install -r requirements.txt
```

## 推荐执行顺序

先进入项目根目录：

```powershell
cd "D:\Codex项目\人工智能2大作业"
```

检查本机模型环境、权重和数据：

```powershell
python scripts/check_local_assets.py
```

运行单个数据集脚本，例如：

```powershell
& ".\.local_envs\tabicl\python.exe" scripts\run_tabicl_pc1.py
& ".\.local_envs\tabpfn\python.exe" scripts\run_tabpfn_pc1.py
& ".\.local_envs\lightgbm\python.exe" scripts\run_lightgbm_pc1.py
& ".\.local_envs\xgboost\python.exe" scripts\run_xgboost_pc1.py
```

实验结果写入：

```text
results/raw/{model}/{experiment_axis}/{scale_group}/{dataset_name}/{run_id}/
```

## 重要说明

- `data/selected_talent/` 当前体积较小，随 GitHub 仓库共享，用于保证两名成员跑同一批数据。
- `.local_envs/`、`.local_models/`、`.local_external/` 不上传 GitHub。
- 完整 TALENT 原始包和 TabArena 仓库只作为外部参考，不是运行当前 `selected_talent` 主实验的必要条件。
- 如果本地 CPU 太慢，优先把 TabICL 和大规模实验放到远端 GPU 上运行。

更多说明：

- GitHub 协作规则：`docs/github_collaboration.md`
- TALENT 数据集筛选准则：`docs/dataset_selection_criteria.md`
- 仓库结构说明：`docs/repository_structure.md`
- 实验测量计划：`docs/measurement_plan.md`
