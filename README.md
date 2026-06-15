# Tabular Foundation Models Project

本项目用于完成“人工智能大模型”期末作业 Project 2：Tabular Foundation Models。

## 当前目标

先搭建一套可复现实验流程：

1. 从 TALENT/OpenML 等来源读取代表性表格数据集。
2. 运行 TabPFN v2、TabICL、LightGBM、XGBoost。
3. 记录 accuracy、balanced accuracy、macro F1、置信度、训练耗时、推理耗时、墙钟时间、峰值内存和训练吞吐量。
4. 分开做样本数扩展性、特征数扩展性、缺失值鲁棒性和 retrieval-based in-context learning 加分实验。
5. 生成报告和 PPT 可用的结果表与图表。

## 依赖安装

本项目推荐用“每个模型一个环境”的方式协作，避免 TabPFN、TabICL、LightGBM/XGBoost 的依赖互相影响。

baseline 环境安装：

```powershell
python -m pip install -r requirements-baselines.txt
```

当前推荐按模型分别维护环境：

```text
E:\Software_Download\Anaconda\envs\tabpfn
E:\Software_Download\Anaconda\envs\tabicl
E:\Software_Download\Anaconda\envs\xgboost
E:\Software_Download\Anaconda\envs\lightgbm
```

TabPFN 环境安装：

```powershell
python -m pip install -r requirements-tabpfn.txt
```

TabICL 环境安装：

```powershell
python -m pip install -r requirements-tabicl.txt
```

如果确认依赖兼容，也可以一次安装全部依赖：

```powershell
python -m pip install -r requirements.txt
```

如果 `python` 命令不可用，需要先安装 Python 3.10+，或把 Python 加入系统 PATH。

如果担心一次安装全部模型包时间太长，也可以先装基础依赖：

```powershell
python -m pip install -r requirements-core.txt
```

当前机器可直接使用的 Python 路径是：

```powershell
& "E:\Software_Download\Anaconda\python.exe" --version
```

GitHub 协作规则见：

```text
docs/github_collaboration.md
```

TALENT 数据集筛选准则见：

```text
docs/dataset_selection_criteria.md
```

完整仓库结构说明见：

```text
docs/repository_structure.md
```

## 推荐执行顺序

先检查本机已下载的模型环境和数据集：

```powershell
python scripts/check_local_assets.py
```

当前默认检查：

```text
E:\Software_Download\Anaconda\envs\tabpfn
E:\Software_Download\Anaconda\envs\tabicl
E:\Software_Download\TALENT-tabular-benchmark
E:\Software_Download\tabarena
```

再准备数据：

```powershell
python scripts/prepare_data.py --datasets all
```

再跑主实验。样本数扩展性先按轻量或稳健方案之一执行，特征数扩展性优先使用 `mfeat` 系列：

```powershell
python scripts/run_benchmark.py --datasets all --models all --sample-sizes 1000,3000,10000
```

做样本数扩展性实验（轻量或稳健方案二选一；正式脚本更新后结果写入 `results/raw/`）：

```powershell
python scripts/run_benchmark.py --datasets pc1,kc1,sylvine,ringnorm,jm1,default_of_credit_card_clients --models all --sample-sizes 1000,3000,10000,30000
```

做特征数扩展性实验：

```powershell
python scripts/run_benchmark.py --datasets mfeat-morphological,mfeat-zernike,mfeat-karhunen,mfeat-fourier,mfeat-factors,mfeat-pixel --models all --sample-sizes 2000
```

做缺失值鲁棒性实验：

```powershell
python scripts/run_missing_value_experiment.py
```

做 retrieval-based in-context learning 加分实验：

```powershell
python scripts/run_retrieval_icl_experiment.py
```

生成图表：

```powershell
python scripts/analyze_results.py
```

合并两名成员各自跑出的结果：

```powershell
python scripts/merge_results.py
```

没有网络或模型包未安装时，可以先跑离线冒烟测试，确认脚本框架可用：

```powershell
python scripts/run_smoke_test.py
```

## 重要说明

- TabPFN 和 TabICL 首次运行可能会下载模型权重，因此需要网络。
- 如果本地 CPU 太慢，优先把 TabICL 和大规模实验放到远端 GPU 上运行。
- 不建议全量跑大型 benchmark；先按 `docs/dataset_selection_criteria.md` 分开做样本数扩展性和特征数扩展性实验，再逐步扩大。样本数扩展性目前保留轻量和稳健两套方案，待审核后定稿。
