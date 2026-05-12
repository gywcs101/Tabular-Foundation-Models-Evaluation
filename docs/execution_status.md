# Execution Status

更新时间：2026-05-12

## 当前阶段

当前项目处于：

```text
GitHub 协作仓库准备阶段
```

当前重点不是运行模型，也不是下载数据，而是确保项目文件夹可以作为 GitHub 仓库上传，并让两名成员后续能基于同一套代码、同一套协作规则分别在本地运行实验。

## 已完成

- 整理课程项目要求到 `AGENTS.md`。
- 完成项目策划案：`docs/project_plan.md`。
- 完成文献笔记：`docs/literature_notes.md`。
- 完成 GitHub 协作说明：`docs/github_collaboration.md`。
- 完成仓库结构说明：`docs/repository_structure.md`。
- 建立项目目录结构：
  - `docs/`
  - `scripts/`
  - `data/openml_cache/`
  - `results/raw/`
  - `results/final/`
  - `results/figures/`
  - `report/`
  - `slides/`
- 建立依赖清单：
  - `requirements-core.txt`
  - `requirements-baselines.txt`
  - `requirements-tabpfn.txt`
  - `requirements-tabicl.txt`
  - `requirements-models.txt`
  - `requirements.txt`
- 建立实验脚本：
  - `scripts/prepare_data.py`
  - `scripts/run_benchmark.py`
  - `scripts/run_missing_value_experiment.py`
  - `scripts/run_retrieval_icl_experiment.py`
  - `scripts/analyze_results.py`
  - `scripts/merge_results.py`
  - `scripts/run_smoke_test.py`
- 建立共享工具模块：
  - `scripts/project_config.py`
  - `scripts/data_utils.py`
  - `scripts/model_utils.py`
  - `scripts/experiment_utils.py`
- 补强 `.gitignore`，避免上传：
  - 本地 Python/Conda 环境；
  - OpenML 数据缓存；
  - 模型权重；
  - Python 缓存；
  - 临时实验结果。
- 清理了不适合上传的临时文件：
  - `scripts/__pycache__/`
  - `results/smoke_test.csv`
  - `data/openml_cache/sklearn_openml/`

## 当前本地模型环境说明

用户已在本地准备两个独立 Anaconda 环境：

```text
E:\Software_Download\Anaconda\envs\tabpfn
E:\Software_Download\Anaconda\envs\tabicl
```

这两个环境使用的是 Python 3.11.15。项目后续默认采用 Python 3.11 环境，不再以 Anaconda base 的 Python 3.13 作为实验环境。

说明：

- 之前提到 “如果 TabPFN/TabICL 在 Python 3.13 下安装失败，改用 Python 3.10/3.11 环境” 是早期环境探索阶段的说法。
- 现在这个判断已经过时，因为团队已经决定使用 Python 3.11，并且模型环境已经放在 Anaconda 的 `envs` 目录下。
- 后续文档和协作应以 Python 3.11 为准。

当前检查结果：

- `tabpfn` 环境：已包含 `tabpfn`、`torch`、`sklearn`、`pandas`、`lightgbm`。
- `tabicl` 环境：已包含 `tabicl`、`torch`、`sklearn`。

是否补齐 `openml`、`pandas`、`lightgbm`、`xgboost` 等依赖，由用户和队友讨论后自行安装。Codex 当前不负责下载模型或数据集。

## 数据和模型下载策略

团队当前策略：

- Codex 不主动下载数据集或模型。
- 用户和队友讨论后，使用一致的本地下载路径保存数据集和模型。
- 模型不上传 GitHub。
- OpenML 原始缓存不上传 GitHub。
- GitHub 仓库只保存代码、文档、依赖说明、可复现配置和可共享的小型结果文件。

这意味着：

- 每个成员本地可以有自己的模型缓存路径。
- 如果需要在脚本中引用本地模型路径，应通过配置或命令行参数传入，不要把个人绝对路径写死进代码。
- 结果合并时，应统一输出 CSV 字段，而不是依赖两个人的本地目录结构一致。

## 当前 GitHub 准备状态

当前项目文件夹已经基本符合 GitHub 仓库要求：

- 代码、文档和依赖文件已经在仓库内。
- 本地环境、缓存、模型权重已通过 `.gitignore` 排除。
- 结果协作目录 `results/raw/` 和 `results/final/` 已准备。
- 仓库结构说明和协作说明已写入 `docs/`。

尚未完成：

- 当前项目根目录还没有初始化为 Git 仓库。
- 还没有绑定 GitHub 远端仓库。
- 还没有进行首次 commit。

## 建议下一步

当前不急着跑模型。建议按以下顺序继续：

1. 用户检查当前仓库文件结构和文档是否满意。
2. 初始化 Git 仓库。
3. 首次提交当前代码和文档。
4. 创建 GitHub 远端仓库并推送。
5. 队友 clone 仓库。
6. 两人确认本地数据和模型路径约定。
7. 之后再进入最小真实实验阶段。

