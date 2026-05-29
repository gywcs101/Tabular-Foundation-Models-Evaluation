# Repository Structure

本文档说明当前 GitHub 仓库的目录结构，以及每个文件/目录的用途。

## 根目录

```text
.
├── AGENTS.md
├── README.md
├── .gitignore
├── requirements.txt
├── requirements-core.txt
├── requirements-models.txt
├── requirements-baselines.txt
├── requirements-tabpfn.txt
├── requirements-tabicl.txt
├── data/
├── docs/
├── report/
├── results/
├── scripts/
└── slides/
```

### `AGENTS.md`

Codex 项目级上下文文件。保存作业要求、项目目标、默认实验策略、文献启发和后续对话约定。

### `README.md`

项目入口说明。给团队成员和 GitHub 访问者快速了解项目目标、依赖安装和基本运行命令。

### `.gitignore`

控制哪些文件不上传 GitHub。主要排除本地环境、缓存、模型权重、大型数据和临时结果。

### `requirements*.txt`

依赖文件：

- `requirements-core.txt`：基础数据处理和画图依赖。
- `requirements-baselines.txt`：LightGBM / XGBoost 环境依赖。
- `requirements-tabpfn.txt`：TabPFN 环境依赖。
- `requirements-tabicl.txt`：TabICL 环境依赖。
- `requirements-models.txt`：一次性安装所有模型包时使用。
- `requirements.txt`：总入口，引用 core 和 models。

## `docs/`

保存项目文档。

- `project_plan.md`：完整项目策划案。
- `literature_notes.md`：论文和 benchmark 笔记。
- `github_collaboration.md`：GitHub 协作规则。
- `repository_structure.md`：仓库结构说明。
- `execution_status.md`：当前执行进度和环境状态。
- `dataset_selection_criteria.md`：TALENT 数据选取报告，说明样本数扩展性和特征数扩展性的控制变量设计、轻量/稳健两套候选、不全量跑 TALENT 的理由和报告可用表述。

## `scripts/`

保存实验代码。

- `project_config.py`：统一配置，例如样本数扩展性/特征数扩展性数据集名单、轻量/稳健方案候选、随机种子、结果目录。
- `data_utils.py`：数据读取、划分、预处理、缺失值注入。
- `model_utils.py`：按名称创建模型，并兼容不同模型的训练接口。
- `experiment_utils.py`：训练、预测、计时、算指标、保存结果。
- `prepare_data.py`：下载并缓存 OpenML 数据。
- `run_benchmark.py`：主 benchmark 实验入口。
- `run_missing_value_experiment.py`：缺失值鲁棒性实验入口。
- `run_retrieval_icl_experiment.py`：retrieval-based in-context learning 加分实验入口。
- `run_smoke_test.py`：离线冒烟测试入口。
- `check_local_assets.py`：检查本机模型环境、TALENT 数据包和 TabArena 元数据是否可用。
- `merge_results.py`：合并两名成员各自跑出的结果 CSV。
- `analyze_results.py`：读取结果并生成图表。

## `data/`

保存本地数据缓存。原则上不上传真实数据。

- `data/openml_cache/.gitkeep`：保留目录结构。
- `data/openml_cache/`：本地 OpenML 数据缓存，已被 `.gitignore` 忽略。

## `results/`

保存实验结果。

- `results/raw/`：成员各自提交的小型原始结果 CSV。
- `results/final/`：合并后的最终结果 CSV。
- `results/figures/`：最终报告和 PPT 使用的图表。

注意：根目录下的临时结果文件会被忽略；需要协作共享的结果应放到 `results/raw/` 或 `results/final/`。

## `report/`

保存最终报告、报告草稿和相关素材。

## `slides/`

保存答辩 PPT、讲稿或展示素材。

## 不上传但本地可能存在的目录

### `.conda/`

本地 Python 环境，体积大，不上传 GitHub。

### `__pycache__/`

Python 自动缓存，不上传 GitHub。

### 模型权重和下载缓存

包括 `.pt`、`.pth`、`.ckpt`、`.safetensors`、OpenML 缓存等，都不上传 GitHub。
