# GitHub Collaboration Guide

本文档说明本项目如何整理成 GitHub 仓库，以及两名成员如何各自本地运行实验、合并结果。

## 1. 仓库原则

GitHub 仓库只保存轻量、可复现、适合协作的内容：

- 代码：`scripts/`
- 文档：`docs/`
- 环境依赖说明：`requirements*.txt`
- 报告和 PPT 草稿：`report/`、`slides/`
- 经过整理的小型结果文件：`results/raw/`、`results/final/`

不上传：

- Python/Conda 环境，例如 `.conda/`
- OpenML 原始下载缓存，例如 `data/openml_cache/`
- TabPFN、TabICL、PyTorch 等模型权重
- 大型中间结果、临时图表、缓存文件

原因：

- 模型和环境体积大，不适合 GitHub 普通仓库。
- 每个人的系统路径不同，上传环境没有复用价值。
- 模型权重可能有自己的许可协议，不应随意二次上传。
- 代码、依赖文件和固定随机种子足以保证可复现。

## 2. 推荐环境策略

本项目采用“每个模型一个环境”的协作策略：

```text
env_tabpfn   -> TabPFN
env_tabicl   -> TabICL
env_xgboost  -> XGBoost
env_lightgbm -> LightGBM
```

优点：

- 新模型依赖不容易互相冲突。
- 两名成员可以分工维护不同模型环境。
- 某个模型环境坏了，不影响其他实验。

缺点：

- 不能一个命令跑完所有模型。
- 需要统一结果格式，最后再合并。

当前本机路径示例：

```text
E:\Software_Download\Anaconda\envs\tabpfn
E:\Software_Download\Anaconda\envs\tabicl
E:\Software_Download\Anaconda\envs\xgboost
E:\Software_Download\Anaconda\envs\lightgbm
```

## 3. 依赖安装

baseline 环境：

```powershell
python -m pip install -r requirements-baselines.txt
```

TabPFN 环境：

```powershell
python -m pip install -r requirements-tabpfn.txt
```

TabICL 环境：

```powershell
python -m pip install -r requirements-tabicl.txt
```

如果确认所有依赖兼容，也可以使用总依赖：

```powershell
python -m pip install -r requirements.txt
```

## 4. 结果文件规则

两名成员不要直接同时修改同一个 `metrics.csv`。建议每个人输出自己的结果文件：

```text
results/raw/member_a_tabpfn.csv
results/raw/member_b_tabicl.csv
results/raw/member_a_baselines.csv
results/raw/member_b_baselines.csv
```

整理后再合并成：

```text
results/final/final_metrics.csv
```

每条结果至少包含：

```text
dataset
model
actual_sample_size
train_rows
test_rows
accuracy
balanced_accuracy
macro_f1
fit_time_seconds
predict_time_seconds
seconds_per_test_sample
device
environment_name
package_versions
success
error_message
```

## 5. 本地模型和数据

模型各自保存在本地，不上传 GitHub。

当前本机数据路径示例：

```text
E:\Software_Download\TALENT-tabular-benchmark
E:\Software_Download\tabarena
```

TALENT 数据包可直接作为实验数据来源。TabArena 当前是源码仓库和元数据，适合参考数据集列表和评测思路，不等于完整可训练数据缓存。

OpenML、TALENT、TabArena 原始数据都不上传 GitHub。只要脚本中固定：

- 数据集名称或 OpenML id
- 数据集版本
- 随机种子
- 训练/测试划分规则

两个人的结果就可以比较和合并。

## 6. 提交前检查

提交到 GitHub 前检查：

```powershell
git status --short
```

不应该出现在待提交列表里的内容：

```text
.conda/
data/openml_cache/
__pycache__/
*.pt
*.pth
*.ckpt
*.safetensors
results/smoke_test.csv
```
