# Project Context: Tabular Foundation Models

本项目是“人工智能大模型”期末作业的 Project 2，主题为 Tabular Foundation Models。后续在本工作区内的所有对话都应默认了解并遵守以下项目要求。

## 课程项目原始要求

### Background

Tabular Foundation Models 是面向结构化表格数据的预训练模型，可以通过 in-context learning 完成分类和回归任务，在不针对单个数据集专门训练的情况下做出较强的 zero-shot 预测。该方向由 TabPFN 开创，并由 TabICL、TabDPT 等模型继续扩展，正在真实表格数据 benchmark 上挑战长期占优的梯度提升树模型。

### Resource Note

Tabular Foundation Models 是本课程中计算资源需求较轻的选题之一。TabPFN v2 和 TabICL 可以通过 pip 安装，并能在 CPU 或小型 GPU 上运行。OpenML、TALENT 等 benchmark 数据集容易获取且规模适中。计算资源有限的学生尤其适合选择该主题。

本团队资源情况：

- 团队共 2 人。
- 每人有一台核显笔记本。
- 如果本地资源不足，可以租用远端 GPU。

### Project Goals

1. 运行至少两个 state-of-the-art Tabular Foundation Models。
   可选模型包括但不限于 TabPFN v2、TabICL、TabDPT。建议优先选择能在本地轻量运行、代码和安装较成熟的模型。

2. 在 benchmark 数据集上评估模型。
   可选数据来源包括 TALENT、TabArena、OpenML 等。评估至少覆盖：
   - accuracy 或其他合适的预测性能指标；
   - inference speed；
   - 不同数据集规模下的 scalability。

3. 设计实验分析现有模型的优缺点。
   重点分析方向包括：
   - small dataset 与 large dataset 上的性能差异；
   - categorical features 的处理能力；
   - missing values 的鲁棒性；
   - 与梯度提升树 baseline 的计算成本和性能对比，例如 XGBoost、LightGBM。

4. Big Plus：提出新的改进方向。
   可考虑但不限于：
   - 面向特定领域下游应用 fine-tune tabular foundation models；
   - 设计带 retrieval 的 in-context learning 策略，选择更有代表性的上下文样本；
   - 将 tabular foundation models 扩展到 regression 或 survival analysis 等任务。

### Key Papers and Code

| Paper | Venue | Code |
| --- | --- | --- |
| Accurate Predictions on Small Data with a Tabular Foundation Model (TabPFN v2) | Nature 2025 | GitHub |
| TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models | arXiv 2025 | GitHub |
| TabICL: A Tabular Foundation Model for In-Context Learning on Large Data | ICML 2025 | GitHub |
| TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model | arXiv 2026 | GitHub |
| TabDPT: Scaling Tabular Foundation Models | arXiv 2024 | arXiv |
| Tabular Survey: Representation Learning for Tabular Data | Survey | GitHub |

## 易理解的任务拆解

这个项目不是单纯复现一篇论文，而是做一个小型 benchmark 和分析报告。核心问题是：新一代表格基础模型在真实表格任务上是否真的好用，它们和传统 XGBoost、LightGBM 相比有哪些优势和短板。

最低完成标准：

- 至少选 2 个表格基础模型，例如 TabPFN v2 + TabICL。
- 至少选若干个公开 benchmark 数据集，最好包含小数据、中等数据、较大数据。
- 至少加入 1 到 2 个传统 baseline，例如 LightGBM 和 XGBoost。
- 记录预测性能、推理耗时、资源占用或可运行的数据规模上限。
- 做对比分析，而不只是贴结果表格。

建议的实验维度：

- 数据规模：少样本、中等规模、较大规模。
- 任务类型：优先分类任务；如果时间允许，再加入回归任务。
- 特征类型：数值特征、类别特征、混合特征。
- 缺失值：原始缺失、人工加入缺失、不同缺失比例。
- 成本：安装复杂度、运行时间、是否需要 GPU、显存或内存压力。

建议的团队分工：

- 成员 A：负责 TabPFN v2、LightGBM/XGBoost baseline、部分数据集整理。
- 成员 B：负责 TabICL 或 TabDPT、速度/规模实验、缺失值或类别特征实验。
- 两人共同完成结果汇总、图表、误差分析和最终报告。

## OpenML、TALENT、TabArena 的关系

OpenML、TALENT、TabArena 不是同一层级的东西，后续不要把它们都理解成“一个数据集”。

- OpenML：公开机器学习数据平台，更像“数据仓库”。适合本项目实际下载和运行具体表格数据集。
- TALENT：表格学习工具箱和 benchmark，包含大量分类、回归数据集和很多模型结果。适合参考它如何划分任务类型、数据规模和模型类别。
- TabArena：表格模型评测系统和排行榜，强调公平比较不同模型、速度、调参和集成。适合参考它如何做 benchmark 设计和结果分析。

本项目的数据策略：

- 当前本机已下载 TALENT 数据包，第一阶段主实验优先从 TALENT 中选择分类数据集。
- OpenML 作为补充取数平台，当需要复现具体公开数据集或补充缺失任务时再使用。
- TabArena 当前作为 benchmark 设计和元数据参考，不作为完整本地训练数据缓存。
- 实验设计参考 TALENT 和 TabArena，不全量复现 TALENT 或 TabArena。
- 扩展性实验拆成两条控制变量实验线：样本数扩展性和特征数扩展性。
- 样本数扩展性只采用 A/B/C 三组：800-2000 行、2000-10000 行、10000-30000 行，并尽量控制特征数接近。
- 特征数扩展性优先使用样本数约 2000 行的 mfeat 系列，并让特征数从低维到高维变化。
- 所有模型使用同一训练集和同一测试集；模型超时、失败或超出限制也作为实验结果记录。
- TALENT 数据筛选准则见 `docs/dataset_selection_criteria.md`。

## 文献对本项目的启发

更完整的文献笔记见 `docs/literature_notes.md`。后续做实验和写报告时应优先使用这些思想组织论述。

- TabPFN v2：强调在小中规模表格数据上通过 in-context learning 做强 zero-shot 预测。启发本项目必须重点测试小数据表现、推理速度，并和 LightGBM/XGBoost 对比。
- TabPFN-2.5：强调扩展 TabPFN 的规模、延迟和部署能力。启发本项目可以把它作为最新进展和未来工作，不作为第一阶段主线模型。
- TabICL：强调面向更大规模表格数据的 in-context learning。启发本项目把 TabICL 作为第二个主模型，并重点做规模实验。
- TabICLv2：强调更快、更强、更开放和可扩展。启发本项目在报告中讨论该方向的发展趋势，如合成数据、模型结构和 benchmark 表现。
- TabDPT：强调结合真实数据预训练、自监督学习和 retrieval。启发本项目把 retrieval-based in-context learning 作为 Big Plus 创新点。
- Tabular Survey：提供表格学习方法分类框架。启发本项目报告背景可以按 traditional tree models、representation learning、tabular foundation models 的逻辑展开。

## 推荐执行路线

1. 先确定模型组合。
   优先考虑 TabPFN v2 + TabICL。如果 TabICL 安装或运行受限，再考虑 TabDPT 或其他可运行的开源替代。

2. 先跑通最小实验。
   使用 1 到 2 个小型 TALENT 数据集，确认数据读取、预处理、指标统计和模型调用流程可行。

3. 加入传统 baseline。
   使用 LightGBM 和/或 XGBoost，作为性能、速度、资源消耗的对照组。

4. 扩展到代表性 benchmark。
   从 TALENT 中按 `docs/dataset_selection_criteria.md` 选择数据集，分别做样本数扩展性和特征数扩展性实验。

5. 做鲁棒性实验。
   对测试集或训练上下文样本人工加入缺失值，比较各模型性能下降幅度。

6. 做 Big Plus 改进。
   推荐优先尝试 retrieval-based in-context learning：为测试样本检索相似训练样本作为上下文，而不是随机选择上下文样本。这个方向和本地算力较匹配，不一定需要大规模训练。

## 资源与风险判断

基于团队只有两台核显笔记本，建议优先选择轻量实验路线：

- 本地优先运行 TabPFN v2、TabICL、LightGBM、XGBoost。
- 数据集先从 A/B/C 规模组开始，不把 30000 行以上作为当前主线。
- 速度和规模实验可以使用子采样，例如 1000、3000、10000、30000 条样本。
- 如果某个模型在 CPU 上推理过慢，再租用远端 GPU 做补充实验。
- 不建议把主要工作量放在 full fine-tuning 上，除非已有稳定 GPU 资源和足够时间。

主要风险：

- 某些最新模型代码可能依赖特定 CUDA、PyTorch 或包版本。
- Tabular foundation models 对输入格式、类别特征编码、上下文长度可能有硬限制。
- 大数据集上不一定能完整运行，需要记录失败原因或使用子采样。
- 如果只跑 accuracy 而没有速度、规模、鲁棒性分析，容易不满足题目要求。

## 后续对话默认约定

当用户在本项目中继续提问时，应默认：

- 这是 Project 2: Tabular Foundation Models 的期末作业。
- 目标是做可复现的实验、对比分析和最终报告。
- 实验应至少包含两个表格基础模型，以及 XGBoost/LightGBM 等传统 baseline。
- 需要关注 accuracy、inference speed、scalability、categorical features、missing values、computational cost。
- 团队计算资源有限，应优先采用可在普通笔记本运行的方案，必要时再使用远端 GPU。
- 如果提出创新点，优先考虑 retrieval-based in-context learning，因为它更适合有限算力条件。
- TALENT 是当前优先实际取数来源；OpenML 是补充平台；TabArena 是重要评测参考，不要求全量复现。
- 当前数据选择采用两条线：样本数扩展性和特征数扩展性。
- 项目文献综述和实验依据应参考 `docs/literature_notes.md`。
- 当前可执行策划案见 `docs/project_plan.md`。
- 当前执行进度和环境阻塞点见 `docs/execution_status.md`。
- TALENT 数据集选择和“不全量跑”的报告理由见 `docs/dataset_selection_criteria.md`。
