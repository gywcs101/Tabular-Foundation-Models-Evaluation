# Literature Notes: Tabular Foundation Models

本文档整理 Project 2 要求中给出的核心文献和代码线索，并说明它们对本项目实验设计的启发。

## 1. 数据来源与 benchmark 的关系

### OpenML

OpenML 是公开机器学习数据平台，可以理解为“数据仓库”。它提供大量表格数据集和任务元信息，适合本项目直接下载数据、固定随机种子、复现实验。

对本项目的作用：

- 作为第一阶段实际取数来源。
- 方便选择小、中、大不同规模数据集。
- 适合两台核显笔记本做轻量 benchmark。

### TALENT

TALENT 是表格学习工具箱和 benchmark，包含大量分类、回归任务，也覆盖深度学习、传统机器学习和表格基础模型等方法。

对本项目的作用：

- 不要求全量复现。
- 用它指导数据集选择：分类/回归、小数据/大数据、数值特征/类别特征/缺失值。
- 用它指导报告写法：说明本项目是轻量复现和代表性评测，而不是完整 benchmark 平台。

### TabArena

TabArena 是表格模型评测系统和排行榜，强调公平、可复现、持续更新的 benchmark。

对本项目的作用：

- 不要求全量复现。
- 参考它的对比思路：多个模型、多个数据集、统一指标、速度和成本分析。
- 参考它说明为什么完整 benchmark 计算量过大，本项目采用代表性数据集和子采样规模实验。

## 2. Key Papers

### Accurate Predictions on Small Data with a Tabular Foundation Model (TabPFN v2)

核心思想：

- 把表格预测问题看作一种可以通过预训练学习的任务。
- 使用大量合成表格任务训练模型，使模型在新表格数据上通过 in-context learning 直接预测。
- 重点优势在小中规模数据上，不需要为每个数据集重新训练大模型。

对本项目的启发：

- TabPFN v2 应作为主模型之一。
- 必须重点测试小数据集表现，因为这是它最可能体现优势的场景。
- 必须记录推理速度和数据规模上限，因为 TabPFN 对样本数、特征数和类别数通常有实际限制。
- 必须和 LightGBM/XGBoost 对比，否则无法证明它是否真正优于传统表格强模型。

项目中的用法：

- 在 6 个代表性数据集上运行。
- 在规模实验中优先跑到 10000 行以内。
- 在报告中作为“表格基础模型起点和主线模型”介绍。

### TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models

核心思想：

- 在 TabPFN v2 基础上进一步提升性能、规模适应能力和部署效率。
- 关注更强的真实数据表现、更低延迟和更实用的模型形态。

对本项目的启发：

- 可作为“最新进展”和“未来工作”写入报告。
- 如果安装和运行非常顺利，可以作为额外尝试；但不建议作为第一阶段必须完成的模型。
- 可以用于解释该方向正在从“小数据强模型”向“更大规模、更低成本、更易部署”发展。

项目中的用法：

- 默认不作为主线实验模型。
- 写入文献综述和后续工作。

### TabICL: A Tabular Foundation Model for In-Context Learning on Large Data

核心思想：

- 继续使用 in-context learning 思路处理表格数据。
- 重点解决 TabPFN 类模型在更大数据集上扩展性不足的问题。
- 面向大规模表格任务，强调更强的 scalability。

对本项目的启发：

- TabICL 应作为第二个主模型。
- 规模实验不能只看 TabPFN，还要比较 TabICL 在 10000 行以上是否更稳定。
- 适合回答课程要求中的 “scalability across different dataset sizes”。

项目中的用法：

- 与 TabPFN v2 并列为两个表格基础模型。
- 在 adult、bank-marketing 等较大数据上重点测试。
- 如果本地 CPU 太慢，可作为租远端 GPU 的主要理由。

### TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model

核心思想：

- 在 TabICL 基础上进一步提升速度、性能、开放性和可扩展性。
- 强调更好的预训练策略、模型结构和 benchmark 表现。
- 同时覆盖分类和回归，是表格基础模型方向的进一步发展。

对本项目的启发：

- 报告中可把它作为最新趋势讨论。
- 如果 pip 或 GitHub 环境直接可用，可以作为附加实验尝试。
- 但为了保证项目完成，主线仍以 TabPFN v2 + TabICL 为准。

项目中的用法：

- 文献综述和未来工作。
- 不作为最低完成标准。

### TabDPT: Scaling Tabular Foundation Models

核心思想：

- 尝试扩展表格基础模型的规模和适用性。
- 重视真实数据预训练、自监督学习和 retrieval 等机制。
- retrieval 的思想是从已有样本中找相关样本，帮助模型更好地预测当前样本。

对本项目的启发：

- 支持本项目 Big Plus 选 retrieval-based in-context learning。
- 说明“相似样本检索”不是随意想法，而是和已有研究方向一致。
- 如果 TabICL 安装失败，TabDPT 可作为备用模型方向之一。

项目中的用法：

- 用于支撑创新点设计。
- 可作为备用模型或相关工作。

### Tabular Survey: Representation Learning for Tabular Data

核心思想：

- 系统梳理表格数据学习方法。
- 可将方法理解为传统树模型、表征学习方法、可迁移模型、通用表格基础模型等类别。
- 帮助解释为什么表格数据长期由 XGBoost/LightGBM 占优，以及为什么 TabPFN/TabICL 是新的研究方向。

对本项目的启发：

- 报告背景不要直接从 TabPFN 开始，应先讲表格数据为什么重要、为什么传统树模型强、为什么基础模型值得研究。
- 实验部分应体现“新模型 vs 传统强 baseline”的比较。

项目中的用法：

- 作为报告背景和相关工作章节的结构依据。

## 3. 对实验设计的直接结论

本项目应采用以下固定策略：

- 两个主模型：TabPFN v2、TabICL。
- 两个传统 baseline：LightGBM、XGBoost。
- 主任务：分类。
- 数据来源：优先 OpenML 实际取数，参考 TALENT/TabArena 的评测思路。
- 数据数量：约 6 个代表性数据集，不全量复现任何大型 benchmark。
- 规模实验：使用 500、1000、2000、5000、10000、30000 行子采样。
- 鲁棒性实验：人工加入 10%、30%、50% 缺失值。
- 创新点：retrieval-based in-context learning。

## 4. 参考链接

- TabPFN v2 Nature: https://www.nature.com/articles/s41586-024-08328-6
- TabPFN usage tips: https://tabpfn.github.io/site/intended_use/
- TabPFN-2.5: https://arxiv.org/abs/2511.08667
- TabICL ICML 2025: https://proceedings.mlr.press/v267/qu25d.html
- TabICL GitHub: https://github.com/soda-inria/tabicl
- TabICLv2: https://arxiv.org/abs/2602.11139
- TabDPT: https://arxiv.org/abs/2410.18164
- TALENT GitHub: https://github.com/LAMDA-Tabular/TALENT
- TabArena GitHub: https://github.com/autogluon/tabarena
- OpenML data docs: https://docs.openml.org/concepts/data/
- Tabular survey: https://arxiv.org/abs/2504.16109
