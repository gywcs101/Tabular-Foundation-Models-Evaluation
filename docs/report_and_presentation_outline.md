# Final Report and 15-Minute Presentation Outline

This document defines the planned structure of the English final report and the 15-minute group presentation. The current analysis includes five completed result sets: two light-laptop runs and three gaming-laptop runs.

## Required First Page

The first page of both the report and the slides must include:

- Project title: **Benchmarking Tabular Foundation Models under Limited Compute**
- Course project: **Project 2: Tabular Foundation Models**
- Team members:
  - **Wu Zhaolin**
  - **Yao Huaiyu**
- A short subtitle, optional: **A comparison of TabICL, TabPFN, LightGBM, and XGBoost on selected TALENT classification datasets**

If the instructor expects Chinese names exactly as registered, use:

- **吴兆临**
- **姚怀宇**

## Report Structure

The report should follow an ML-paper style, but it does not need to reach conference-paper depth. A recommended length is about 8-12 pages excluding appendix.

### 1. Introduction

Purpose:

- Introduce tabular foundation models as pretrained models for structured tabular data.
- Explain why they are interesting: they can perform classification with little or no dataset-specific training, and they challenge traditional gradient-boosted trees.
- State the practical motivation of this project: our team has limited compute, so we evaluate whether these models are useful on CPU laptops.

Main questions:

- Can TabICL and TabPFN achieve stronger predictive performance than LightGBM and XGBoost?
- What computational cost do they pay in inference time and memory?
- How do they scale as sample size or feature count increases?
- Are their confidence estimates useful for understanding correct and wrong predictions?

Recommended figures/tables:

- No heavy result figure is required in Introduction.
- Use a small conceptual table comparing model families:
  - Tabular foundation models: TabICL, TabPFN
  - Gradient boosting baselines: LightGBM, XGBoost

Current likely message:

- Tabular foundation models are attractive because they improve accuracy on many selected tasks, but their computational cost can be much higher than tree baselines on CPU.

### 2. Problem Definition

Purpose:

- Define the task formally as supervised tabular classification.
- Clarify that this project focuses on classification datasets from TALENT.
- Define the evaluation setting: same train/validation/test split for all models, same selected datasets, same metrics.

Problem formulation:

- Given a tabular dataset with samples `X` and labels `y`, train or condition a model using the training plus validation split and evaluate predictions on the test split.
- For tabular foundation models, "fit" mainly means preparing or conditioning on context examples rather than training model weights from scratch.
- For LightGBM and XGBoost, "fit" means standard supervised training on the same train plus validation split.

Metrics to define:

- Accuracy
- Macro F1
- Mean confidence
- Correct mean confidence
- Wrong mean confidence
- Confidence gap
- Fit time
- Predict time
- Wall time
- Seconds per test sample
- Peak memory
- Scalability exponent `alpha`, fitted by:

```text
log(T) = alpha * log(N) + C
```

where `T` is time or memory cost, and `N` is sample size or feature count.

Recommended interpretation of `alpha`:

- Use `alpha` as an empirical scalability indicator, not as a theoretical complexity proof.
- In the main report, prioritize time-based `alpha`, especially for sample-size scalability.
- Use memory `alpha` only as appendix or supporting analysis. Peak-memory bar charts are more intuitive and more reliable for the main text because memory readings are sensitive to runtime state, model caching, and Python process overhead.

Recommended figures/tables:

- `results/final/input_completeness_report.csv`
- `results/final/final_metrics.csv`
- A short table of selected datasets, grouped by:
  - sample scale
  - feature scale

Current likely message:

- The project uses a controlled benchmark design: sample-size scaling and feature-count scaling are separated, so each experiment changes one main variable at a time.

### 3. Algorithm Design

Purpose:

- Describe the models and the experimental pipeline.
- This section does not need to propose a new neural architecture; it should explain the design of our benchmark and analysis workflow.

Models:

- **TabICL**: a tabular in-context learning model. It uses examples as context and is expected to be strong on selected tabular tasks, but can be computationally heavier.
- **TabPFN**: a tabular foundation model designed for strong prediction on small to medium tabular datasets. It often performs well, but CPU inference can become slow as data size increases.
- **LightGBM**: a fast gradient-boosted tree baseline.
- **XGBoost**: another strong gradient-boosted tree baseline, usually fast and stable on tabular tasks.

Pipeline:

1. Load selected TALENT dataset.
2. Use train plus validation split as model fitting or context data.
3. Evaluate on the original test split.
4. Record performance, confidence, time, memory, and status.
5. Save raw results.
6. Aggregate results into final tables.
7. Generate figures after all experiments finish.

Recommended figures/tables:

- A simple pipeline diagram in the report or slides.
- `results/final/model_dataset_summary.csv`
- `results/final/device_type_summary.csv`

Current likely message:

- The core algorithmic contribution of this project is not a new model, but a reproducible benchmark pipeline for comparing modern tabular foundation models against classical tree baselines under limited compute.

Optional Big Plus idea:

- Retrieval-based in-context learning: choose more representative context examples for TabICL/TabPFN instead of using all or random examples. This can be presented as future work if not implemented.

### 4. Experiments

Purpose:

- Describe dataset selection, hardware setting, and experiment groups.

Dataset design:

- Main source: selected TALENT classification datasets.
- Two controlled experiment lines:
  - **Sample scale**: feature count is kept close, while sample size changes.
  - **Feature scale**: sample size is fixed around 2000 rows, while feature count changes using mfeat datasets.

Current dataset count:

- 12 selected datasets.
- 6 sample-scale datasets.
- 6 feature-scale datasets.

Models:

- TabICL
- TabPFN
- LightGBM
- XGBoost

Current hardware:

- Two light laptops and three gaming laptops have completed the experiment set.
- Each result set covers 48 model-dataset runs: 4 models times 12 selected datasets.
- One gaming-laptop result directory contains duplicate successful LightGBM/XGBoost runs, but the analysis pipeline keeps the latest successful run for each model-dataset pair. The final metrics table has no duplicate rows.

Recommended tables:

- Dataset selection table from `docs/dataset_selection_criteria.md`
- Completion table from `results/final/input_completeness_report.csv`

Recommended figures:

- `sample_scale_accuracy_line.png`
- `sample_scale_macro_f1_line.png`
- `feature_scale_accuracy_line.png`
- `feature_scale_macro_f1_line.png`

Current likely message:

- The selected datasets are not exhaustive, but they are intentionally chosen to expose different sample sizes and feature counts while staying feasible on CPU laptops.

### 5. Evaluation

This is the most important section. It should be organized by question, not by model.

#### 5.1 Overall Predictive Performance

Question:

- Which models are more accurate overall?

Recommended figures:

- `performance_accuracy_by_model.png`
- `performance_macro_f1_by_model.png`
- `performance_accuracy_by_dataset.png`
- `performance_macro_f1_by_dataset.png`

Current evidence from light-laptop results:

- Average accuracy:
  - TabICL: about 0.913
  - TabPFN: about 0.906
  - LightGBM: about 0.883
  - XGBoost: about 0.883
- Average macro F1:
  - TabICL: about 0.851
  - TabPFN: about 0.844
  - XGBoost: about 0.825
  - LightGBM: about 0.824

Likely final message:

- TabICL and TabPFN show stronger average predictive performance than LightGBM and XGBoost on the selected datasets.
- The advantage is more visible on some mfeat multiclass datasets, while on some binary datasets the difference is smaller.

#### 5.2 Sample-Size Scalability

Question:

- How do performance and cost change as sample size increases?

Recommended figures:

- `sample_scale_accuracy_line.png`
- `sample_scale_macro_f1_line.png`
- `sample_scale_predict_time_seconds_line_by_device.png`
- `sample_scale_wall_time_seconds_line_by_device.png`
- `sample_scale_peak_memory_bar_by_device.png`
- `sample_scale_loglog_time.png`
- Optional appendix: `sample_scale_loglog_memory.png`

Current evidence:

- TabICL and TabPFN remain competitive in accuracy, but their prediction time grows much faster with sample size.
- LightGBM and XGBoost are much faster on CPU.
- The time-based scalability exponent should be reported as an empirical trend. It supports the claim that foundation-model runtime grows faster with sample size than tree baselines.
- The memory-based scalability exponent should not be a main-text claim; use the peak-memory bar chart for memory discussion.

Likely final message:

- Foundation models gain accuracy but scale worse in CPU inference time as sample size grows.
- This is one of the central trade-offs discovered in the project.

#### 5.3 Feature-Count Scalability

Question:

- How do models behave when feature count increases while sample size stays close to 2000?

Recommended figures:

- `feature_scale_accuracy_line.png`
- `feature_scale_macro_f1_line.png`
- `feature_scale_predict_time_seconds_line_by_device.png`
- `feature_scale_wall_time_seconds_line_by_device.png`
- `feature_scale_peak_memory_bar_by_device.png`
- `feature_scale_loglog_time.png`
- Optional appendix: `feature_scale_loglog_memory.png`

Current evidence:

- TabICL and TabPFN generally perform strongly on mfeat feature-scale datasets.
- Their prediction and wall time increase with feature count.
- LightGBM and XGBoost remain very fast, and their memory stays nearly flat.
- Time-based `alpha` can be shown as a supplementary scalability indicator for feature count, but the report should avoid presenting it as a theoretical complexity estimate because the six mfeat datasets differ in feature representation, not only feature count.
- Memory `alpha` is not recommended for the main report. Use peak-memory bars instead.

Likely final message:

- Feature dimensionality increases the cost of foundation models more clearly than tree baselines.
- However, the performance advantage of TabICL/TabPFN on several mfeat datasets suggests that the extra cost can sometimes be worthwhile.

#### 5.4 Confidence and Error Behavior

Question:

- Are the models more confident when they are correct than when they are wrong?

Recommended figures:

- `confidence_histogram_by_model.png`
- `confidence_correct_vs_wrong.png`

Current likely message:

- All models tend to have higher confidence on correct predictions than on wrong predictions.
- Wrong predictions still often have non-trivial confidence, so confidence is useful but not a perfect reliability signal.
- This supports a short discussion of model calibration and decision risk.

#### 5.5 Confusion Matrix Analysis

Question:

- What types of multiclass errors do models make?

Recommended figures:

- Use only selected multiclass confusion matrices:
  - `confusion_matrix_tabicl_mfeat-morphological_2000rows_6feat_multiclass.png`
  - `confusion_matrix_tabpfn_mfeat-morphological_2000rows_6feat_multiclass.png`
  - `confusion_matrix_lightgbm_mfeat-morphological_2000rows_6feat_multiclass.png`
  - `confusion_matrix_xgboost_mfeat-morphological_2000rows_6feat_multiclass.png`
  - optionally the same four for `mfeat-pixel`

Recommended use:

- Do not put all 8 confusion matrices in the main report unless space allows.
- In the report, choose 2-4 representative matrices only if they support a concrete error-pattern conclusion.
- In the appendix, include the remaining matrices.

Current likely message:

- Confusion matrices are more informative for multiclass digit-recognition datasets than for binary yes/no datasets.
- The heatmaps should not be used merely to show diagonal dominance; accuracy and macro F1 already show that.
- The useful conclusion should focus on specific off-diagonal confusions. For example, low-dimensional mfeat variants may repeatedly confuse visually or morphologically similar digit classes across several model families.
- If the same confusion pair appears across TabICL, TabPFN, LightGBM, and XGBoost, interpret it primarily as a dataset or feature-representation limitation rather than a weakness of a single model.
- If the confusion largely disappears in `mfeat-pixel`, use that as evidence that richer feature representation reduces multiclass ambiguity.

#### 5.6 Hardware Sensitivity

Question:

- How much do results depend on the machine?

Current status:

- Light-laptop data from both members is complete.
- Gaming-laptop data from three additional result sets is complete.

Recommended figures:

- `sample_scale_predict_time_seconds_line_by_device.png`
- `feature_scale_predict_time_seconds_line_by_device.png`
- `sample_scale_wall_time_seconds_line_by_device.png`
- `feature_scale_wall_time_seconds_line_by_device.png`
- `sample_scale_peak_memory_bar_by_device.png`
- `feature_scale_peak_memory_bar_by_device.png`
- `results/final/device_consistency.csv`

Likely final message:

- Accuracy should remain almost unchanged across devices because the model and data split are the same.
- Time and memory are hardware-sensitive, so they should be shown by device rather than averaged blindly.
- Running the same experiments on two device classes makes the computational-cost analysis more rigorous.

### 6. Conclusion

Purpose:

- Summarize what the team learned or discovered.

Expected final points:

- Tabular foundation models can outperform traditional gradient-boosted baselines in predictive performance on selected TALENT classification datasets.
- This improvement comes with a substantial CPU-time and memory cost.
- TabICL is currently the strongest average performer in our results.
- TabPFN is also strong but can be particularly expensive in prediction time on larger sample groups.
- LightGBM and XGBoost remain extremely practical baselines because they are fast, memory-light, and stable.
- Scalability must be evaluated separately for sample size and feature count; otherwise, the reason behind cost growth is unclear.
- Confidence analysis shows that correct predictions tend to be more confident than wrong predictions, but confidence alone cannot fully explain errors.
- A practical lesson: under limited compute, tabular foundation models are attractive for accuracy-focused experiments, but tree baselines are still difficult to beat in efficiency.
- Confusion matrices should be used selectively: they are valuable only when they reveal concrete error pairs, especially in multiclass digit-recognition datasets.

Limitations:

- Dataset selection is representative, not exhaustive.
- Only classification tasks are included.
- No model fine-tuning is performed.
- Hardware comparison uses two device classes, but the devices are not laboratory-controlled machines. Timing results should therefore be interpreted as practical runtime evidence rather than exact hardware benchmarking.

Future work:

- Add regression or missing-value robustness experiments.
- Try retrieval-based context selection for foundation models.
- Add GPU or gaming-laptop comparison.

## Recommended Report Figure Placement

| Report section | Figure/table | Purpose |
| --- | --- | --- |
| Problem Definition | dataset selection table | Explain benchmark design |
| Experiments | `input_completeness_report.csv` summary | Show all planned experiments completed |
| Evaluation 5.1 | `performance_accuracy_by_model.png` | Overall accuracy comparison |
| Evaluation 5.1 | `performance_macro_f1_by_model.png` | Overall balanced performance comparison |
| Evaluation 5.1 | `performance_macro_f1_by_dataset.png` | Show dataset-level variation |
| Evaluation 5.2 | `sample_scale_macro_f1_line.png` | Performance across sample sizes |
| Evaluation 5.2 | `sample_scale_predict_time_seconds_line_by_device.png` | Sample-size time cost |
| Evaluation 5.2 | `sample_scale_loglog_time.png` | Empirical sample-size runtime scalability |
| Evaluation 5.2 | `sample_scale_peak_memory_bar_by_device.png` | Sample-size memory cost |
| Evaluation 5.3 | `feature_scale_macro_f1_line.png` | Performance across feature counts |
| Evaluation 5.3 | `feature_scale_predict_time_seconds_line_by_device.png` | Feature-count time cost |
| Evaluation 5.3 | `feature_scale_loglog_time.png` | Empirical feature-count runtime scalability |
| Evaluation 5.3 | `feature_scale_peak_memory_bar_by_device.png` | Feature-count memory cost |
| Evaluation 5.4 | `confidence_correct_vs_wrong.png` | Confidence reliability |
| Evaluation 5.5 | selected multiclass confusion matrices | Error pattern analysis only if concrete off-diagonal confusions are discussed |
| Appendix | all generated figures | Full evidence |
| Appendix | `sample_scale_loglog_memory.png`, `feature_scale_loglog_memory.png` | Supplementary memory scaling trends |

## 15-Minute Presentation Structure

Target: 12-14 slides. Keep the talk result-driven. Do not spend too long explaining every model internals.

### Slide 1. Title

Content:

- Project title
- Team members
- Course/project name

Time:

- 0:30

Figure:

- None, or a simple model-family diagram.

### Slide 2. Motivation

Main message:

- Tabular foundation models promise strong zero-shot or in-context performance on tabular data, but their real usefulness depends on cost and scalability.

Time:

- 1:00

Figure:

- Small comparison table: TabICL/TabPFN vs LightGBM/XGBoost.

### Slide 3. Research Questions

Questions:

- Are foundation models more accurate?
- Are they slower or more memory hungry?
- How do they scale with samples and features?
- What did we learn from confidence and errors?

Time:

- 1:00

Figure:

- No required figure.

### Slide 4. Benchmark Design

Main message:

- We used selected TALENT classification datasets and separated sample-scale and feature-scale experiments.

Time:

- 1:15

Figure/table:

- Dataset grouping table.

### Slide 5. Experimental Pipeline

Main message:

- All models use the same splits and the same evaluation protocol.

Time:

- 1:00

Figure:

- Pipeline diagram:
  - selected dataset
  - train plus validation context/training
  - model prediction
  - metrics and figures

### Slide 6. Overall Performance

Main message:

- TabICL and TabPFN have better average predictive performance than tree baselines on selected datasets.

Time:

- 1:30

Figures:

- `performance_accuracy_by_model.png`
- `performance_macro_f1_by_model.png`

### Slide 7. Dataset-Level Performance

Main message:

- The advantage is not uniform; some datasets show large gaps, others show smaller gaps.

Time:

- 1:15

Figure:

- `performance_macro_f1_by_dataset.png`

### Slide 8. Sample-Size Scalability

Main message:

- Foundation models remain accurate but prediction time grows much faster with sample size.

Time:

- 1:30

Figures:

- `sample_scale_macro_f1_line.png`
- `sample_scale_predict_time_seconds_line_by_device.png`
- Optional: `sample_scale_loglog_time.png`

### Slide 9. Feature-Count Scalability

Main message:

- Increasing feature count also increases the computational pressure on foundation models.

Time:

- 1:15

Figures:

- `feature_scale_macro_f1_line.png`
- `feature_scale_predict_time_seconds_line_by_device.png`
- Optional: `feature_scale_loglog_time.png`

### Slide 10. Memory Cost

Main message:

- TabICL has the highest memory pressure; tree models are much lighter.

Time:

- 1:00

Figure:

- `sample_scale_peak_memory_bar_by_device.png` or `feature_scale_peak_memory_bar_by_device.png`

### Slide 11. Confidence and Error Behavior

Main message:

- Correct predictions tend to have higher confidence than wrong predictions, but wrong predictions can still be confident.

Time:

- 1:00

Figure:

- `confidence_correct_vs_wrong.png`

### Slide 12. Confusion Matrix Case Study

Main message:

- Multiclass digit datasets are useful only if we discuss concrete off-diagonal error patterns.

Time:

- 1:00

Figures:

- One or two selected confusion matrices from `results/figures/confusion_matrix_selected/`

Recommended choice:

- `confusion_matrix_tabicl_mfeat-morphological_2000rows_6feat_multiclass.png`
- `confusion_matrix_xgboost_mfeat-morphological_2000rows_6feat_multiclass.png`

Speaker note:

- Do not simply say "the diagonal is strong." Instead, identify the largest off-diagonal confusions and explain whether they are shared by multiple models.
- Shared confusion across multiple model families should be framed as a feature-representation or dataset ambiguity issue.
- A comparison with `mfeat-pixel` can support the point that richer features reduce some digit-class confusions.

### Slide 13. What We Learned

Main message:

- Better accuracy is not free.
- Tabular foundation models are promising, but cost and scalability matter.
- Classical tree models remain very strong efficiency baselines.
- Controlled benchmark design matters: sample-size and feature-count effects should be separated.

Time:

- 1:00

Figure:

- Optional compact table of key findings.

### Slide 14. Conclusion and Future Work

Main message:

- TabICL/TabPFN are useful but computationally expensive on CPU.
- Future work: gaming-laptop/GPU comparison, retrieval-based context selection, missing-value robustness, regression tasks.

Time:

- 0:45

Figure:

- None, or a final trade-off table.

## Recommended Main Story for the Presentation

The talk should not be "we ran many models and here are many charts." A clearer story is:

1. New tabular foundation models promise higher accuracy.
2. We tested that promise under realistic limited-compute conditions.
3. They do improve average accuracy and macro F1 on our selected TALENT datasets.
4. But their CPU prediction time and memory cost are much higher.
5. The cost grows especially with sample size for TabPFN and TabICL.
6. Therefore, the practical answer is nuanced: use foundation models when accuracy matters and the dataset size is manageable; keep LightGBM/XGBoost as strong efficiency baselines.

## Finalization Notes After Gaming-Laptop Results

The report can now use both light-laptop and gaming-laptop results. However:

- Accuracy should be summarized across all result sets because the same data splits are used and the values should be stable.
- Time and memory should be shown by device class. Do not blindly average light-laptop and gaming-laptop timing into a single number when making hardware claims.
- If the gaming-laptop results reduce runtime but do not change the relative ranking, the final conclusion should emphasize that better hardware improves practicality but does not remove the accuracy-cost trade-off.
- For duplicated successful runs, the analysis uses the latest successful run per runner, model, dataset, and scale group.

