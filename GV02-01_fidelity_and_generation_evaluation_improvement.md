# GV02-01 忠实度与生成质量评价改进说明

> 项目：GV02-01 蓝藻 GvpA 序列生成  
> 文档用途：统一解释“忠实度（fidelity）”在本项目中的含义，并建立可执行的生成序列评价体系。  
> 评价对象：随机/频率基线、k-mer、LSTM、VAE 及后续其他生成模型。

## 1. 核心结论

本项目中的**忠实度**，指生成序列对真实蓝藻 GvpA 数据规律和已知生物学约束的保持程度。它不是“生成序列必须和某条真实序列完全一样”，也不是单纯的 sequence identity。

更准确地说，忠实度回答的是：

> 生成模型是否真正学到了真实 GvpA 的序列分布、家族身份、保守模式、物理化学特征和结构相关约束，而不是只生成合法的氨基酸字符串，或者简单复制训练集。

对 GV02-01，一个有价值的生成模型应同时满足：

```text
忠实度较高
+ 新颖性合理
+ 候选集多样性合理
```

这三个目标不能被压缩为单一的“越高越好”：

- 忠实度太低：生成结果可能已经不像 GvpA；
- 忠实度过高但新颖性很低：模型可能只是复制训练集；
- 新颖性和多样性很高但忠实度很低：模型可能只是在生成随机离群序列。

因此，本项目应采用“硬约束过滤 + 多维度评价 + Pareto 权衡”的方式，而不是依赖一个总分或单一相似度阈值。

## 2. 忠实度在 GvpA 任务中的具体含义

GvpA 是气囊的主要壳体蛋白。生成序列至少应在以下层面保持对真实蓝藻 GvpA 的忠实。

### 2.1 序列格式忠实度

生成结果需要满足最基本的蛋白质序列要求：

- 只包含 20 种标准氨基酸；
- 不包含空序列；
- 长度处于真实 GvpA 支持的合理范围；
- 没有异常终止或明显截断；
- 能够被统一 FASTA 解析器正确读取。

这属于最基础的忠实度，只能证明“它是一条合法的蛋白质字符串”，不能证明它是 GvpA。

### 2.2 GvpA 家族忠实度

需要判断生成序列是否仍属于目标 GvpA/GvpA-like family，而不是其他 Gvp 蛋白或仅仅命中宽泛 family 的离群序列。

建议使用：

- 官方 `PF00741 / Gas_vesicle` profile；
- 本地高可信 `GvpA_core95` profile；
- profile score；
- profile coverage；
- 序列长度；
- GvpA 相关保守模式。

基本指标为：

```text
Family fidelity =
通过 GvpA profile 的生成序列数
/
生成序列总数
```

例如，100 条生成序列中有 85 条同时通过指定的 family/profile 条件，则 family fidelity 为 0.85。

需要特别注意：

> `PF00741` 是较宽泛的 Gas Vesicle family，不能单独证明序列一定是 GvpA。正式判定需要结合本地 GvpA profile、长度、保守模式以及其他证据。

当前项目的真实 `dataset_v1` 中，165 条序列均通过 PF00741 和本地 GvpA core profile；但生成结果的 family/profile 评价尚未真正接入 evaluator，当前相关字段仍为 `not_run`。

### 2.3 保守模式和结构代理忠实度

生成序列不需要每个位置都与某条真实序列完全相同，但应保留关键结构和物理化学模式。

可评价：

- MSA 保守位点一致性；
- 氨基酸性质类别保持率；
- 核心列 occupancy；
- position-wise entropy；
- 局部 motif 破坏率；
- 疏水性分布；
- 正负电荷分布；
- β-sheet 相关结构代理指标。

保守位点不一定要求同一个具体氨基酸。例如真实 MSA 中某个位点主要由 `V/I/L/M` 构成时，可以把它解释为疏水性类别约束，而不是要求生成序列必须使用其中某一个字母。

可以定义：

```text
Conservation fidelity =
满足保守位点或氨基酸性质约束的位点数
/
参与评价的保守位点总数
```

### 2.4 真实分布忠实度

要判断生成序列是否落在真实蓝藻 GvpA 的合理分布区域，而不是完全跑到分布外。

可比较：

- 长度分布；
- 氨基酸组成分布；
- 疏水性和电荷分布；
- MSA 模式；
- ESM-2 embedding 分布；
- MMD；
- kernel distance；
- embedding 空间离群比例。

一般而言，真实序列和生成序列在特征空间中的 MMD 或核距离越小，说明分布更接近。但这些是分布层面的计算证据，不是实验功能证明。

### 2.5 功能忠实度

如果接入 GV01-02/GV01-03 的功能预测模型，可以额外评估：

- 预测为 GvpA-like 的比例；
- 功能预测置信度；
- 关键位点是否保留；
- 生成序列的预测功能分布是否接近真实 GvpA；
- 是否出现大量功能离群样本。

可以定义：

```text
Functional fidelity =
被功能模型预测为 GvpA-like 的序列数
/
生成序列总数
```

但这仍然是模型的间接验证，不能等价于真实表达、折叠、组装或湿实验功能。

## 3. 忠实度与其他指标的区别

| 指标 | 主要回答的问题 | 是否等于忠实度 |
| --- | --- | --- |
| Sequence identity | 与已知序列逐位有多像 | 否 |
| Novelty | 是否避免复制已知序列 | 否 |
| Diversity | 候选之间是否彼此不同 | 否 |
| Fidelity | 是否保留真实 GvpA 的规律和约束 | 是，本文核心概念 |
| Functional prediction | 是否被模型预测为具有目标功能 | 是忠实度的间接证据之一 |

### 3.1 identity 高不一定代表忠实度高

如果生成序列与训练集 identity 很高，可能只是复制了训练集。它的序列相似度高，但生成能力和新颖性不足。

### 3.2 identity 低不一定代表忠实度低

如果生成序列 identity 较低，但仍然：

- 通过 GvpA profile；
- 保留关键保守模式；
- 组成和结构代理合理；
- 位于真实 GvpA embedding 分布附近；

那么它可能是具有价值的新颖候选。

### 3.3 diversity 高不一定代表模型好

如果候选之间差异很大，但大量序列不通过 family/profile，或者 embedding 明显离群，那么这种多样性只是无效探索。

## 4. 忠实度、新颖性和多样性的三目标关系

| 目标 | 过低时的问题 | 过高时的问题 |
| --- | --- | --- |
| 忠实度 | 生成结果不像 GvpA，可能无法满足结构约束 | 可能退化成训练集复制 |
| 新颖性 | 没有新的探索价值 | 可能破坏保守结构和家族身份 |
| 多样性 | 候选集模式塌缩 | 可能产生大量随机或分布外序列 |

理想目标是：

```text
在忠实度达到可接受门槛后，
尽可能提高新颖性和候选集多样性。
```

因此，模型比较应先过滤掉不满足 GvpA 硬约束的序列，再在有效候选中比较 novelty 和 diversity。

## 5. 忠实度的分层指标体系

### 5.1 一级：硬忠实度

这些指标是必要条件，不满足时不应把序列作为合格 GvpA 候选。

| 指标 | 解释 | 目标 |
| --- | --- | --- |
| `legal_char_rate` | 标准氨基酸字符比例 | 越高越好，理想为 1 |
| `length_in_train_range_rate` | 长度落入训练范围的比例 | 越高越好 |
| PF00741 hit rate | 通过官方 Gas Vesicle profile 的比例 | 越高越好 |
| GvpA core profile hit rate | 通过本地 GvpA profile 的比例 | 越高越好 |
| profile coverage | profile 对序列的覆盖程度 | 越高越好，但需结合长度解释 |
| exact copy rate | 与训练集完全复制的比例 | 不能过高 |

### 5.2 二级：模式忠实度

| 指标 | 解释 | 目标 |
| --- | --- | --- |
| 保守位点一致性 | 是否保持 MSA 核心位置模式 | 越高越好 |
| 氨基酸类别保持率 | 疏水、带电、极性等性质是否保留 | 越高越好 |
| 核心列 occupancy 偏差 | 生成序列在核心列的覆盖偏差 | 越小越好 |
| 局部 motif 破坏率 | 关键局部模式被破坏的比例 | 越低越好 |
| 疏水性分布距离 | 生成与真实疏水性模式差异 | 越小越好 |
| 电荷分布距离 | 生成与真实电荷模式差异 | 越小越好 |

### 5.3 三级：分布忠实度

| 指标 | 解释 | 目标 |
| --- | --- | --- |
| 长度分布距离 | 真实与生成长度分布差异 | 越小越好 |
| AA composition L1 | 氨基酸组成差异 | 越小越好，但不能单独判断质量 |
| ESM-2 MMD | embedding 分布差异 | 越小越好 |
| Kernel distance | 表示空间核距离 | 越小越好 |
| embedding 离群率 | 远离真实分布的比例 | 越低越好 |

### 5.4 四级：功能忠实度

| 指标 | 解释 | 目标 |
| --- | --- | --- |
| GvpA-like 预测率 | 被功能模型判定为 GvpA-like 的比例 | 越高越好 |
| 功能预测置信度 | 预测可信程度 | 越高越好 |
| 关键位点保留率 | 已知功能相关位点保留比例 | 越高越好 |
| 预测功能分布距离 | 生成与真实功能预测分布差异 | 越小越好 |

## 6. 推荐的两步判定流程

### 6.1 第一步：硬门槛过滤

对每条生成序列先进行：

```text
标准氨基酸检查
+ 长度检查
+ PF00741/profile 检查
+ 本地 GvpA core profile 检查
+ exact copy 检查
```

不满足硬门槛的序列标记为 invalid，不参与后续优质候选排名。

### 6.2 第二步：有效候选综合排序

对通过硬门槛的序列比较：

```text
保守位点保持程度
+ 氨基酸组成距离
+ 疏水性和电荷合理性
+ embedding 分布位置
+ 新颖性
+ 候选集多样性
```

最终可以将候选分为：

| 类型 | 典型特征 | 适合用途 |
| --- | --- | --- |
| 保守型 | profile 高、保守位点保持好、identity 较高 | 可靠性对照 |
| 平衡型 | profile 高、新颖性适中、多样性合理、分布接近真实数据 | 主要候选 |
| 探索型 | novelty 和 diversity 高，但部分结构/分布证据不足 | 后续结构或实验验证 |
| 失败/离群型 | profile 低、长度异常、组成偏离或 embedding 离群 | 失败模式分析 |

高新颖但 profile、组成或 embedding 明显异常的序列，不能直接作为最佳候选。

## 7. 当前 evaluator 已实现的内容

当前统一评价器位于：

```text
gvpa-sequence-generation/evaluation/evaluate_all.py
```

已经实现：

- `legal_char_rate`；
- `length_in_train_range_rate`；
- `exact_copy_train_rate`；
- `exact_copy_any_real_rate`；
- `mean_nearest_train_identity`；
- `max_nearest_train_identity`；
- `novel_sequence_rate`；
- `unique_ratio`；
- `mean_pairwise_diversity`；
- `max_duplicate_fraction`；
- `aa_composition_l1_distance`；
- 每条序列的 `per_sequence_metrics.csv`；
- 每个 run 的 `metrics.json`；
- 多个 run 的 `model_comparison.csv`。

当前尚未真正接入：

- HMMER PF00741 自动扫描；
- 本地 GvpA core profile 自动扫描；
- MSA 保守位点评分；
- 疏水性和电荷分布评分；
- ESM-2 embedding；
- t-SNE/UMAP 可视化；
- MMD 或 kernel distance；
- GV01 功能预测模型评价。

因此，当前 evaluator 可以完成初步的合法性、新颖性、多样性和组成评价，但还不能完整判断生成序列的 GvpA 忠实度。

## 8. 当前 baseline 结果如何理解

当前已有 baseline 结果：

| 模型 | 平均最近邻 identity | 平均 pairwise diversity | AA composition L1 |
| --- | ---: | ---: | ---: |
| AA-frequency | 0.5583 | 0.5480 | 0.0294 |
| k-mer | 0.7295 | 0.3792 | 0.0817 |

### 8.1 AA-frequency baseline

它的特点是：

- nearest-train identity 较低；
- 候选之间多样性较高；
- AA composition distance 较小；
- 可能学到了氨基酸频率；
- 未必学到了局部顺序、保守位点和 GvpA 结构模式。

因此不能因为它 diversity 高、composition distance 小，就直接判定忠实度高。

### 8.2 k-mer baseline

它的特点是：

- nearest-train identity 较高；
- 候选内部多样性低于 AA-frequency；
- 可能更好地保留局部序列模式；
- 也可能更接近训练集，存在更高记忆化风险；
- composition distance 当前反而更大。

因此也不能因为它 identity 高，就直接判定忠实度高。

### 8.3 当前结论边界

在 HMMER/profile、保守位点和结构代理指标接入之前，只能说：

> AA-frequency 更偏向新颖性和候选集多样性；k-mer 更偏向局部模式保持和真实序列邻近性。两者谁具有更高的 GvpA 忠实度，目前尚不能正式判定。

LSTM/VAE 当前仍属于 smoke test，生成数量较少，并且存在长度控制问题，不能据此做正式模型优劣结论。

## 9. 忠实度对 RQ1--RQ3 的作用

### 9.1 RQ1：哪种生成策略最善于保持结构约束？

RQ1 的核心指标不应是单独的 sequence identity，而应比较：

- family/profile fidelity；
- 长度和合法字符通过率；
- MSA/保守位点 fidelity；
- 疏水性、电荷等结构代理；
- 相同生成量和随机种子条件下的有效候选率。

可能出现：

- identity 较低但 profile 和保守模式很好：有价值的新颖候选；
- identity 较高但大量 exact copy：模型记忆化；
- diversity 很高但 profile 很低：无效探索。

### 9.2 RQ2：约束、新颖性、多样性是否存在甜点？

忠实度应作为三目标权衡中的安全边界：

```text
先保证 fidelity 达到可接受门槛
再在有效候选中比较 novelty 和 diversity
```

应绘制：

- fidelity vs novelty；
- fidelity vs diversity；
- novelty vs diversity；
- 三目标 Pareto 前沿；
- 不同 temperature、VAE beta、latent scale 下的变化。

重点回答：

- 新颖性提高到什么程度后 profile 通过率开始下降；
- 哪个模型能在不明显损失忠实度的情况下提高多样性；
- 是否存在同时满足约束、具有新颖性且不过度趋同的参数区域。

### 9.3 RQ3：PLM embedding 分布揭示什么？

embedding fidelity 可以帮助回答：

- 生成样本是否覆盖真实 GvpA 的多个分布区域；
- 是否只挤在真实分布中心；
- 是否出现明显分布外样本；
- 是否存在 profile 通过但 embedding 很异常的序列；
- MMD 变化是否和 profile、novelty、diversity 一致。

t-SNE/UMAP 主要用于可视化，二维距离不能直接作为严格生物学距离。MMD 和 kernel distance 在本项目中属于探索性分布指标，不能替代 family/profile 或实验验证。

## 10. 推荐最终模型比较表

每个模型至少报告：

| 类别 | 指标 |
| --- | --- |
| 合法性 | `legal_char_rate` |
| 长度 | `length_in_train_range_rate`、平均长度、长度分布距离 |
| 家族约束 | PF00741 通过率、本地 GvpA profile 通过率、profile coverage |
| 复制风险 | exact copy rate、nearest-train identity |
| 新颖性 | identity < 0.90/0.95/0.98 的 novelty rate |
| 覆盖 | query coverage、target coverage |
| 多样性 | unique ratio、pairwise diversity、最大重复比例 |
| 组成 | AA composition L1 distance |
| 保守性 | MSA 保守位点一致性、氨基酸类别保持率 |
| 结构代理 | 疏水性、电荷、β-sheet 相关指标 |
| 分布 | ESM-2 MMD、kernel distance、离群比例 |
| 功能 | GvpA-like 预测率、预测置信度、关键位点保留率 |
| 稳定性 | 多随机种子均值、标准差和置信区间 |

## 11. 推荐实现优先级

结合当前项目状态，建议按以下顺序补齐：

### P0：立即完成

1. 统一生成数量、随机种子、FASTA header 和 evaluator 参数；
2. 修正 LSTM/VAE 的长度控制问题；
3. 对每个模型运行多个 seed，而不是只看单次输出；
4. 报告现有合法性、长度、复制、新颖性、多样性和组成指标。

### P1：最重要的生物学评价

1. 将 HMMER PF00741 扫描接入 evaluator；
2. 将本地 GvpA core profile 扫描接入 evaluator；
3. 输出每条序列的 profile score、E-value、coverage 和 pass/fail；
4. 计算真实数据和生成数据的 profile score 分布。

### P2：保守模式与物理化学约束

1. 从真实 MSA 提取核心列和保守位点；
2. 计算具体氨基酸保留率和氨基酸类别保持率；
3. 计算疏水性、电荷和极性分布；
4. 标记保守模式明显破坏的序列。

### P3：表示空间和功能辅助评价

1. 接入 ESM-2 embedding；
2. 生成真实/生成样本 t-SNE 或 UMAP 图；
3. 计算 MMD 或 kernel distance；
4. 接入 GV01 功能预测模型；
5. 对高 novelty、高 fidelity 和离群候选分别抽样分析。

## 12. 报告中推荐使用的定义

可以直接使用以下表述：

> 本项目中的忠实度，是指生成序列在合法性、GvpA 家族身份、长度范围、保守位点、氨基酸组成、物理化学模式和表示空间分布等方面，对真实蓝藻 GvpA 数据规律的保持程度。忠实度不等同于与某条已知序列的高 identity，也不等同于真实实验功能。我们将忠实度与新颖性和候选集多样性联合评价，以避免生成模型在复制训练集和生成无效离群序列之间偏向任一端。

还可以简化为：

> 忠实度决定生成结果是否仍然像一个可信的 GvpA；新颖性决定它是否不只是复制已知序列；多样性决定模型能否提供一组彼此不同的候选。

不建议使用：

> 生成序列与真实序列越相似，忠实度就越高。

也不建议使用：

> 生成序列与已知序列越不相似，说明模型越成功。

这两种表述都把忠实度错误地等同于单一方向的序列相似度。

## 13. 项目边界

即使某条生成序列同时满足：

- profile 通过；
- 保守模式合理；
- embedding 位于真实分布附近；
- novelty 和 diversity 指标良好；

也只能说明它在计算层面具有较强的 GvpA-like 特征，不能仅凭这些结果声称：

- 一定能够正确表达；
- 一定能够正确折叠；
- 一定能够与其他 Gvp 蛋白组装；
- 一定能够形成真实气囊；
- 一定具有实验上验证的目标功能。

最终候选仍需要进一步结构预测、功能验证或湿实验。

## 14. 相关项目文件

- [数据集详细说明](./GV02-01_dataset_detailed_description.md)
- [序列相似度与约束评价](./GV02-01_sequence_similarity_and_constraints.md)
- [当前已完成内容](./GV02-01_2026-09-10完成内容.md)
- [两周流水线任务计划](./GV02-01_蓝藻GvpA序列生成_流水线版两周任务计划_2026-09-06至09-19.md)
- [统一评价器](./gvpa-sequence-generation/evaluation/evaluate_all.py)
- [主数据集冻结报告](./gvpa-sequence-generation/data/processed/dataset_v1/dataset_report.md)
- [MSA 报告](./gvpa-sequence-generation/data/processed/dataset_v1/audit/msa_report.md)
- [profile 复核报告](./gvpa-sequence-generation/data/processed/dataset_v1/audit/profile_review_report.md)

