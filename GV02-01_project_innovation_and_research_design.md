# GV02-01 蓝藻 GvpA 序列生成：项目创新、模型规划与未来研究构想

> 项目：GV02-01 蓝藻 GvpA 序列生成
> 文档定位：为小组讨论、实验设计、阶段汇报和最终答辩提供统一的研究思路。
> 核心原则：创新不是简单增加模型数量，而是提出有依据、可验证、能解释的研究问题。

## 1. 项目的真正问题

本项目不是“输入几条蛋白质序列，训练一个模型，再输出一些新序列”。更准确的研究问题是：

> 在真实蓝藻 GvpA 数据很少、序列高度保守、结构和组装约束不完全可观测的情况下，生成模型能否学习到 GvpA 的有效序列分布，并在约束满足度、新颖性和候选集多样性之间取得可解释的平衡？

因此，项目的研究对象包含三个部分：

```text
真实蓝藻 GvpA 的数据规律
→ 生成模型学习到的规律
→ 生成结果满足约束的程度
```

最终输出不是简单的“最好模型”，而是：

- 哪些模型更容易保持 GvpA 家族和保守模式；
- 哪些模型更容易产生新颖序列；
- 哪些模型更容易发生模式塌缩或分布外生成；
- 是否存在忠实度、新颖性、多样性之间的平衡区域；
- 哪些候选值得进一步结构预测或实验验证。

## 2. “创新”应该如何理解

老师强调自由度高、需要有自己的理解，这意味着你们不应只照搬“自回归/VAE/GAN”这几个名称，而应回答以下问题：

1. 为什么选择这个方法？
2. 它解决了 GvpA 数据中的什么困难？
3. 它和其他方法相比有什么可检验的差异？
4. 如果结果失败，如何解释失败原因？
5. 这个结论是否超出了数据和评价方法能够支持的范围？

可以把创新分成四种层次：

| 创新类型 | 含义                                  | 本项目示例                               |
| -------- | ------------------------------------- | ---------------------------------------- |
| 问题创新 | 对任务提出更清晰、更有价值的研究问题  | 从“能生成”转向“约束下生成”           |
| 方法创新 | 针对数据困难设计新的模型或约束机制    | 保守位点感知的约束解码                   |
| 评价创新 | 不只看 identity，建立更合理的评价体系 | fidelity-novelty-diversity Pareto 分析   |
| 分析创新 | 对结果和失败模式提出新的解释          | 识别 profile 通过但 embedding 离群的样本 |

对于你们当前的数据规模，**评价和分析创新比盲目设计复杂模型更稳妥**。165 条主数据、95% identity 下 35 个簇、90% identity 下 7 个簇，不足以支撑大规模复杂模型的稳定比较，却足以支撑严谨的约束评价、消融实验和失败案例分析。

## 3. 当前项目的事实基础

主数据集为冻结的 `dataset_v1`：

| 项目                         |  当前情况 |
| ---------------------------- | --------: |
| 蓝藻 GvpA 序列               |    165 条 |
| train/valid/test             | 139/13/13 |
| 序列长度                     | 59--91 aa |
| 中位长度                     |     72 aa |
| 完全重复                     |         0 |
| PF00741 通过                 |   165/165 |
| 本地 GvpA core profile 通过  |   165/165 |
| MSA 异常                     |         0 |
| 跨集合 >=95% identity 泄漏对 |         0 |

当前已经具备：

- AA-frequency baseline；
- k-mer/Markov baseline；
- 统一 FASTA、tokenizer 和 dataset loader；
- LSTM 最小训练和生成流程；
- VAE 最小训练和生成流程；
- evaluator v0.1；
- `metrics.json`、`per_sequence_metrics.csv` 和模型汇总表。

当前仍缺少或未完成：

- 生成序列的 HMMER/PF00741 自动评价；
- 本地 GvpA core profile 批量评价接入 evaluator；
- 保守位点和氨基酸类别保持率；
- ESM-2 embedding、MMD 和离群分析；
- LSTM temperature 和 VAE beta/latent-scale 正式实验；
- 多随机种子和统计稳定性分析；
- 辅助预训练集的正式对照实验。

所以当前项目的主线应是“完善可信评价并回答研究问题”，而不是无限扩展模型数量。

## 4. 研究假设：不要想当然

建议在实验开始前写出可证伪的假设。这样即使结果和预期相反，也能形成有意义的结论。

### H1：自回归模型更容易学习局部顺序约束

理由：LSTM/Transformer 按 token 条件概率建模，理论上能学习相邻残基和较长程依赖。

可检验指标：

- profile 通过率；
- MSA 保守位点保持率；
- nearest-train identity；
- 生成序列长度分布；
- 与 k-mer baseline 的差异。

可能失败：数据太少时，LSTM 可能记忆训练序列，或者因为 EOS 学习不足生成异常长度。

### H2：VAE 更容易产生潜在空间中的连续变化

理由：VAE 通过 latent variable 表示序列，理论上可以在潜在空间采样和插值。

可检验指标：

- 不同 latent 的 unique ratio；
- latent interpolation 的序列变化是否平滑；
- pairwise diversity；
- profile 通过率是否随 latent scale 变化；
- KL loss 是否正常。

可能失败：posterior collapse 使 decoder 忽略 latent，导致不同 latent 生成相同或近似序列。

### H3：辅助泛 GvpA 预训练能改善分布学习，但可能削弱蓝藻特异性

理由：非蓝藻 GvpA 能增加可学习样本，但 Bacteria/Archaea 与 Cyanobacteria 存在分布差异。

可检验指标：

- 从头训练与预训练微调的 profile 通过率；
- 蓝藻 test 上的长度、组成和 embedding 分布；
- novelty 和 diversity；
- 是否更接近泛 GvpA、却偏离蓝藻 GvpA。

可能失败：预训练集带来 domain shift，模型学到了泛 GvpA 规律，却没有学好蓝藻目标分布。

### H4：约束解码能提高忠实度，但可能降低新颖性和多样性

理由：在生成过程中限制保守位点或 profile 评分，会减少无效序列。

可检验指标：

- fidelity 提升幅度；
- novelty 变化；
- pairwise diversity 变化；
- profile 通过率和 exact copy rate 的联合变化。

可能失败：硬约束过强，模型退化为已知序列复制。

### H5：ESM-2 表示空间能帮助发现“表面新颖但生物学异常”的序列

理由：PLM embedding 可以提供不同于逐位 identity 的语义/表示空间信息，但它不是功能证明。

可检验指标：

- 真实/生成 embedding 的 MMD 或 kernel distance；
- embedding 离群比例；
- profile 通过但 embedding 离群的样本数量；
- embedding 距离与 novelty、profile score 的关系。

可能失败：小样本下 embedding 分布估计不稳定，或模型表示对 GvpA 特异性不足。

## 5. 总体研究路线

推荐路线如下：

```text
问题定义与假设
→ 冻结数据和数据审计
→ EDA：长度、组成、MSA、聚类
→ 低复杂度 baseline
→ LSTM/VAE 基础模型
→ 统一 fidelity/novelty/diversity 评价
→ 参数权衡和消融实验
→ embedding 分布分析
→ Pareto 候选筛选
→ 失败案例和研究结论
```

每个阶段都要回答一个具体问题，不能只把阶段理解为“多跑一个脚本”。

## 6. 阶段一：问题定义与数据边界

### 6.1 明确生成目标

生成目标不是“所有 gas vesicle protein”，而是：

```text
Cyanobacteria 来源的 GvpA 主壳体蛋白候选
```

主数据集使用冻结的 `dataset_v1`。辅助 `aux_pan_gvpa_v1` 只能用于预训练对照。

### 6.2 明确证据层级

对真实数据和生成数据，都要区分不同证据：

| 证据              | 能说明什么            | 不能说明什么     |
| ----------------- | --------------------- | ---------------- |
| 合法字符          | 是可解析的蛋白序列    | 是 GvpA          |
| 长度范围          | 长度处于数据支持区间  | 能正确折叠       |
| PF00741           | 像 gas vesicle family | 一定是 GvpA      |
| GvpA core profile | 更接近目标 GvpA 分布  | 一定能组装气囊   |
| MSA 保守模式      | 保留真实序列模式      | 实验功能一定保留 |
| ESM-2 embedding   | 表示空间接近真实数据  | 真实功能已被证明 |
| GV01 功能模型     | 得到计算功能支持      | 替代湿实验       |

这个证据层级是项目中避免想当然的关键。

### 6.3 明确数据不确定性

主数据中有 3 条 provisional 来源记录：2 条 MAG 和 1 条 uncultured。建议将其作为敏感性分析对象，而不是默认为完全可靠或直接删除。

实验设置：

```text
主分析：完整 dataset_v1
敏感性分析：移除 3 条 provisional 记录
```

如果两组结果一致，说明主要结论对这些记录不敏感；如果差异明显，就应在最终报告中明确说明。

## 7. 阶段二：EDA 不是装饰，而是模型设计依据

### 7.1 长度分析

当前数据长度主要集中在 71--72 aa，整体范围为 59--91 aa。需要检查：

- train/valid/test 的长度差异；
- 长度与 cluster 的关系；
- 较长序列是否集中在 test；
- 生成模型是否偏向固定长度。

当前 test 平均长度高于 train，因此不能只用总体长度范围率评价生成质量，建议增加按长度区间分层的结果。

### 7.2 氨基酸组成分析

当前真实数据中 V、A、L、S、I 等比例较高，而 C、H、F 很少。EDA 应为以下工作提供依据：

- AA-frequency baseline 的合理性；
- composition distance 的解释；
- 生成序列的组成异常检测；
- 疏水性、电荷和极性分析。

### 7.3 MSA 和保守性分析

当前 MSA 有 165 条序列、115 个 alignment 列，其中覆盖至少 90% 序列的核心列有 71 个。建议输出：

- position-wise entropy；
- occupancy；
- gap rate；
- consensus identity；
- 具体氨基酸保守性；
- 氨基酸性质类别保守性。

创新点不应是简单声称“某个位点必须固定”，而应区分：

```text
严格保守位点
vs.
氨基酸性质保守位点
vs.
高度可变位点
```

这会直接影响后续软约束设计。

### 7.4 相似度聚类分析

当前 95% identity 下 35 个簇，90% identity 下 7 个簇，说明名义数据量大于独立模式数量。这个发现本身就是一个重要的项目结论：

> GvpA 生成任务不是普通的大样本序列生成，而是小样本、高保守、强记忆化风险下的约束生成。

## 8. 阶段三：为什么必须保留多个 baseline

### 8.1 AA-frequency baseline

它只学习：

- 长度分布；
- 单个氨基酸出现频率。

它可以回答：

> 深度模型是否学到了超过一阶氨基酸统计的信息？

如果深度模型的 profile 通过率、保守位点保持率和分布质量都不优于 AA-frequency，说明复杂模型没有带来可观测收益。

### 8.2 k-mer/Markov baseline

它学习局部顺序模式，可以回答：

> 仅仅建模局部 k-mer，能否获得接近深度模型的结果？

建议比较 `k=2/3/4`，但要避免只报告一个 k 值导致结论偶然。

### 8.3 Baseline 的创新用法

baseline 不只是“凑实验数量”，还可以作为分解工具：

| 对比                   | 可能说明的问题               |
| ---------------------- | ---------------------------- |
| AA-frequency vs k-mer  | 局部顺序模式是否有贡献       |
| k-mer vs LSTM          | 长程依赖是否有额外贡献       |
| LSTM vs VAE            | 自回归拟合与潜变量采样的差异 |
| 从头训练 vs 预训练微调 | 泛 GvpA 知识是否有迁移价值   |

## 9. 阶段四：当前模型训练方案

### 9.1 LSTM 自回归模型

#### 选择理由

- 结构和训练逻辑容易解释；
- 适合小序列、短长度生成；
- 可以直接调节 temperature、top-k/top-p；
- 适合研究局部与顺序依赖。

#### 输入输出

```text
输入：<BOS> M A V E ...
目标：M A V E ... <EOS>
```

使用 next-token cross entropy，loss 忽略 PAD。

#### 推荐实验变量

```text
temperature = 0.7 / 1.0 / 1.3
top-k = 0 / 5 / 10
```

模型容量不宜过大。当前数据量下，增加 hidden size 可能提高训练拟合，却降低真正的新颖生成能力。

#### 需要重点监控

- validation loss；
- 生成长度；
- exact copy rate；
- profile 通过率；
- novelty/diversity；
- temperature 改变后的 Pareto 位置。

### 9.2 VAE

#### 选择理由

- 可以显式引入 latent representation；
- 可以研究从一个序列模式平滑变化到另一个序列模式；
- 能够提供与 LSTM 不同的生成机制。

#### Loss

```text
total loss = reconstruction loss + beta * KL loss
```

必须单独记录 reconstruction、KL 和 total loss。

#### 推荐实验变量

```text
beta = 0.01 / 0.1 / 0.5
latent_scale = 0.5 / 1.0 / 1.5
```

#### 需要重点监控

- KL 是否迅速接近 0；
- 不同 latent 是否生成不同序列；
- latent interpolation 是否平滑；
- unique ratio；
- 最大重复比例；
- profile 通过率随 latent scale 的变化。

### 9.3 GAN 是否适合作为当前主线

GAN 可以作为未来扩展，但不建议在当前小样本两周主线上优先投入。原因是：

- 序列是离散 token，梯度传递更困难；
- GAN 训练不稳定；
- discriminator 很容易记忆少量真实序列；
- mode collapse 难以和数据本身高度保守区分；
- 即使生成结果新颖，也需要完整 profile 和结构代理评价。

如果后续做 GAN，应将问题定义为：

> 在相同训练数据和评价体系下，GAN 是否能在不降低 GvpA 忠实度的情况下提高候选集多样性？

而不是简单地把 GAN 作为“更高级模型”。

## 10. 统一评价框架：项目最重要的创新基础

### 10.1 忠实度 fidelity

忠实度表示生成结果对真实蓝藻 GvpA 规律的保持程度，包括：

- 合法字符；
- 长度；
- PF00741 和本地 GvpA profile；
- MSA 保守位点；
- 氨基酸性质类别；
- 疏水性、电荷和极性；
- ESM-2 embedding 分布；
- 功能预测支持。

### 10.2 新颖性 novelty

包括：

- exact copy rate；
- nearest-train identity；
- 多个 identity 阈值下的 novelty rate；
- query/target coverage。

### 10.3 多样性 diversity

包括：

- unique ratio；
- mean pairwise diversity；
- 最大重复比例；
- 生成序列相似性 cluster 分布。

### 10.4 不要把三个目标合成一个“万能分数”

建议使用：

```text
硬约束过滤
→ 在有效候选中比较 novelty/diversity
→ 绘制 Pareto 前沿
```

如果确实需要总分，只能作为排序辅助，必须同时保留各个原始指标，不能用总分替代分析。

## 11. 适合当前项目的创新点

### 创新点一：约束优先的生成质量评价

传统生成任务容易只关注“生成是否新”，你们可以明确提出：

> 对 GvpA 生成而言，新颖性必须建立在家族身份和结构相关约束已经通过的基础上。

落地方式：

```text
profile/长度/字符硬过滤
→ 保守模式和组成软评分
→ novelty/diversity 排序
```

意义：把评价从“生成像不像字符串”推进到“是否是可信的 GvpA 候选”。

### 创新点二：fidelity-novelty-diversity 三目标 Pareto 分析

不选择一个指标作为唯一冠军，而是研究不同模型在三目标空间中的位置。

落地方式：

- 每个模型生成相同数量；
- 使用相同 seed 方案和 evaluator；
- 先标记 profile invalid；
- 对有效候选绘制三目标散点图；
- 标记 Pareto 前沿；
- 输出保守型、平衡型、探索型候选。

意义：直接对应 RQ2，并且比“某模型指标最高”更有研究价值。

### 创新点三：把数据冗余纳入生成评价

你们的数据在 95% identity 下有 35 个簇，在 90% identity 下只有 7 个簇。可以提出：

> 生成模型的多样性不能只看候选之间的 pairwise distance，还要看候选是否覆盖真实数据中的多个相似性簇。

落地方式：

- 将生成序列与真实数据共同聚类；
- 检查生成样本落入哪些真实 cluster 邻域；
- 报告 cluster coverage；
- 区分“覆盖多个真实区域”和“远离所有真实区域”。

意义：避免把随机离群样本误判为多样性。

### 创新点四：辅助预训练的双刃剑分析

你们已有 `aux_pan_gvpa_v1`，不要只把它当作增加数据量的技巧，而应研究：

> 泛 GvpA 预训练是在帮助模型学习基本规律，还是把非蓝藻分布迁移进了蓝藻目标？

对照：

```text
蓝藻 train 从头训练
vs.
泛 GvpA 预训练 → 蓝藻 train 微调
```

同时比较：

- 蓝藻 profile fidelity；
- 蓝藻 test embedding 距离；
- novelty/diversity；
- 长度和组成分布。

### 创新点五：具体氨基酸保守与性质保守分离

不要把 MSA 保守性简单变成“这个位置只能是某个字母”。可以将位点分成：

```text
exact residue conserved
property conserved
variable
```

例如将疏水氨基酸归为一类，比较生成序列是否保留“性质”而不是只保留“字符”。

意义：更符合蛋白质序列-结构关系，也直接回应你们此前关于“低相似度仍可能保留功能”的思考。

### 创新点六：失败案例驱动的模型分析

不要只展示成功样本。建议建立失败类型：

| 失败类型                    | 可能原因                   |
| --------------------------- | -------------------------- |
| 长度异常                    | EOS 学习不足、解码策略问题 |
| profile 不通过              | 家族约束未学到或新颖性过度 |
| composition 异常            | 只学到局部模式或采样偏移   |
| embedding 离群              | 分布外生成                 |
| identity 过高               | 记忆化或 mode collapse     |
| diversity 极低              | latent/decoder collapse    |
| diversity 极高但 profile 低 | 无约束随机探索             |

意义：失败模式可以成为你们最有“自己的理解”的部分。

### 创新点七：参数不是调优旋钮，而是研究变量

temperature、top-k、VAE beta、latent scale 不应只用于找最高分，而应被解释为生成行为控制变量：

- temperature 增大是否提高 novelty，降低 fidelity？
- beta 增大是否改善 latent 结构，降低重构质量？
- latent scale 增大是否提高 diversity，产生更多离群样本？

这会把普通超参数实验转化为生成机制研究。

### 创新点八：可复现的模型无关评价管线

所有模型都输出同一 FASTA 格式，进入同一个 evaluator，使用同一组阈值和数据版本。

意义：

- 结果可直接比较；
- 新增模型不必重写评价逻辑；
- 外部工具失败时保留 `null/error`，不伪造分数；
- 更容易形成工程创新和可复现交付物。

## 12. 建议的消融实验

推荐消融链条：

```text
AA-frequency
→ k-mer
→ LSTM/VAE
→ + profile 约束筛选
→ + 保守位点软约束
→ + embedding 分布筛选
```

每一步都要回答“增加了什么”。

| 版本              | 研究问题                  | 主要指标                        |
| ----------------- | ------------------------- | ------------------------------- |
| AA-frequency      | 一阶组成能做到什么程度    | composition、diversity、profile |
| k-mer             | 局部顺序是否有贡献        | identity、profile、保守模式     |
| LSTM/VAE          | 深度生成是否超过 baseline | fidelity、novelty、diversity    |
| 加 profile 约束   | family 约束是否改善       | profile rate、novelty           |
| 加保守软约束      | 关键模式是否改善          | conservation、composition       |
| 加 embedding 筛选 | 是否减少分布外样本        | MMD、outlier rate               |

## 13. 未来模型创新路线

下面按“适合当前项目”和“适合后续扩展”分层。

### 13.1 短期可落地：约束解码

#### 硬约束解码

在生成时禁止某些明显不合法选择，例如：

- 长度超出范围；
- EOS 过早出现；
- 关键位点不允许出现不符合定义的残基类别。

优点：简单、可解释、容易做消融。
风险：硬约束过强会造成复制和多样性下降。

#### 软约束解码

对候选 token 的 logits 增加约束奖励或惩罚：

```text
logit' = logit
       + λ1 * 保守模式奖励
       - λ2 * profile 风险
       - λ3 * 组成偏离风险
```

优点：允许探索。
风险：profile 分数通常是整条序列或局部序列的结果，如何在线计算需要设计近似方法。

### 13.2 短期可落地：保守位点感知生成

根据 MSA 将位点分成严格保守、性质保守和可变三类，生成时使用不同强度的约束。

这比“一刀切地锁定所有保守位点”更合理：

```text
严格保守位点：强约束
性质保守位点：允许同类替换
可变位点：鼓励探索
```

### 13.3 中期：masked infilling / mutation generation

相比从 BOS 开始完整生成，可以给模型一条真实 GvpA 序列，只 mask 一部分位点，让模型生成局部变体。

适合研究：

- 单点或多点变异；
- 保守位点与可变位点的差异；
- 局部修改对 profile、embedding 和功能预测的影响；
- “最小改变获得最大新颖性”的设计。

它比从零生成更适合小样本，也更容易进行反事实分析。

### 13.4 中期：条件生成

可以把物种、taxonomic group、长度区间或目标 profile 状态作为条件。

核心问题不是“加入条件就算完成”，而是要验证：

> 条件信息是否真的改变了生成结果？

验证方法：

- 条件分类器预测准确率；
- 不同条件生成序列的 embedding 分布差异；
- 条件切换实验；
- 条件信息消融。

由于当前 organism 字段有高阶分类和未分类值，条件生成前必须先清理条件标签，不能从 accession 直接猜物种。

### 13.5 中期：domain-adaptive PLM

可以使用 ESM-2 embedding 作为输入特征，或在泛 GvpA 数据上进行轻量领域适配，再用于蓝藻 GvpA 生成/筛选。

推荐原则：

- 优先冻结大部分 PLM 参数；
- 只训练小型 adapter 或最后少量层；
- 使用严格 held-out test；
- 对比原始 ESM-2 与领域适配后的 embedding；
- 检查是否出现 domain overfit。

ESM-2 表示能够提供结构和进化相关信号，但并非所有保守结构域都能被单一 PLM 同等识别，因此不能把 embedding 分数当作唯一真值。它应作为 family/profile、MSA 保守模式和其他计算证据的补充，而不是替代这些证据。

### 13.6 中期：结构感知生成

后续可以将序列表示与结构信息结合：

- ESMFold/AlphaFold 结构预测；
- 二级结构标签；
- 3Di 或结构 token；
- β-sheet 相关约束；
- 结构相似度和置信度。

评价应关注：

- predicted β-sheet 比例；
- 结构模型置信度；
- 与真实 GvpA 结构的 TM-score/RMSD 等；
- 结构是否出现明显不合理的长无序区域或拓扑异常。

注意：结构预测结果也是模型预测，不是实验结构。结构约束可以提高证据强度，但不能替代实验。

### 13.7 长期：组装上下文建模

GvpA 的真实表型依赖 GvpC、GvpN、GvpF/L 等其他蛋白和基因簇环境。未来可以研究：

- GvpA 与 GvpC 的联合表示；
- GvpA-GvpC 界面约束；
- gvp 基因簇上下文作为条件；
- 蛋白-蛋白相互作用模型；
- 序列与操纵子结构联合建模。

这比单独生成 GvpA 更接近真实气囊组装，但数据要求和工作量会显著增加，不适合作为当前两周主线。

## 14. 未来评价创新路线

### 14.1 生成分布质量的 precision/recall

可以借鉴生成模型评价中的 precision/recall 思路：

- precision：生成样本中有多少落在真实 GvpA 分布附近；
- recall：真实 GvpA 的多少区域被生成样本覆盖。

这比单一 MMD 更容易区分：

```text
只生成真实分布中心
vs.
覆盖真实分布多个区域
```

不过小样本下估计会不稳定，需要使用 bootstrap 和 cluster-aware 分析。

### 14.2 多模型一致性作为不确定性

对同一候选使用多个评价器：

- HMMER/profile；
- MSA 保守评分；
- ESM-2 likelihood/embedding；
- GV01 功能模型；
- 结构预测模型。

如果所有评价器都支持，置信度较高；如果 profile 通过但 embedding、结构或功能模型不支持，应标记为不确定候选，而不是直接接受或拒绝。

### 14.3 反事实突变分析

对真实和生成序列逐位改变残基，观察：

- profile score 变化；
- ESM-2 likelihood 变化；
- 功能模型输出变化；
- 结构代理变化。

可以构建“模型认为重要的位点”与 MSA 保守位点的重叠率，分析模型是否学到了合理的约束。

### 14.4 约束预算曲线

将约束强度作为实验变量，绘制：

```text
约束强度
→ profile fidelity
→ novelty
→ diversity
```

这可以回答：

> 约束增加到什么程度后，生成模型开始从“探索”退化为“复制”？

### 14.5 主动学习闭环

未来如果有少量实验验证结果，可以采用：

```text
生成候选
→ 计算筛选
→ 选择高价值/高不确定性样本
→ 湿实验验证
→ 加入训练数据
→ 重新生成
```

选择策略可以同时考虑：

- 预测性能高；
- 模型不确定性高；
- 与已有候选差异大；
- 能覆盖新的 cluster 或 embedding 区域。

这会把项目从一次性生成升级为“生成-评价-验证”的闭环设计。

## 15. 建议的正式实验矩阵

### 实验 A：模型基础对比

| 方法         | 训练数据   | 变量              | 目的         |
| ------------ | ---------- | ----------------- | ------------ |
| AA-frequency | 蓝藻 train | seed              | 一阶组成基线 |
| k-mer        | 蓝藻 train | k=2/3/4           | 局部模式基线 |
| LSTM         | 蓝藻 train | temperature/top-k | 自回归生成   |
| VAE          | 蓝藻 train | beta/latent-scale | 潜变量生成   |

### 实验 B：辅助预训练对照

| 方法           | 训练方式                 | 目的         |
| -------------- | ------------------------ | ------------ |
| LSTM/VAE-FT    | 蓝藻 train 从头训练      | 主基线       |
| LSTM/VAE-PT-FT | 泛 GvpA 预训练后蓝藻微调 | 检验迁移价值 |

### 实验 C：约束消融

| 版本 | 约束                          |
| ---- | ----------------------------- |
| C0   | 无额外生成约束                |
| C1   | 长度和 EOS 约束               |
| C2   | profile 后过滤                |
| C3   | 保守位点软约束                |
| C4   | profile + 保守位点 + 分布筛选 |

### 实验 D：embedding 分布

比较真实 train/valid/test 与各模型生成集：

- PCA/t-SNE/UMAP；
- MMD/kernel distance；
- embedding 离群率；
- 真实 cluster 覆盖率。

### 实验 E：敏感性分析

至少进行：

- 移除 3 条 provisional 记录；
- 多随机种子；
- identity 阈值变化；
- profile 阈值变化；
- 生成数量变化。

## 16. 模型选择原则

不要预先假定 Transformer、GAN 或更大模型一定更好。建议使用以下决策门：

### 选择 LSTM 作为主模型的条件

- 训练稳定；
- 长度控制可用；
- profile 通过率不低于 baseline；
- 有可解释的 temperature 权衡；
- 结果可以稳定复现。

### 选择 VAE 作为主模型的条件

- latent 变化能影响生成结果；
- 没有严重 posterior collapse；
- diversity 提升不是以 profile 失效为代价；
- beta/latent-scale 变化呈现可解释规律。

### 暂不增加 GAN 的条件

- 当前 profile evaluator 尚未完成；
- LSTM/VAE 尚未完成正式多 seed；
- 当前结果还无法解释；
- GAN 只会增加模型数量，却不能回答新的问题。

模型多不等于研究强。一个完成约束评价、消融和失败案例分析的 LSTM/VAE 项目，通常比四个没有统一评价的模型更有说服力。

## 17. 风险清单和应对方式

| 风险           | 表现                                  | 应对                                     |
| -------------- | ------------------------------------- | ---------------------------------------- |
| 数据太少       | 训练 loss 下降但生成复制              | 降低模型容量、多 seed、报告 cluster 数量 |
| 长度失控       | 大量达到 max_len 或过短               | EOS/长度控制、按长度评价                 |
| profile 不通过 | novelty 高但不像 GvpA                 | 约束解码、后过滤、失败案例分析           |
| mode collapse  | unique ratio 低                       | VAE KL 调整、采样参数、报告最大重复比例  |
| 随机离群       | diversity 高但 embedding/profile 异常 | 约束优先、分布筛选                       |
| 预训练负迁移   | 泛 GvpA 接近但蓝藻偏离                | 与从头训练对照、冻结部分参数             |
| 指标互相矛盾   | identity、profile、embedding 不一致   | 采用多证据分层，不强行合成单分数         |
| test 过小      | 单次结果波动大                        | bootstrap、多个 seed、谨慎表述           |
| 工具阈值依赖   | 换阈值结论改变                        | 记录版本，做阈值敏感性分析               |

## 18. 最终报告应该讲出的故事

推荐最终叙事顺序：

1. GvpA 数据少且高度保守，普通随机生成不足以解决问题；
2. 我们先冻结并审计目标数据，控制近重复泄漏；
3. 通过 AA-frequency 和 k-mer baseline 分离组成规律与局部顺序规律；
4. 用 LSTM 和 VAE 比较不同生成机制；
5. 发现单一 identity 不能代表生成质量，因此引入 fidelity、novelty、diversity 多维评价；
6. 用 profile、MSA、组成和 embedding 判断生成结果是否仍处于 GvpA 合理区域；
7. 通过 temperature、beta、latent-scale 或约束消融研究目标权衡；
8. 给出保守型、平衡型和探索型候选；
9. 对失败案例进行分类，说明模型到底在哪些环节失效；
10. 明确计算结果不能替代结构预测和湿实验。

这个叙事能够体现你们自己的理解：

> 生成任务的价值不在于输出更多序列，而在于理解“哪些新序列仍然可信、为什么可信、证据有多强”。

## 19. 可直接使用的创新点表述

### 项目总体创新

> 本项目不将蓝藻 GvpA 序列生成简化为无约束的字符串采样，而是将其建模为小样本、高保守蛋白家族中的约束生成问题。我们在统一生成管线中联合评价 GvpA 家族忠实度、新颖性、候选集多样性和真实数据分布一致性，并通过参数权衡、辅助预训练对照和失败案例分析研究不同生成策略的行为差异。

### 方法与评价创新

> 项目将 family/profile 和保守模式作为生成质量的前置约束，将 sequence identity 主要用于检测复制和衡量新颖性，而不把 identity 直接等同于功能相似度。在此基础上，通过 Pareto 前沿区分保守型、平衡型和探索型候选，避免单一指标造成错误排序。

### 未来工作创新

> 后续可进一步引入保守位点感知的软约束解码、masked infilling、条件生成、ESM-2 领域适配、结构感知生成以及基于实验反馈的主动学习闭环，使系统从“无约束序列生成”逐步发展为“可解释的多目标蛋白设计系统”。

## 20. 结论边界

即使某条序列同时满足：

- 合法字符和长度约束；
- PF00741 和 GvpA core profile；
- 保守模式合理；
- embedding 位于真实分布附近；
- novelty 和 diversity 指标良好；

也只能说：

> 该候选在计算层面具有较强的 GvpA-like 特征，值得进一步结构或实验验证。

不能仅凭本项目声称：

- 一定能正确表达；
- 一定能正确折叠；
- 一定能与其他 Gvp 蛋白组装；
- 一定能形成真实气囊；
- 一定具有已被实验确认的功能。

## 21. 执行优先级

### 当前两周主线

1. 保持 `dataset_v1` 冻结；
2. 完成 LSTM/VAE 正式多 seed 实验；
3. 修复生成长度控制；
4. 把 HMMER/profile 扫描接入 evaluator；
5. 完成 fidelity、novelty、diversity 对比；
6. 完成 temperature、beta/latent-scale 权衡；
7. 输出失败案例和 Pareto 图；
8. 完成一次干净环境复现。

### 下一阶段优先扩展

1. 保守位点和氨基酸性质软约束；
2. ESM-2 embedding/MMD 和离群分析；
3. 从头训练与辅助预训练对照；
4. masked infilling 变体生成；
5. 结构预测和 β-sheet 相关分析。

### 长期扩展

1. 条件生成；
2. domain-adaptive PLM；
3. 结构感知生成；
4. GvpA-GvpC 组装上下文；
5. 主动学习和湿实验闭环；
6. 多目标优化器和不确定性驱动候选选择。

## 22. 相关项目文件

- [数据集详细说明](./GV02-01_dataset_detailed_description.md)
- [序列相似度与约束评价](./GV02-01_sequence_similarity_and_constraints.md)
- [忠实度与生成质量评价改进](./GV02-01_fidelity_and_generation_evaluation_improvement.md)
- [当前已完成内容](./GV02-01_2026-09-10完成内容.md)
- [流水线任务计划](./GV02-01_蓝藻GvpA序列生成_流水线版两周任务计划_2026-09-06至09-19.md)
- [统一评价器](./gvpa-sequence-generation/evaluation/evaluate_all.py)
- [主数据集冻结报告](./gvpa-sequence-generation/data/processed/dataset_v1/dataset_report.md)
- [辅助预训练集报告](./gvpa-sequence-generation/data/processed/aux_pan_gvpa_v1/dataset_report.md)

## 23. 延伸参考方向

以下工作可作为后续阅读方向，重点关注其评价方法和约束建模思想，不建议未经验证地直接照搬到当前项目：

- ESM-2 与蛋白质表示、突变效应和进化约束分析；
- 蛋白质序列生成中的 naturalness、novelty、diversity 和 fidelity 评价；
- masked protein language model 的局部变体生成；
- 结构感知蛋白质语言模型；
- 蛋白质生成模型的结构质量和分布评价。
