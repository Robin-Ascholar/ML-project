# RQ1：不同生成策略对 Pfam 家族约束的保持能力

## 1. 研究问题

**RQ1：不同生成策略（自回归、VAE、GAN）产出的序列在 Pfam 域约束满足度上的通过率有何差异？哪种策略更善于保持结构约束？**

本问题比较三类序列生成策略在生成 GvpA-like 蛋白序列时，是否仍然满足预先定义的 GvpA 家族约束。这里的“结构约束”首先指**序列层面的家族/profile 约束**，而不是已经由实验或三维结构计算证明的折叠结构。

本项目使用 Pfam `PF00741` 作为家族层面的核心筛选依据，并结合序列合法性和长度范围定义忠实性（faithfulness）。

---

## 2. 实验对象与数据基础

### 2.1 冻结数据集

模型使用项目冻结的 `dataset_v1`：

| 项目 | 数值 |
|---|---:|
| 蓝藻 GvpA 序列总数 | 165 |
| 训练集 | 139 |
| 验证集 | 13 |
| 测试集 | 13 |
| 训练序列长度范围 | 59--91 aa |

该数据集经过序列清洗、分类学检查、注释检查、重复检查、profile 检查和数据划分审计。模型训练使用固定的 `train.fasta`，没有在模型之间重新随机划分数据。

### 2.2 Pfam profile

使用官方 Pfam `PF00741` profile，对每条生成序列运行 HMMER `hmmsearch`。正式忠实性评价采用：

```text
E-value <= 1e-3
```

PF00741 对应广义的 gas-vesicle protein family。它可以检验生成序列是否保留该家族的统计特征，但 PF00741 单独不能证明：

- 序列一定是严格意义上的 GvpA；
- 蛋白一定能正确折叠；
- 蛋白一定能表达、组装或形成气囊；
- 蛋白一定具有实验验证的生物学功能。

因此，本文将 PF00741 命中解释为**家族/profile 约束满足度**，而不直接等同于真实三维结构正确率。

---

## 3. 忠实性指标定义

对每条生成序列，检查以下三个条件：

1. **合法字符条件**：只包含 20 种标准氨基酸；
2. **长度条件**：序列长度位于训练集范围 `59--91 aa`；
3. **家族 profile 条件**：通过 PF00741 的 HMMER 扫描，且 E-value 不超过 `1e-3`。

最终忠实性定义为：

```text
faithful = legal_chars AND length_in_train_range AND family_profile_hit
```

由于正式初始实验中三种方法的合法字符率和长度范围率均为 1.00，所以初始实验中的忠实性差异主要由 PF00741 profile 命中率决定。

---

## 4. 原始模型的正式比较

每种模型生成 100 条序列，使用相同的 `dataset_v1` 训练长度范围和相同的 PF00741/HMMER 判定标准。

### 4.1 主要结果

| 生成策略 | 典型运行 | 样本数 | PF00741 通过率 | 通过数量 | 长度范围率 | 合法字符率 |
|---|---|---:|---:|---:|---:|---:|
| 自回归 LSTM | `lstm_temp0.8_seed42` | 100 | **1.00** | 100/100 | 1.00 | 1.00 |
| VAE | `vae_scale1_temp0.8_seed42` | 100 | **0.95** | 95/100 | 1.00 | 1.00 |
| GAN | `gan_100ep_temp0.1_seed42` | 100 | **0.85** | 85/100 | 1.00 | 1.00 |

原始模型的排序为：

```text
LSTM > VAE > GAN
```

按通过率计算：

- LSTM 比 VAE 高 5 个百分点；
- LSTM 比 GAN 高 15 个百分点；
- VAE 比 GAN 高 10 个百分点。

### 4.2 与新颖性和多样性的联合解读

Pfam 通过率不能脱离新颖性和多样性单独解读。原始模型的相关结果如下：

| 模型 | PF00741 通过率 | 新颖序列率 | 最近训练序列平均 identity | 平均 pairwise diversity | AA composition L1 |
|---|---:|---:|---:|---:|---:|
| LSTM | 1.00 | 0.01 | 0.988 | 0.054 | 0.031 |
| VAE | 0.95 | 0.54 | 0.907 | 0.234 | 0.121 |
| GAN | 0.85 | 1.00 | 0.773 | 0.331 | 0.173 |

这些结果体现出明显的保真度—新颖性折中：

- **LSTM**：PF00741 通过率最高，但生成结果非常接近训练序列，初始版本存在较明显的记忆倾向；
- **VAE**：在保持较高 profile 通过率的同时产生更多新颖序列，但有一部分序列偏离家族 profile；
- **GAN**：新颖性和多样性最高，但未经额外约束时更容易偏离 PF00741 家族分布。

因此，原始结果不能简单概括为“LSTM 全面最好”。更准确的说法是：LSTM 在原始设置下最擅长保持家族约束，但 GAN 在产生新颖变体方面更强。

---

## 5. 结果的机制解释

### 5.1 自回归 LSTM

LSTM 逐位点生成序列。每个新残基的预测依赖此前已经生成的残基，因此模型较容易学习：

- GvpA 序列中的局部保守 motif；
- 残基的顺序依赖关系；
- 常见长度和终止模式；
- 训练集中反复出现的家族序列模式。

这解释了其 1.00 的 PF00741 通过率。但同一机制也会导致模型生成结果靠近训练序列，表现为低新颖率、低 pairwise diversity 和较高 nearest-train identity。

### 5.2 VAE

VAE 将序列映射到潜在空间，再从潜变量采样生成序列。潜变量能够提高生成变化程度，但小样本训练也会带来：

- 潜变量扰动过大时偏离家族分布；
- EOS 学习不稳定；
- 序列长度出现异常；
- 不同训练或采样 seed 之间结果波动较大。

因此，VAE 的 PF00741 通过率低于 LSTM，但新颖性和多样性明显高于初始 LSTM。

### 5.3 GAN

GAN 通过生成器和判别器的对抗训练学习序列分布。小规模蛋白序列数据对 GAN 较困难，模型容易在提高新颖性的同时产生：

- 氨基酸组成漂移；
- 局部保守位点损失；
- 疏水残基连续堆积；
- 不符合 PF00741 profile 的序列。

这解释了原始 GAN 的 PF00741 通过率仅为 0.85，而新颖序列率达到 1.00。

---

## 6. 优化后的模型比较

后续实验对三种模型分别加入了针对性改进：

- LSTM：辅助 GvpA 数据预训练、蓝藻数据微调、dropout、weight decay 和采样温度调整；
- VAE：经验长度生成、目标长度下强制结束、latent scale 和 temperature 调整；
- GAN：位置组成约束、疏水连续片段惩罚和输出筛选。

### 6.1 多采样 seed 的优化结果

| 模型 | 优化设置概述 | PF00741 平均通过率 | 通过率范围 | 新颖率均值 | 主要特点 |
|---|---|---:|---:|---:|---|
| LSTM | 辅助预训练 + 蓝藻微调 + 正则化 | 0.990 | 0.980--1.000 | 0.320 | 高保真、稳定，仍保留一定相似性 |
| VAE | 经验长度控制 + latent scale 0.75 + temperature 0.6 | 0.967 | 0.960--0.970 | 0.330 | 保真度和多样性较平衡，但稳定性较弱 |
| GAN | composition regularization + hydrophobic-run penalty | **1.000** | 1.000--1.000 | **0.997** | 高保真、高新颖性，但依赖显式约束 |

优化后表面上的排序为：

```text
约束 GAN >= 优化 LSTM > 优化 VAE
```

但这个排序必须注明实验条件。优化 GAN 的 1.00 不是未经处理的原始 GAN 自然得到的，而是加入了专门的组成和疏水性约束后得到的结果。因此它说明：

> GAN 可以通过显式约束显著提高 Pfam 通过率，但不能说明原始 GAN 本身比 LSTM 更容易保持家族结构。

### 6.2 优化前后的 GAN 对比

| GAN 版本 | PF00741 通过率 | 新颖率 | 主要差异 |
|---|---:|---:|---|
| 原始 GAN | 0.85 | 1.00 | 新颖性高，但 profile 偏离较多 |
| 约束 GAN | 1.00 | 0.997 | 加入组成和疏水连续片段约束后显著改善 |

GAN 从 0.85 提高到 1.00，说明 profile 通过率对生成约束和输出筛选非常敏感。

---

## 7. 独立训练 seed 的稳定性验证

为避免只依据一次训练结果，项目对选定配置进行了 training seed=7 的独立重训，生成时使用 generation seed=42。

| 模型 | 初始/选定配置结果 | 独立重训结果 | 稳定性评价 |
|---|---:|---:|---|
| 优化 LSTM | PF00741 约 0.98--1.00 | **1.00** | 最稳定的高保真模型 |
| 优化 VAE | PF00741 约 0.97 | 0.83 | 对训练初始化和潜变量设置敏感 |
| 约束 GAN | 1.00 | **1.00** | 通过率高，但依赖显式约束和筛选 |

独立重训改变了对 VAE 的判断：第一次结果约为 0.967，第二次下降到 0.83，说明 VAE 的高通过率不够稳定，不能只依据第一次运行做强结论。

LSTM 在独立重训后仍达到 1.00，说明其高 profile 保真度具有更好的可重复性。GAN 也保持 1.00，但其结果应始终与组成约束、疏水性约束和最终输出过滤一起报告。

---

## 8. RQ1 的直接回答

### 8.1 原始生成策略的回答

在相同的 100 条样本和相同 PF00741/HMMER 判定标准下：

```text
自回归 LSTM：1.00
VAE：0.95
GAN：0.85
```

因此，在不加入额外后处理约束的原始模型比较中，**自回归 LSTM 最善于保持 Pfam 家族约束，其次是 VAE，GAN 最弱**。

### 8.2 经过优化后的回答

经过专门的生成约束后：

```text
约束 GAN：1.00
优化 LSTM：约 0.99
优化 VAE：约 0.967，且重训时下降到 0.83
```

因此，如果允许使用模型内约束和输出筛选，**约束 GAN 可以获得最高的 Pfam 通过率和最高的新颖性**；但从跨训练运行的可靠性和不依赖大量后处理的角度，**优化 LSTM 是更稳定的高保真策略**。

### 8.3 最终综合判断

本项目支持的最稳妥结论是：

> 原始架构比较显示，自回归 LSTM 最能保持 PF00741 家族约束，VAE 居中，GAN 的原始约束保持能力较弱。引入组成、长度和疏水片段等显式约束后，GAN 的 PF00741 通过率可提升至 1.00，并同时保持很高的新颖性；然而，优化 LSTM 在独立训练 seed 下表现更加稳定，因此应将其视为最可靠的高保真生成器。VAE 能提供较好的潜空间多样性和长度控制，但其 profile 保真度受训练初始化和采样设置影响较大。

---

## 9. 适合论文正文的简洁表述

> We evaluated whether sequences generated by autoregressive, VAE, and GAN strategies retained the GvpA family constraint defined by the PF00741 HMM profile. In the initial 100-sequence comparison, the PF00741 pass rates were 1.00 for the autoregressive LSTM, 0.95 for the VAE, and 0.85 for the GAN. The LSTM therefore showed the strongest unconstrained family fidelity, although it also exhibited the lowest novelty and pairwise diversity. After introducing empirical-length control, amino-acid composition regularization, and hydrophobic-run constraints, the optimized GAN reached a PF00741 pass rate of 1.00, while optimized LSTM and VAE reached approximately 0.99 and 0.967, respectively. However, independent retraining showed that the optimized LSTM remained highly stable, whereas the VAE pass rate decreased to 0.83. Overall, the autoregressive LSTM was the most reliable high-fidelity generator, while the constrained GAN provided the best combination of family-profile retention and novelty under explicit output constraints. PF00741 hits were interpreted as family-level sequence evidence rather than proof of correct folding, assembly, or biological function.

对应中文表述：

> 本研究使用 PF00741 HMM profile 评价不同生成策略对 GvpA 家族序列约束的保持能力。在初始的 100 条序列比较中，自回归 LSTM、VAE 和 GAN 的 PF00741 通过率分别为 1.00、0.95 和 0.85，说明未经额外约束时 LSTM 的家族保真度最高。LSTM 同时具有最低的新颖性和成对多样性，表明其高保真度伴随较强的训练分布相似性。加入经验长度控制、氨基酸组成约束和疏水连续片段约束后，优化 GAN 的 PF00741 通过率提高到 1.00，优化 LSTM 和 VAE 分别约为 0.99 和 0.967。然而，独立重训显示优化 LSTM 的结果仍保持稳定，而 VAE 的通过率下降至 0.83。因此，综合家族约束保持能力、独立重训稳定性和生成条件，优化 LSTM 是最可靠的高保真生成策略；约束 GAN 在允许显式后处理时能够同时获得最高的 profile 通过率和新颖性。PF00741 命中应被解释为家族层面的序列证据，不能单独证明蛋白的正确折叠、组装或生物学功能。

---

## 10. 数据来源与可复现文件

### 正式初始比较

- `runs/formal_model_comparison.csv`
- `runs/formal_lstm_50ep_seed42/eval_profile_v2/metrics.json`
- `runs/formal_lstm_50ep_seed42/eval_profile_v2/per_sequence_metrics.csv`
- `runs/formal_lstm_50ep_seed42/eval_profile_v2/hmmsearch.tblout`
- `runs/formal_vae_50ep_seed42/eval_profile_v2/metrics.json`
- `runs/formal_vae_50ep_seed42/eval_profile_v2/per_sequence_metrics.csv`
- `runs/formal_vae_50ep_seed42/eval_profile_v2/hmmsearch.tblout`

### 优化模型比较

- `runs/optimized_model_comparison.csv`
- `runs/optimization_summary.md`
- `runs/optimization_retest.md`
- `runs/opt_lstm_pretrain_finetune_seed42/`
- `runs/formal_vae_50ep_seed42/`
- `runs/opt_gan_composition_hydro_seed42/`
- `runs/retest_*_seed7/`

### 数据和 profile

- `data/processed/dataset_v1/train.fasta`
- `data/processed/dataset_v1/valid.fasta`
- `data/processed/dataset_v1/test.fasta`
- `data/processed/dataset_v1/profiles/PF00741.hmm`
- `data/processed/dataset_v1/dataset_report.md`

### 重要分支说明

当前 `diversity` 分支中的部分通用评价表将 `family_profile_hit_rate` 标记为 `null/not_run`，因为这些 diversity 批次没有接入 HMMER profile 扫描。本文的 PF00741 数值来自正式的 `formal_*` profile evaluation 和优化实验产物，不能用 diversity 分支中 `not_run` 的空字段替代。

---

## 11. 研究限制

1. 初始模型的主要正式比较每种方法只有一次训练 seed，样本量为每次 100 条；应继续增加独立训练 seed。
2. PF00741 是广义 gas-vesicle family profile，不是 GvpA 功能的充分证明。
3. Pfam 命中率属于序列/profile 层面的约束指标，不能代替结构预测、分子模拟或湿实验验证。
4. 优化 GAN 的高通过率依赖显式约束和输出筛选，必须与原始 GAN 结果分开报告。
5. VAE 的经验长度控制是生成时施加的约束，不能被解释为模型独立学习到了真实长度分布。
6. LSTM 的高通过率伴随较高的训练集相似度，必须同时报告新颖性、最近邻 identity 和多样性。
7. 通过率是比例指标，当前实验不应被写成对所有 GvpA 序列或所有数据集的普遍结论。

## 12. 一句话结论

**原始模型中 LSTM 最善于保持 Pfam 家族约束；经过显式约束后 GAN 可达到最高通过率和新颖性，但综合独立重训稳定性，优化 LSTM 仍是最可靠的高保真生成策略，VAE 则表现出较强的多样性优势和明显的训练敏感性。**
