# GV02-01 蓝藻 GvpA 数据集详细说明

> 项目：GV02-01 蓝藻 GvpA 序列生成  
> 主数据版本：`dataset_v1`  
> 文档用途：说明数据来源、清洗流程、数据规模、质量控制、划分方式、辅助预训练集及建模注意事项。

## 1. 数据集总览

本项目的数据体系分为两部分：

1. **主数据集：蓝藻 GvpA `dataset_v1`**
   - 用于正式训练、验证和最终测试；
   - 目标对象是 Cyanobacteria 中的 GvpA 主壳体蛋白；
   - 所有模型对比实验都应回到该数据集的固定划分。

2. **辅助预训练集：泛 GvpA `aux_pan_gvpa_v1`**
   - 包含非蓝藻来源的 GvpA-like 序列；
   - 只用于 LSTM/VAE 的预训练或迁移学习实验；
   - 不能替代蓝藻 valid/test，也不能改变主任务的目标分布。

目录结构：

```text
gvpa-sequence-generation/data/processed/
├── dataset_v1/
└── aux_pan_gvpa_v1/
```

## 2. 主数据集 dataset_v1

主数据集位于：

```text
gvpa-sequence-generation/data/processed/dataset_v1/
```

冻结报告见：[dataset_report.md](./gvpa-sequence-generation/data/processed/dataset_v1/dataset_report.md)

### 2.1 基本规模

| 项目 | 数值 |
| --- | ---: |
| 最终序列数 | 165 |
| train | 139 |
| valid | 13 |
| test | 13 |
| 长度范围 | 59--91 aa |
| 平均长度 | 73.22 aa |
| 中位长度 | 72 aa |
| exact duplicate | 0 |
| 标准氨基酸字符 | 165/165 通过 |
| 蓝藻 taxonomy | 165/165 通过 |
| PF00741 命中 | 165/165 |
| 本地 GvpA core profile 命中 | 165/165 |
| MSA 异常序列 | 0 |
| 跨集合 >=95% identity 泄漏对 | 0 |
| 跨集合最高 identity | 94.40% |

该数据集属于**小规模、高保守、强约束蛋白序列数据集**。165 条序列不能简单理解为 165 个完全独立的样本，因为序列之间存在较强的同源性和聚类结构。

### 2.2 文件清单

| 文件 | 用途 |
| --- | --- |
| `all.fasta` | 全部 165 条冻结序列 |
| `train.fasta` | 训练集，139 条 |
| `valid.fasta` | 超参数选择和模型调试，13 条 |
| `test.fasta` | 最终评价，13 条 |
| `metadata.csv` | 序列、来源、注释、profile、cluster 和 split 的完整审计表 |
| `all.mafft.fasta` | 165 条序列的 MAFFT 多序列比对 |
| `manifest.json` | 数据版本、冻结时间、工具版本和验证摘要 |
| `checksums.sha256` | 冻结文件完整性校验 |
| `profiles/PF00741.hmm` | 官方 Pfam profile |
| `profiles/GvpA_core95.hmm` | 本地高可信 GvpA core profile |
| `audit/` | 清洗、覆盖、profile、MSA、聚类和划分审计文件 |

## 3. 数据来源和构建过程

主数据集不是直接使用课程 FASTA，而是经过课程数据清洗、NCBI 补充、profile 复核、去重、MSA 检查和防泄漏划分形成的。

总体数据流：

```text
课程原始 GvpA.fasta
856 条
    ↓ 严格清洗
144 条
    ↓ NCBI 双路检索
1141 个 accession
    ↓ accession 去重和候选整理
241 条非重复候选
    ↓ taxonomy / GvpA / CDD / 完整性 / 字符 / 长度检查
157 条
    ↓ profile 复核
救回 9 个 accession，其中 1 条为已有序列重复
    ↓
最终冻结 dataset_v1：165 条
```

主数据来源分布：

| `dataset_source` | 数量 |
| --- | ---: |
| `course_strict_v1` | 144 |
| `ncbi_discovery_20260907` | 13 |
| `ncbi_discovery_profile_rescue` | 8 |

覆盖审计显示：NCBI 两路检索合并得到 1141 个 accession，去完全重复后得到 241 条候选；其中 156 条通过严格检查，课程数据已覆盖其中 143 条，新增确认序列 13 条。该覆盖率只是截至检索日期、针对可自动检索和确认记录的覆盖，不代表自然界中的 GvpA 序列已经全部收集。

## 4. 严格清洗规则

序列进入主数据集前，需要同时满足以下条件：

1. NCBI 当前 lineage 属于 `Cyanobacteria` 或 `Cyanobacteriota`；
2. NCBI definition、product 或 gene 明确支持 GvpA；
3. NCBI CDD 命中 `PRK09371 / CDD:181805`；
4. 明确标记为 full length，或者 protein feature 覆盖全长且没有 partial 标记；
5. 不能是 partial、fragment 或 incomplete；
6. 只含 20 种标准氨基酸；
7. 本地序列与当前 NCBI 序列一致；
8. 不与已纳入记录完全重复；
9. 长度处于设定的合理范围内。

课程原始数据 856 条中，严格纳入 144 条，排除 712 条。主要排除原因如下：

| 排除原因 | 数量 |
| --- | ---: |
| `not_cyanobacteria_by_ncbi_lineage` | 695 |
| `gvpa_cdd_prk09371_not_confirmed` | 38 |
| `partial_fragment_or_incomplete` | 19 |
| `annotated_as_other_gvpA_like_protein` | 17 |
| `ambiguous_or_non_gvpa_annotation` | 2 |

同一条记录可能同时触发多个原因，因此上表数量不能直接相加得到 712。

## 5. GvpA 身份验证和 profile 说明

### 5.1 PF00741 的使用

当前数据流程采用：

```text
PF00741 / Gas_vesicle
```

所有 165 条冻结序列均通过官方 PF00741 gathering threshold。

但是，PF00741 是较宽泛的 gas vesicle protein family，不能单独证明一条序列就是 GvpA。正式身份判断还结合：

- GvpA-specific annotation；
- NCBI CDD `PRK09371 / CDD:181805`；
- Cyanobacteria taxonomy；
- 本地高可信 GvpA core profile；
- 长度和序列特征；
- 必要时的人工核验。

课程材料中曾出现 `PF01132`，但项目审计指出它与当前 GvpA 注释不一致，因此不能未经确认地把 `PF01132` 写死为 GvpA validator。

### 5.2 profile 复核

30 条待复核候选经过官方 PF00741 和本地 GvpA core profile 检查：

| 项目 | 数量 |
| --- | ---: |
| 待复核 | 30 |
| profile 救回 | 9 |
| 最终排除 | 21 |
| 严格核心集最低 local profile bit score | 124.8 |
| 净新增 | 8 |

救回条件包括：

- 通过官方 PF00741 gathering threshold；
- 长度在 50--100 aa；
- 本地 core profile 得分不低于严格核心集最低值 124.8；
- 只含标准氨基酸；
- 有明确 GvpA 名称或 PRK09371/CDD 支持。

## 6. 序列长度分布

FASTA 实际统计结果：

| split | 数量 | 最短 | 最长 | 平均 | 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 165 | 59 | 91 | 73.22 | 72 |
| train | 139 | 59 | 91 | 72.27 | 72 |
| valid | 13 | 72 | 90 | 75.69 | 72 |
| test | 13 | 72 | 91 | 80.92 | 78 |

全体数据中最常见的长度：

| 长度 | 数量 |
| ---: | ---: |
| 72 | 84 |
| 71 | 34 |
| 73 | 12 |
| 75 | 10 |
| 90 | 5 |
| 80 | 5 |
| 74 | 4 |
| 89 | 2 |
| 91 | 2 |

可以看出，数据绝大多数集中在 71--72 aa 附近。对生成模型而言，长度分布是一个重要约束：

- 生成大量过短序列可能意味着提前终止或解码失败；
- 生成大量过长序列可能意味着 EOS 学习不足；
- 只生成 72 aa 也可能意味着模型过度记忆训练集分布。

## 7. 氨基酸组成

全体 165 条序列的氨基酸总组成如下：

| 氨基酸 | 数量 | 比例 |
| --- | ---: | ---: |
| V | 2040 | 16.88% |
| A | 1896 | 15.69% |
| L | 1183 | 9.79% |
| S | 1165 | 9.64% |
| I | 1085 | 8.98% |
| E | 1004 | 8.31% |
| G | 521 | 4.31% |
| R | 515 | 4.26% |
| D | 514 | 4.25% |
| K | 503 | 4.16% |
| T | 449 | 3.72% |
| Y | 335 | 2.77% |
| M | 189 | 1.56% |
| P | 180 | 1.49% |
| N | 171 | 1.42% |
| W | 165 | 1.37% |
| Q | 150 | 1.24% |
| F | 13 | 0.11% |
| H | 3 | 0.02% |
| C | 1 | 0.01% |

数据表现出明显的组成偏好：V、A、L、I 等疏水或小体积残基比例较高，而 C、H、F 等残基非常少。生成结果不应只检查“是否使用了合法氨基酸”，还应检查其组成是否严重偏离真实 GvpA 分布。

## 8. Taxonomy 和来源覆盖

metadata 中共有 107 个不同的 `ncbi_organism` 字段值。该字段有时是具体物种，有时是属、科、目或未分类名称，因此不能简单等价为 107 个完整物种。

出现较多的 organism 字段包括：

| organism | 数量 |
| --- | ---: |
| Cyanophyceae | 15 |
| Nostoc sp. | 8 |
| Nostocales | 7 |
| Nodularia spumigena | 7 |
| Anabaena azotica | 5 |
| Microcystis aeruginosa | 3 |
| Microcystis | 3 |
| Trichodesmium erythraeum IMS101 | 2 |
| Kamptonema formosum | 2 |
| Planktothrix | 2 |

主要覆盖蓝藻类群包括：

- Nostocales；
- Oscillatoriales / Oscillatoriophycideae；
- Leptolyngbyales；
- Pseudanabaenales；
- Oculatellales；
- Synechococcales；
- Chroococcales；
- Gloeobacterales 等。

这说明数据不是来自单一蓝藻物种，而是覆盖多个谱系；但不同谱系的样本量并不均衡。

## 9. Provisional 来源记录

最终数据集中保留 3 条 provisional 来源记录，并在 `metadata.csv` 中持续标记：

| accession | 标记 | organism | split | 长度 |
| --- | --- | --- | --- | ---: |
| `GAB4214070.1` | MAG | Synechococcales cyanobacterium | valid | 72 |
| `GAB4386638.1` | MAG | Elainellaceae cyanobacterium | train | 72 |
| `WP_298916699.1` | uncultured | uncultured Nostoc sp. | test | 75 |

这些记录没有被删除，但由于 MAG 和 uncultured 来源可能存在注释或组装不确定性，正式分析建议补做一次敏感性实验：

```text
完整 dataset_v1
vs.
移除 3 条 provisional 记录后的数据集
```

然后比较模型指标和主要结论是否变化。

## 10. 相似度聚类和数据冗余

聚类使用 MMseqs2，参数为：

```text
--min-seq-id 0.95/0.90
-c 0.8
--cov-mode 0
```

聚类结果：

| identity 阈值 | 簇数 | 最大簇 | 最大簇比例 |
| ---: | ---: | ---: | ---: |
| 95% | 35 | 38 条 | 23.0% |
| 90% | 7 | 74 条 | 44.8% |

这说明：

- 165 条序列中存在明显冗余；
- 90% identity 下，大量序列属于少数相似性连通组；
- 数据集名义规模大于真正的独立序列模式数量；
- 模型很容易记忆高频序列模式。

因此，生成模型评价必须包括：

- exact copy rate；
- 与训练集最近邻的 identity；
- 最近邻的 query/target coverage；
- 候选集 unique ratio；
- 候选内部 pairwise diversity；
- 最大重复簇比例。

## 11. train/valid/test 划分

划分报告见：[split_report.md](./gvpa-sequence-generation/data/processed/dataset_v1/audit/split_report.md)

最终划分为：

| split | 序列数 | 95% identity / 80% coverage 连通组数 |
| --- | ---: | ---: |
| train | 139 | 1 |
| valid | 13 | 4 |
| test | 13 | 10 |

该划分没有强行追求标准的 70/15/15 比例，原因是最大相似性连通组本身就有 139 条。如果将这个大连通组拆散到 train、valid 和 test，就会让近重复序列跨集合出现，造成数据泄漏。

因此当前划分优先保证：

```text
跨集合 identity >= 95%
且双方 coverage >= 80%
的序列对为 0
```

数据使用规范：

- `train.fasta`：训练模型和估计训练分布；
- `valid.fasta`：选择超参数、temperature、VAE beta 等；
- `test.fasta`：只用于最终结果；
- 不得重新随机划分；
- 不得直接修改 `dataset_v1`；
- 如果数据发生增删，应创建 `dataset_v2`。

## 12. MSA 多序列比对

MSA 使用 MAFFT 生成，结果文件为：

```text
data/processed/dataset_v1/all.mafft.fasta
```

质量统计：

| 项目 | 数值 |
| --- | ---: |
| 序列数 | 165 |
| alignment 长度 | 115 列 |
| 覆盖至少 90% 序列的核心列 | 71 列 |
| identity to consensus 最低值 | 80.8% |
| identity to consensus 中位数 | 93.1% |
| identity to consensus 最高值 | 97.2% |
| 明显 MSA 异常序列 | 0 |

MSA 可用于：

- 识别保守位点；
- 计算 position-wise entropy；
- 计算 occupancy 和 gap rate；
- 识别生成序列是否破坏核心模式；
- 构造保守位点或软约束评分；
- 为结构和物理化学分析提供位置对应关系。

MSA 中的保守性不应只理解为每一列必须是同一个氨基酸，也应考虑氨基酸性质类别，例如疏水、带电、极性和体积。

## 13. metadata.csv 字段

`metadata.csv` 是数据集的核心审计表，不只是序列 ID 列表。主要字段如下：

| 字段类别 | 代表字段 | 含义 |
| --- | --- | --- |
| 输入信息 | `input_order`, `sequence_id`, `full_header` | 原始顺序、序列 ID 和完整 header |
| 序列信息 | `sequence`, `length`, `valid_amino_acids` | 序列、长度和字符合法性 |
| 清洗决策 | `decision`, `reasons` | 是否纳入及纳入/排除原因 |
| 重复信息 | `duplicate_id_of`, `duplicate_sequence_of` | ID 或序列重复关系 |
| NCBI 信息 | `ncbi_accession`, `ncbi_definition`, `ncbi_organism` | 数据库 accession、定义和 organism |
| 分类信息 | `ncbi_taxonomy`, `ncbi_taxid` | lineage 和 taxonomic ID |
| 注释证据 | `ncbi_products`, `ncbi_genes`, `ncbi_region_names`, `ncbi_cdd_xrefs` | product、gene、CDD 区域及交叉引用 |
| 完整性 | `ncbi_protein_partial5`, `ncbi_protein_partial3`, `ncbi_complete_by_feature` | partial 和 full-length 检查 |
| 身份判断 | `is_cyanobacteria`, `explicit_gvpa_annotation`, `ncbi_sequence_matches_local` | taxonomy、GvpA 注释和序列一致性 |
| 数据来源 | `dataset_source`, `provisional_source_flag`, `admission_route` | 来源、 provisional 标记和纳入路径 |
| profile | `pf00741_ga_pass`, `pf00741_bitscore`, `core_profile_bitscore` | 官方和本地 profile 结果 |
| 聚类划分 | `cluster95_component`, `split` | 95% 相似性组件及 train/valid/test |
| 版本控制 | `dataset_version`, `frozen` | 数据版本和冻结状态 |

## 14. 辅助预训练集 aux_pan_gvpa_v1

辅助集位于：

```text
gvpa-sequence-generation/data/processed/aux_pan_gvpa_v1/
```

报告见：[aux_pan_gvpa_v1 dataset_report.md](./gvpa-sequence-generation/data/processed/aux_pan_gvpa_v1/dataset_report.md)

### 14.1 数据规模

| 项目 | 数量 |
| --- | ---: |
| 原始高可信非蓝藻 GvpA 候选 | 655 |
| 因接近蓝藻 valid/test 被排除 | 105 |
| 安全辅助池 | 550 |
| 95% 去冗余代表 | 302 |
| 90% 去冗余代表 | 168 |
| 默认预训练组合 | 441 = 302 + 139 |

分类域分布：

| domain | 数量 |
| --- | ---: |
| Bacteria | 343 |
| Archaea | 207 |

全部 550 条安全辅助序列通过官方 PF00741 gathering threshold，并且没有与蓝藻冻结 valid/test 达到 80% identity 且双方 coverage 不低于 80% 的序列。

### 14.2 文件用途

| 文件 | 用途 |
| --- | --- |
| `all_safe.fasta` | 550 条安全辅助序列 |
| `pretrain_95rep.fasta` | 302 条 95% 去冗余代表，默认预训练集 |
| `pretrain_90rep.fasta` | 168 条 90% 去冗余代表，强去冗余备选 |
| `pretrain_plus_blue_train.fasta` | 302 条辅助代表 + 139 条蓝藻 train，共 441 条 |
| `blue_finetune_train.fasta` | 蓝藻 train，139 条 |
| `metadata.csv` | 辅助集来源和聚类 metadata |
| `training_recipe.md` | 预训练和微调使用规则 |
| `checksums.sha256` | 文件完整性校验 |

### 14.3 推荐训练方式

实验 A：蓝藻从头训练

```text
dataset_v1/train.fasta
        ↓
从头训练 LSTM/VAE
        ↓
固定 dataset_v1/valid.fasta 调参
        ↓
固定 dataset_v1/test.fasta 最终评价
```

实验 B：泛 GvpA 预训练后蓝藻微调

```text
pretrain_plus_blue_train.fasta
        ↓
泛 GvpA 预训练
        ↓
blue_finetune_train.fasta 微调
        ↓
固定 dataset_v1/valid.fasta 和 test.fasta 评价
```

两组实验必须同时报告，才能判断辅助预训练是否真正提升了蓝藻 GvpA 生成质量。

## 15. 对模型训练和评价的影响

### 15.1 小样本风险

训练集只有 139 条序列，而 LSTM/VAE 等深度模型参数量可能远大于数据中真正独立的模式数量。因此需要关注：

- train loss 与 valid loss 的差异；
- exact copy rate；
- nearest-train identity；
- 多个随机种子下的稳定性；
- 生成长度是否异常；
- 候选集是否模式塌缩。

### 15.2 不能只看 sequence identity

sequence identity 适合检测复制、数据泄漏和新颖性，但不能直接代表功能相似度。生成序列需要联合检查：

- family/profile 通过率；
- 长度范围；
- MSA 保守位点；
- 氨基酸组成；
- 疏水性、电荷和极性分布；
- 结构代理指标；
- ESM-2 embedding 分布；
- 候选集内部多样性。

### 15.3 不能只追求 novelty

与已知序列差异越大，不代表序列越好。过度追求 novelty 可能破坏：

- GvpA family 身份；
- β-sheet 壳体相关结构；
- 保守位点和局部 motif；
- 与其他 Gvp 蛋白的组装界面。

更合理的目标是：在 profile、长度和结构代理约束通过的前提下，提高新颖性和候选集多样性。

## 16. 如何评价数据集质量

需要区分两个问题：

1. **数据集本身好不好**；
2. **模型生成的序列好不好**。

两者使用的指标不同。不能因为模型生成效果差，就直接认为数据集质量差；也不能因为生成了很多新序列，就认为模型一定学到了 GvpA 的结构和功能规律。

### 16.1 数据合法性和完整性

这是数据集的基础质量检查：

| 指标 | 含义 | `dataset_v1` 当前情况 |
| --- | --- | ---: |
| 有效字符率 | 是否只含 20 种标准氨基酸 | 165/165 |
| 空序列率 | 是否存在空序列 | 0 |
| 完全重复率 | 是否存在 exact duplicate | 0 |
| partial/fragment 比例 | 是否存在截断或不完整序列 | 当前冻结集均通过 |
| 本地/数据库一致率 | 本地序列是否与 NCBI 当前序列一致 | 165/165 |
| 长度异常率 | 是否存在明显异常长度 | 当前长度为 59--91 aa |

这些属于硬门槛。若生成序列出现非法字符、空序列、大量截断或明显异常长度，应直接判为不合格，而不是用其他指标弥补。

### 16.2 生物学身份可靠性

对 GV02-01，核心问题是数据是否真的属于“蓝藻 GvpA”，而不是其他 GvpA-like 蛋白或其他 Gvp 亚型。

建议报告：

- Cyanobacteria/Cyanobacteriota taxonomy 通过率；
- GvpA definition/product/gene 注释支持率；
- `PRK09371 / CDD:181805` 证据比例；
- 官方 `PF00741` 通过率；
- 本地 GvpA core profile 通过率；
- profile score 和 coverage 分布；
- provisional 记录比例。

当前冻结数据中：

- 蓝藻 taxonomy：165/165；
- PF00741：165/165；
- 本地 GvpA core profile：165/165；
- 163 条具有明确 GvpA 注释，2 条通过组合证据保留；
- 159 条具有 `gvpA` gene 或相关数据库支持；
- 3 条记录带有 MAG/uncultured provisional 标记。

需要强调，PF00741 是广义 Gas Vesicle family，不能单独证明序列一定是 GvpA，必须和 taxonomy、CDD、注释及本地 profile 联合解释。

### 16.3 去重、冗余和有效独立规模

数据量不是越大越好，还要看其中有多少独立序列模式。建议报告：

- exact duplicate rate；
- 95% 和 90% identity 下的 cluster 数量；
- 最大 cluster 大小和最大簇占比；
- cluster-aware split 结果。

当前结果为：

| identity 阈值 | 簇数 | 最大簇 | 最大簇比例 |
| ---: | ---: | ---: | ---: |
| 95% | 35 | 38 条 | 23.0% |
| 90% | 7 | 74 条 | 44.8% |

因此，165 条序列不能简单理解为 165 个完全独立样本。95% identity 下约 35 个簇、90% identity 下约 7 个簇，可以作为真实数据复杂度的近似参考。

### 16.4 数据泄漏

应检查 train-valid、train-test、valid-test 之间是否存在高相似序列对。当前采用：

```text
identity >= 95%
且双方 coverage >= 80%
```

`dataset_v1` 当前结果：

- 跨集合达到 95% identity 且 coverage 不低于 80% 的序列对：0；
- 跨集合最高 identity：94.40%。

这是数据集质量的重要优点。它说明模型测试结果不太可能仅由 95% 以上近重复序列泄漏造成。

### 16.5 长度、组成和 MSA 分布

还应检查：

- all/train/valid/test 的长度分布是否合理；
- 训练集与测试集是否存在明显长度偏移；
- 氨基酸组成是否符合真实 GvpA 特征；
- MSA 中是否存在异常序列；
- 核心列 occupancy、entropy 和 gap rate 是否合理。

当前数据的长度主要集中在 71--72 aa，test 平均长度为 80.92 aa，高于 train 的 72.27 aa。因此模型评价最好增加按长度分层的结果，不能只报告一个总体平均值。

## 17. 如何评价生成序列质量

生成序列建议按四层进行评价，而不是依赖一个总分。

### 17.1 第一层：硬约束通过率

包括：

- 合法字符率；
- 长度范围率；
- PF00741 通过率；
- 本地 GvpA profile 通过率；
- profile coverage 和 score；
- exact copy rate。

当前 evaluator 已实现合法字符、长度、exact copy 等指标，但 family/profile 字段仍为 `not_run`，因此当前结果还不能正式判断生成序列是否属于 GvpA-like family。

### 17.2 第二层：新颖性

建议报告：

- `exact_copy_train_rate`；
- `exact_copy_any_real_rate`；
- 最近邻训练序列 identity 的均值、最大值和分位数；
- 多个阈值下的 novelty rate，例如 identity < 0.90、0.95、0.98；
- nearest-train query coverage；
- nearest-train target coverage。

identity 越低通常表示越新，但不代表功能越好。95% 是项目操作阈值，不是功能相同或不同的生物学分界线。

### 17.3 第三层：候选集多样性

建议报告：

- `unique_ratio`；
- `mean_pairwise_diversity`；
- `max_duplicate_fraction`；
- 生成候选的相似性 cluster 分布。

如果 unique ratio 很低，或某一条序列占据很大比例，说明模型可能出现模式塌缩。反过来，pairwise diversity 过高也可能表示生成了大量随机或分布外序列，不能单独视为优点。

### 17.4 第四层：组成、保守模式和表示空间

建议比较：

- `aa_composition_l1_distance`；
- MSA 保守位点一致性；
- 疏水性、电荷和极性分布；
- β-sheet 相关结构代理指标；
- ESM-2 embedding 的 MMD 或 kernel distance；
- embedding 空间离群比例。

这些指标可用于判断生成序列是否保持真实 GvpA 的分布和结构相关模式，但都属于计算代理，不能直接证明真实表达、折叠或气囊组装。

## 18. 当前 evaluator 指标和已有结果

当前实现文件为：[evaluate_all.py](./gvpa-sequence-generation/evaluation/evaluate_all.py)。已经实现的指标包括：

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
- `aa_composition_l1_distance`。

已有 baseline 结果：

| 模型 | 平均最近邻 identity | 平均 pairwise diversity | AA composition L1 |
| --- | ---: | ---: | ---: |
| AA-frequency | 0.5583 | 0.5480 | 0.0294 |
| k-mer | 0.7295 | 0.3792 | 0.0817 |

初步解释：AA-frequency 更偏向新颖性和候选间多样性，但可能没有学到局部 GvpA 模式；k-mer 更接近真实局部模式，但也更接近训练数据，记忆化风险更高。两者谁更好，必须等 profile 通过率和结构代理指标补齐后再判断。

当前 LSTM/VAE 只是 smoke test，生成数量较少，且存在长度控制问题；不能据此做正式模型优劣结论。

## 19. 推荐的模型质量判定方式

建议采用“硬门槛 + 多目标比较”，而不是简单加权得到唯一总分。

### 19.1 合格候选的最低条件

候选至少应满足：

```text
合法字符率高
+ 长度合理率高
+ family/profile 通过率高
+ exact copy rate 低
+ 保守位点破坏率低
+ 组成分布不过度偏离
+ 候选集具有合理多样性
```

### 19.2 三类结果

| 类型 | 典型特征 | 用途 |
| --- | --- | --- |
| 保守型 | profile 高、保守位点保持好、identity 较高 | 可靠性对照 |
| 平衡型 | profile 高、新颖性适中、多样性合理、组成接近真实分布 | 最适合作为主要候选 |
| 探索型 | novelty 和 diversity 高，但部分 profile/embedding 异常 | 结构或实验优先验证 |

高新颖但 profile、组成或 embedding 明显异常的序列，应作为失败案例或离群案例分析，不应直接选为最佳候选。

### 19.3 推荐最终比较表

每个模型至少报告：

| 类别 | 指标 |
| --- | --- |
| 合法性 | `legal_char_rate` |
| 长度 | `length_in_train_range_rate`、平均长度、长度分布距离 |
| 家族约束 | PF00741 通过率、本地 GvpA profile 通过率 |
| 复制风险 | exact copy rate、nearest-train identity |
| 新颖性 | 多个 identity 阈值下的 novelty rate |
| 覆盖 | query coverage、target coverage |
| 多样性 | unique ratio、pairwise diversity、最大重复比例 |
| 组成 | AA composition L1 distance |
| 保守性 | MSA 保守位点一致性、核心列评分 |
| 分布 | ESM-2 MMD、kernel distance、离群比例 |
| 稳定性 | 多随机种子均值、标准差和置信区间 |

## 20. 如何理解“老师已经清洗过的数据”

老师所说的“数据已经清洗过、可以直接使用”，通常意味着数据已经完成了基础质量控制，能够作为课程项目的有效候选数据使用。例如，文件格式、序列可读性、明显错误记录等问题可能已经处理过。

这与“每一条记录都严格属于本项目的蓝藻 GvpA 目标集合”不是完全相同的概念。不同项目使用同一批 GV 数据时，目标集合可能不同：

| 数据用途 | 可以包含的范围 |
| --- | --- |
| GV 相关课程候选数据 | 较广义的 Gvp 及 GvpA-like 蛋白 |
| 泛 GvpA 预训练 | 蓝藻、其他细菌和古菌来源的 GvpA-like 序列 |
| GV02-01 蓝藻 GvpA 主任务 | 需要明确限定为蓝藻来源的目标 GvpA 序列 |

因此，你们的工作不应表述为“老师的数据是错误的，所以我们重新清洗了一遍”，更准确的表述是：

> 在老师提供的、已经具备课程使用条件的候选数据基础上，针对 GV02-01“蓝藻 GvpA 序列生成”的具体目标，进一步进行任务特异性数据核验、目标集合定义和防泄漏划分。

### 16.1 你们实际做的事情

从项目记录看，课程原始 `GvpA.fasta` 有 856 条记录。你们对它们进行了以下方面的任务特异性核验：

1. 是否属于 Cyanobacteria/Cyanobacteriota；
2. definition、product 或 gene 是否支持 GvpA；
3. 是否具有 `PRK09371 / CDD:181805` 证据；
4. 是否为 full length，而不是 partial、fragment 或 incomplete；
5. 是否只含 20 种标准氨基酸；
6. 本地序列是否与 NCBI 当前序列一致；
7. 是否存在完全重复；
8. 是否通过 PF00741 和本地 GvpA core profile；
9. 是否适合进入蓝藻 GvpA 的训练、验证和测试分布。

严格清洗报告中，856 条记录最终保留 144 条。之后又通过 NCBI 补充和 profile 复核，形成 165 条冻结的 `dataset_v1`。

这里的“筛选出蓝藻蛋白”需要稍微准确一点：不是简单地从各种普通蛋白中找出蓝藻蛋白，而是从课程提供的 Gvp 相关候选中，进一步确认哪些记录同时符合：

```text
蓝藻 taxonomy
+ GvpA 相关注释或证据
+ 序列完整性
+ family/profile 支持
+ 长度和字符要求
+ 去重和防泄漏要求
```

所以被排除的记录可能是非蓝藻 GvpA-like 蛋白、其他 Gvp 亚型、注释不明确的同源蛋白、partial/fragment 序列或重复记录，不一定都是完全无关的普通蛋白。

### 16.2 是否属于“过度清洗”

目前不能仅凭“老师说数据已经清洗过”就判断你们的处理过度。关键在于老师对原始 FASTA 的定义：

#### 情况 A：原始文件只是 GV/GvpA 候选集合

如果文件本来包含广义 GvpA-like 蛋白、不同生物来源或其他 Gvp 蛋白，那么你们的严格筛选是必要的，因为 GV02-01 的目标是蓝藻 GvpA，而不是所有 Gvp 相关序列。

#### 情况 B：老师确认 856 条全部是人工确认的蓝藻 GvpA

如果老师明确确认 856 条全部属于蓝藻 GvpA，那么你们就不应直接把被规则排除的记录称为“错误数据”。更稳妥的做法是：

- 把 856 条作为老师提供的原始有效候选集保留；
- 把 165 条称为“严格证据核验后的冻结子集”；
- 说明 165 条采用了比原始课程标签更严格的判定标准；
- 比较 856 条与 165 条对模型训练和生成结果的影响；
- 向老师确认课程数据的原始语义和推荐用法。

建议向老师确认的具体问题是：

> “课程提供的 `GvpA.fasta` 是否保证 856 条全部属于蓝藻 GvpA，还是它只是一个包含 Gvp/GvpA-like 候选的广义数据文件？”

在得到确认前，报告中最好使用“任务特异性核验”和“严格子集”，不要使用“纠正了老师的数据”这样的表述。

### 16.3 为什么仍然需要去重、聚类和防泄漏检查

即使老师已经清洗过数据，以下工作仍然属于机器学习实验设计，而不只是普通数据清洗：

- exact duplicate 检查：避免同一序列重复出现；
- similarity clustering：了解序列冗余和数据独立性；
- cluster-aware split：避免近重复序列跨 train/valid/test 泄漏；
- MSA 检查：确认序列整体结构和保守模式没有明显异常；
- metadata 和 checksum：保证数据来源可追溯、版本可复现。

这些步骤的目的不是怀疑原始数据质量，而是让生成模型比较具有可信度。当前 `dataset_v1` 的 165 条序列在 95% identity/80% coverage 下仍有 35 个簇，在 90% identity 下只有 7 个簇，说明数据天然高度保守，防泄漏尤其重要。

## 21. 当前数据集的局限性

需要在报告中明确说明：

1. 主数据集规模较小，深度模型容易过拟合。
2. 数据高度保守，名义样本数大于真正独立的序列模式数。
3. train/valid/test 数量较少，尤其 valid/test 各只有 13 条，统计不确定性较大。
4. 部分 metadata 的 organism 是高阶分类或未分类名称，不能全部视为精确物种标签。
5. 保留了 2 条 MAG 和 1 条 uncultured provisional 记录，需要做敏感性分析。
6. PF00741 是广义 gas vesicle family，不能单独区分所有 GvpA-like 蛋白。
7. 计算 profile、MSA、embedding 或功能预测只能提供间接证据。
8. 本项目不能仅凭计算指标声称生成序列一定能够表达、正确折叠或组装成真实气囊。

## 22. 推荐的标准使用规范

后续所有模型和实验建议遵守以下规则：

1. 主线模型统一使用冻结的 `dataset_v1`。
2. baseline 只使用 `dataset_v1/train.fasta`。
3. valid 只用于调参，不用于最终结论。
4. test 只在最终实验阶段使用。
5. 不重新随机划分数据。
6. 不直接修改 `dataset_v1`。
7. 数据增删时创建新的 `dataset_v2`。
8. 所有生成模型使用统一 FASTA 格式和统一 evaluator。
9. 所有模型使用相同生成数量、相同评价条件和明确随机种子。
10. 同时报告约束满足度、新颖性、多样性和分布指标。
11. 把高新颖但 profile/组成/embedding 异常的序列作为失败案例分析，而不是直接当作优秀候选。
12. 最终结论使用“计算上具有 GvpA-like 特征”或“值得进一步结构/实验验证”等谨慎表述。

## 23. 相关文件

- [主数据集冻结报告](./gvpa-sequence-generation/data/processed/dataset_v1/dataset_report.md)
- [清洗报告](./gvpa-sequence-generation/data/processed/dataset_v1/audit/cleaning_report.md)
- [覆盖审计](./gvpa-sequence-generation/data/processed/dataset_v1/audit/coverage_audit_report.md)
- [profile 复核报告](./gvpa-sequence-generation/data/processed/dataset_v1/audit/profile_review_report.md)
- [MSA 报告](./gvpa-sequence-generation/data/processed/dataset_v1/audit/msa_report.md)
- [聚类报告](./gvpa-sequence-generation/data/processed/dataset_v1/audit/clustering_report.md)
- [划分报告](./gvpa-sequence-generation/data/processed/dataset_v1/audit/split_report.md)
- [辅助预训练集报告](./gvpa-sequence-generation/data/processed/aux_pan_gvpa_v1/dataset_report.md)
- [辅助集训练说明](./gvpa-sequence-generation/data/processed/aux_pan_gvpa_v1/training_recipe.md)
- [数据清洗与流水线 README](./gvpa-sequence-generation/README.md)
