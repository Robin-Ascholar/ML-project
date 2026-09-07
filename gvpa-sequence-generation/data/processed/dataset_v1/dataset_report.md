# 蓝藻 GvpA dataset_v1 冻结报告

## 冻结结果

- 最终序列：165 条，全部非重复且只含20种标准氨基酸
- 长度：59–91 aa
- train / valid / test：139 / 13 / 13
- 全部通过官方 Pfam PF00741 gathering threshold
- 全部命中由高可信核心序列建立的本地 GvpA profile
- MAFFT异常序列：0 条
- 跨集合达到95% identity且覆盖不低于80%的序列对：0 对
- 跨集合最高identity：94.40%

## 数据流

1. 课程原始数据：856条；严格清洗后144条。
2. NCBI双路检索：1141个 accession、241条非重复候选。
3. 数据库严格补充后：157条。
4. 30条复核候选经官方PF00741和本地核心profile检查，救回9个 accession；其中1条与现有序列完全相同，最终净增加8条。
5. 最终冻结：165条。

## 保留标记

按项目决定保留3条 provisional来源记录（2条MAG和1条uncultured），其标记保存在 `metadata.csv`，评价时可做敏感性分析。

## 使用规则

模型只能使用 `train.fasta` 训练或估计真实分布；`valid.fasta` 用于选择超参数；`test.fasta` 只用于最终评价。不得重新随机拆分，也不得把valid/test序列加入泛GvpA预训练集。
