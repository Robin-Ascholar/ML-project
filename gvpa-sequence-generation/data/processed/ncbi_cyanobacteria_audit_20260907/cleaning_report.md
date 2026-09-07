# GvpA 严格清洗结果（strict_clean_v1）

## 结论

- 输入记录：1141
- 严格纳入：156
- 待人工/工具复核：30
- 排除：955
- 本轮完成 taxonomy、明确 GvpA 注释、NCBI CDD、完整性、字符、长度与 exact duplicate 检查。
- 本轮确认 NCBI CDD `PRK09371 / CDD:181805`；独立的本地 Pfam/HMM 扫描仍由 P2 执行。

## 决策规则

只有同时满足下列条件才进入 strict_included.fasta：

1. NCBI 当前 lineage 明确属于 Cyanobacteria/Cyanobacteriota；
2. NCBI 当前 definition/product/gene 明确支持 GvpA；
3. NCBI CDD 明确包含 `PRK09371 / CDD:181805`；
4. NCBI 明确标记 `COMPLETENESS: full length`，或 Protein feature 完整覆盖全长且无 partial 标记；
5. 未标记为 partial、fragment 或 incomplete；
6. 序列只含 20 种标准氨基酸；
7. 本地序列与当前 NCBI 序列一致；
8. 不与已纳入记录完全重复；
9. 长度处于复核范围 50–100 aa。

## 排除/复核原因统计

| 原因 | 数量 |
|---|---:|
| `exact_duplicate_sequence` | 900 |
| `partial_fragment_or_incomplete` | 508 |
| `gvpa_identity_not_explicit` | 447 |
| `gvpa_cdd_prk09371_not_confirmed` | 26 |
| `length_outside_review_range:50-100` | 19 |
| `ambiguous_or_non_gvpa_annotation` | 9 |
| `full_length_not_explicitly_confirmed` | 4 |
| `invalid_amino_acids:X` | 3 |
| `annotated_as_other_gvpA_like_protein` | 2 |
| `not_cyanobacteria_by_ncbi_lineage` | 1 |

注：同一条记录可能同时触发多个排除原因，因此上表数量不能直接相加。

## 重要限制

`strict_included.fasta` 是严格清洗候选集，还不是冻结的 dataset_v1。
P2 完成 GvpA family/profile 验证、真实 MSA 与相似度聚类后，才能正式冻结数据。
