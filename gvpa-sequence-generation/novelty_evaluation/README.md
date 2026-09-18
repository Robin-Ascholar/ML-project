# GvpA 新颖性评估模块

该模块独立评价生成 GvpA 的序列新颖性，并在提供 PDB 文件时评价结构新颖性与 GvpA 结构保守性。

## 指标

序列指标：

- 合法字符、非空序列和天然参考长度范围
- exact copy
- 最近参考序列 identity、full-length identity 和双向 coverage
- near duplicate：默认 identity >= 95%，且双向 coverage >= 80%
- sequence novelty：`1 - nearest full-length identity`
- 3-mer Jaccard novelty
- 生成集合 unique ratio、重复比例和 pairwise diversity
- 可选 HMMER profile hit、E-value 和 bit score

结构指标：

- PDB CA 原子覆盖率
- 序列引导 Kabsch 刚体叠合后的 CA RMSD
- query/target normalization TM-score 及其对称均值
- GDT-TS
- PDB CA B-factor 均值和最小值；对 AlphaFold/ESMFold PDB 可解释为 pLDDT
- structure novelty：`1 - nearest symmetric TM-score`

这里的 TM-score 是序列引导叠合后的项目内近似值，不冒充外部 `TM-align`。GvpA 很短，固定的 TM-score 阈值只用于初筛，正式结论应使用天然 GvpA 结构的 leave-one-out 分布校准。

## 仅评价序列

在项目目录 `gvpa-sequence-generation/` 下运行：

```bash
python3 -m novelty_evaluation.evaluate \
  --generated runs/xr1_vae_5ep/generated.fasta \
  --reference data/processed/dataset_v1/all.fasta \
  --output-dir runs/xr1_vae_5ep/novelty
```

加入 GvpA HMM profile：

```bash
python3 -m novelty_evaluation.evaluate \
  --generated runs/xr1_vae_5ep/generated.fasta \
  --reference data/processed/dataset_v1/all.fasta \
  --profile data/processed/dataset_v1/profiles/PF00741.hmm \
  --output-dir runs/xr1_vae_5ep/novelty
```

如果系统没有 `hmmsearch`，序列和结构指标仍会运行，profile 状态记录为 `hmmsearch_unavailable`。

## 加入结构评价

结构目录使用 `<FASTA sequence_id>.pdb` 命名。每个 PDB 默认选择 CA 原子最多的链：

```bash
python3 -m novelty_evaluation.evaluate \
  --generated runs/my_model/generated.fasta \
  --reference data/processed/dataset_v1/all.fasta \
  --generated-structures runs/my_model/pdb \
  --reference-structures data/processed/dataset_v1/pdb \
  --output-dir runs/my_model/novelty
```

结构评价依赖 NumPy。没有 NumPy 时会输出 `numpy_unavailable`，不会影响序列评价。

## 输出

- `per_sequence_novelty.csv`：逐序列全部指标和候选分类
- `summary.json`：机器可读的集合汇总
- `report.md`：便于检查的简要报告

推荐优先关注 `sequence_novel_structure_conserved`。`sequence_novel_structure_divergent` 表示虽然远离天然序列，但预测结构也偏离 GvpA，需要谨慎处理。
