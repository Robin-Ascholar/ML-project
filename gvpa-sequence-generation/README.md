# GvpA 序列数据清洗

本目录保存 GV02-01 项目的数据清洗代码与审计结果。原始课程 FASTA 保存在 `data/raw/course/GvpA.fasta`，清洗过程不会修改它。

## 当前结论

- 原始记录：856 条
- 课程数据首次严格清洗：144 条
- NCBI/UniProt 覆盖审计及严格补充：157 条
- profile复核净救回：8 条
- 最终冻结 `dataset_v1`：165 条
- train / valid / test：139 / 13 / 13
- 跨集合最高identity：94.4%；达到或超过95%的跨集合序列对为0

冻结数据位于 `data/processed/dataset_v1`。模型必须使用其中已经固定的拆分文件，不得重新随机划分或直接修改；未来如需增删序列，应另建 `dataset_v2`。

完整的数据来源、过滤、HMM、MSA、划分和限制见 `data/processed/dataset_v1/dataset_report.md`；全部冻结文件的 SHA256 位于 `data/processed/dataset_v1/checksums.sha256`。

## 辅助预训练集

非蓝藻泛 GvpA 辅助集已单独冻结为 `data/processed/aux_pan_gvpa_v1`，不会改变蓝藻主数据集的任务定义和拆分：

- 高可信非蓝藻候选 655 条；排除与蓝藻 valid/test 达到 80% identity、双方 coverage 均不低于 80% 的 105 条；安全池 550 条。
- 550 条均通过 PF00741 gathering threshold，且均为非重复、标准氨基酸序列。
- 默认辅助预训练使用 `pretrain_95rep.fasta`（302 条 95% 去冗余代表）。
- 交给模型组的默认组合文件是 `pretrain_plus_blue_train.fasta`（302 条辅助代表 + 139 条蓝藻 train，共 441 条）。
- 预训练后只在 `blue_finetune_train.fasta` 上微调；验证和测试仍使用 `dataset_v1/valid.fasta` 与 `dataset_v1/test.fasta`。
- baseline 仍只使用蓝藻 `dataset_v1/train.fasta`；LSTM/VAE 同时报告“蓝藻从头训练”和“辅助预训练后蓝藻微调”，才能判断预训练是否有用。

具体文件含义见 `data/processed/aux_pan_gvpa_v1/dataset_report.md` 和 `training_recipe.md`，完整性校验见该目录的 `checksums.sha256`。该版本同样冻结；后续改动应另建 `aux_pan_gvpa_v2`。

## 严格清洗规则

序列必须同时满足：NCBI 当前分类属于蓝藻、注释明确为 GvpA、CDD 命中 `PRK09371 / CDD:181805`、明确标记为 full length（或 Protein feature 完整覆盖全长且无 partial 标记）、只含标准氨基酸、本地序列与 NCBI 当前序列一致、长度处于设定范围，并且不是 exact duplicate。

## 复跑命令

联网获取或复用 NCBI 缓存：

```bash
python preprocessing/strict_clean.py \
  --input data/raw/course/GvpA.fasta \
  --output-dir data/processed/strict_clean_v1 \
  --cache-dir data/interim/ncbi_efetch
```

仅使用已有缓存：

```bash
python preprocessing/strict_clean.py \
  --input data/raw/course/GvpA.fasta \
  --output-dir data/processed/strict_clean_v1 \
  --cache-dir data/interim/ncbi_efetch \
  --offline
```

安装 MMseqs2 后进行冗余聚类（输出目录应为空或使用一个新的目录名）：

```bash
bash preprocessing/run_mmseqs_clustering.sh \
  data/processed/strict_clean_v1/strict_included.fasta \
  data/processed/strict_clean_v1/clustering

python preprocessing/summarize_clusters.py \
  --metadata data/processed/strict_clean_v1/metadata_all.csv \
  --clusters-95 data/processed/strict_clean_v1/clustering/mmseqs_95_cluster.tsv \
  --clusters-90 data/processed/strict_clean_v1/clustering/mmseqs_90_cluster.tsv \
  --output-dir data/processed/strict_clean_v1/clustering
```

## 主要输出

- `data/processed/dataset_v1/all.fasta`：冻结的全部165条序列
- `data/processed/dataset_v1/train.fasta`：模型训练集
- `data/processed/dataset_v1/valid.fasta`：调参验证集
- `data/processed/dataset_v1/test.fasta`：最终测试集
- `data/processed/dataset_v1/metadata.csv`：身份、来源、profile、cluster和split信息
- `data/processed/dataset_v1/all.mafft.fasta`：真实MAFFT多序列比对
- `data/processed/dataset_v1/audit/`：清洗、补充、profile、MSA和泄漏审计
- `data/processed/dataset_v1/profiles/`：冻结时使用的官方与本地profile
- `data/processed/dataset_v1/checksums.sha256`：完整性校验

## 生成与评价最小流水线

当前已补充 9 月 9 日前需要的 baseline/evaluator v0.1 骨架。以下命令均在 `gvpa-sequence-generation/` 目录下运行。

生成 AA-frequency baseline：

```bash
python3 baseline/aa_frequency.py \
  --train data/processed/dataset_v1/train.fasta \
  --num-seqs 100 \
  --seed 42 \
  --output runs/baseline/aa_frequency_seed42.fasta
```

生成 k-mer baseline：

```bash
python3 baseline/kmer.py \
  --train data/processed/dataset_v1/train.fasta \
  --k 3 \
  --num-seqs 100 \
  --seed 42 \
  --output runs/baseline/kmer3_seed42.fasta
```

评价任意统一 FASTA 输出：

```bash
python3 evaluation/evaluate_all.py \
  --generated runs/baseline/kmer3_seed42.fasta \
  --output-dir runs/baseline/kmer3_seed42_eval
```

如果环境中已安装 HMMER，可加入冻结的 GvpA profile 做 family 评价：

```bash
python3 evaluation/evaluate_all.py \
  --generated runs/baseline/kmer3_seed42.fasta \
  --profile data/processed/dataset_v1/profiles/PF00741.hmm \
  --output-dir runs/baseline/kmer3_seed42_profile_eval
```

未提供 `--profile` 时 family 评价状态为 `not_run`；提供 profile 但找不到
`hmmsearch` 时状态为 `hmmsearch_unavailable`，不会生成伪造的命中率。

汇总多个 run 的 `metrics.json`：

```bash
python3 evaluation/summarize_runs.py \
  --metrics runs/baseline/aa_frequency_seed42_eval/metrics.json runs/baseline/kmer3_seed42_eval/metrics.json \
  --output runs/model_comparison.csv
```

评价器 v0.1 已计算合法字符、长度范围、exact copy、nearest-train identity/coverage、novelty、unique ratio、pairwise diversity 和 AA composition distance。family/profile 字段当前显式为 `null/not_run`，等待 P2 接入 HMMER 扫描结果，不使用 mock 分数。

## 新颖性评价模块

独立的序列与结构新颖性评价位于 `novelty_evaluation/`。它修正了短片段高
identity 的 coverage 问题，并支持可选 HMMER profile、PDB CA RMSD、近似
TM-score、GDT-TS 和结构置信度：

```bash
python3 -m novelty_evaluation.evaluate \
  --generated runs/xr1_vae_5ep/generated.fasta \
  --reference data/processed/dataset_v1/all.fasta \
  --output-dir runs/xr1_vae_5ep/novelty
```

结果包括逐序列 `per_sequence_novelty.csv`、集合级 `summary.json` 和可读的
`report.md`。完整参数和结构目录约定见 `novelty_evaluation/README.md`。

## LSTM/VAE 入口

已补充 PyTorch 版 LSTM 与 VAE 最小训练/生成入口。已在 `xr1` 环境（Python 3.10.20、PyTorch 2.11.0+cu130）完成 CPU smoke test。运行时会自动选择 CUDA（可用时）或 CPU，也可以用 `--device cpu/cuda` 显式指定：

```bash
python3 experiments/train_lstm.py --epochs 5 --output-dir runs/lstm_smoke
python3 experiments/generate_lstm.py \
  --checkpoint runs/lstm_smoke/checkpoint.pt \
  --output runs/lstm_smoke/generated.fasta

python3 experiments/train_vae.py --epochs 5 --beta 0.01 --output-dir runs/vae_smoke
python3 experiments/generate_vae.py \
  --checkpoint runs/vae_smoke/checkpoint.pt \
  --output runs/vae_smoke/generated.fasta
```

生成出的 FASTA 可直接交给 `evaluation/evaluate_all.py`。LSTM 支持 `--temperature` 与 `--top-k`；VAE 支持 `--beta` 训练和 `--latent-scale` 生成实验。

本机没有 NVIDIA GPU，因此 CUDA kernel 未在本机实测；模型、batch 和 checkpoint 均通过统一的 `device` 参数管理，已验证 CPU device path，迁移到有兼容 NVIDIA 驱动的机器后可使用 `--device cuda`。

中间清洗输出：

- `strict_included.fasta`：严格保留的序列
- `strict_included_metadata.csv`：严格保留序列的元数据
- `removal_log.csv`：每条被排除记录及原因
- `metadata_all.csv`：全部输入记录的审计表
- `cleaning_report.md`：清洗结果说明
- `clustering/cluster_info.csv`：每条保留序列的 95%/90% 簇编号和簇大小
- `clustering/clustering_report.md`：冗余统计与补数据建议

## 限制

GvpA天然高度保守，在90%阈值下会形成相似性连通整体，因此冻结集采用95% identity、80% coverage连通组进行防泄漏划分。165条中按项目决定保留2条MAG和1条uncultured记录，并在metadata中持续标记，后续评价应补做移除这3条后的敏感性分析。
