# GvpA 忠实性指标评价

## 完成内容

本次新增 `evaluation/evaluate_faithfulness.py`，并将 `faithful` 逐序列字段和
`faithfulness_rate` 汇总字段接入 `evaluation/evaluate_all.py`。

忠实性（faithfulness）定义为生成序列同时满足以下三个条件的比例：

1. 只包含 20 种标准氨基酸；
2. 长度位于冻结数据集 `dataset_v1/train.fasta` 的训练长度范围；
3. 使用 HMMER 对冻结的 GvpA profile（默认 `PF00741.hmm`）扫描时命中，且
   E-value 不大于 `1e-3`。

这个定义把“是否仍然是符合 GvpA 家族约束的序列”与“新颖性、重复率、训练集
相似度”等指标分开。HMMER 不可用或未完成扫描时不会生成伪造分数；独立脚本会
直接失败并报告扫描状态。

## 评价脚本

在本目录（`gvpa-sequence-generation/`）下运行：

```bash
python3 evaluation/evaluate_faithfulness.py \
  --generated runs/formal_lstm_50ep_seed42/generated.fasta \
  --train data/processed/dataset_v1/train.fasta \
  --profile data/processed/dataset_v1/profiles/PF00741.hmm \
  --hmmsearch /home/pppe/miniconda3/envs/gvpa-ml/bin/hmmsearch \
  --output-dir runs/formal_lstm_50ep_seed42/faithfulness_v1
```

脚本输出：

- `faithfulness_metrics.json`：本次运行的忠实性汇总；
- `faithfulness_per_sequence.csv`：每条序列的三个条件及最终判定；
- `hmmsearch.tblout`：HMMER 原始扫描结果。

如需同时运行旧版全量评价器，可使用 `evaluation/evaluate_all.py`；其
`metrics.json` 现在也包含 `faithfulness_rate` 和
`faithfulness_definition`。

## 已完成评价

使用 `dataset_v1` 训练长度范围（59--91 aa）、PF00741 profile 和
`E-value <= 1e-3`，对已有正式生成结果得到：

| 运行 | 样本数 | 合法字符率 | 长度范围率 | profile 命中率 | 忠实性率 |
|---|---:|---:|---:|---:|---:|
| formal LSTM（temperature=0.8） | 100 | 1.00 | 1.00 | 1.00 | **1.00** |
| formal VAE（latent-scale=1，temperature=0.8） | 100 | 1.00 | 1.00 | 0.95 | **0.95** |

结果文件分别位于：

- `runs/formal_lstm_50ep_seed42/faithfulness_v1/`
- `runs/formal_vae_50ep_seed42/faithfulness_v1/`

该指标只衡量生成结果对预先声明的序列/家族约束的遵循程度，不代表功能活性、
结构正确性或实验可行性；这些结论仍需相应的结构和实验验证。
