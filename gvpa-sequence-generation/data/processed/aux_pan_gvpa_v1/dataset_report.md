# aux_pan_gvpa_v1 冻结报告

## 内容

- 原始高可信非蓝藻GvpA候选：655条
- 因与蓝藻valid/test达到80% identity且双方coverage不低于80%而排除：105条
- 安全辅助池：550条
- 95%代表序列：302条（默认辅助预训练集）
- 90%代表序列：168条（更强去冗余备选）
- 默认预训练组合：441条 = 302条非蓝藻代表 + 139条蓝藻train
- taxonomic domain：{'Bacteria': 343, 'Archaea': 207}
- 全部550条通过官方Pfam PF00741 gathering threshold
- 与冻结valid/test达到80%阈值的保留序列：0条

## 推荐用法

1. AA-frequency和k-mer baseline只使用蓝藻 `dataset_v1/train.fasta`。
2. LSTM/VAE的迁移实验先用 `pretrain_plus_blue_train.fasta` 预训练，再用 `blue_finetune_train.fasta` 微调。
3. valid和test始终使用蓝藻 `dataset_v1` 中的固定文件，只用于调参和最终评价。
4. 必须同时报告“只用蓝藻train从头训练”和“泛GvpA预训练后微调”两组结果。
5. 如需使用全部550条而不是95%代表集，应读取 `cluster_balanced_weight_95` 做按簇加权，避免大簇主导训练。

## 边界

该辅助集包含细菌和古菌GvpA，只用于学习跨物种GvpA序列规律，不改变项目的蓝藻目标分布，也不能替代蓝藻valid/test评价。
