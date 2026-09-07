# 模型组使用说明

推荐实验只有两条，保持结果容易解释：

```text
实验A：dataset_v1/train.fasta → 从头训练 → 固定valid/test评价
实验B：pretrain_plus_blue_train.fasta → 预训练
       blue_finetune_train.fasta → 微调 → 同一固定valid/test评价
```

实验B微调时从较小学习率开始（例如预训练学习率的0.1–0.3倍），根据蓝藻valid集早停。任何情况下都不要把蓝藻valid/test加入预训练文件。
