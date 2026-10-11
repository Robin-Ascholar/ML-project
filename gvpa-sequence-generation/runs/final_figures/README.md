# M4 消融与最终图表

本目录由 `evaluation/build_final_figures.py` 从已保存的 FASTA 和 evaluator `metrics.json` 重建，不会重新训练模型。

## 图表与可直接采用的结论

| 文件 | 用途 | 可采用的结论 |
|---|---|---|
| `fig_01_optimisation_package_comparison.png` | 优化前后/约束前后的对照 | LSTM 用少量 profile 换取明显更高的新颖性和多样性，并消除了精确复制；VAE 的长度控制消除了 91 aa 上限堆积，且改善 profile 和组成，但牺牲部分探索性；GAN 约束将 profile 从 0.85 提至 1.00，但多样性下降，组成漂移仍没有根治。 |
| `fig_02_sampling_parameter_controls.png` | 固定 checkpoint 的采样参数对照 | LSTM 的 `T=1.0` 是 fidelity 与 novelty 的折中，`T=1.2` 虽新颖率高但 profile 降到 0.94；VAE 的 `scale=0.75, T=0.6` 是唯一达到 0.95 profile 筛选线的选定设置；GAN 在 0.1--0.2 的温度内几乎不变。 |
| `fig_03_final_three_objective_tradeoff.png` | RQ2 三目标权衡图 | LSTM 最接近真实组成、最保守；VAE 居中；GAN 新颖性最高但组成偏离最大。图中只放了 profile 已评估的深度模型；两个 baseline 仍需补 HMMER 后才适合并列。 |
| `fig_04_rq3_distribution_summary.png` | RQ3 三个采样种子复测汇总 | LSTM 的 MMD 和离群率都显著低于 VAE/GAN，但仍高于真实-真实自然差异；VAE/GAN 的结论在三个采样种子下稳定。 |
| `fig_05_vae_length_control.png` | 失败模式及修复图 | 自由采样 VAE 中有 47% 序列停在 91 aa 最大长度；经验长度控制将该比例降到 0。 |

## 消融的准确表述

本项目已有两类对照，报告时必须区分：

1. **优化方案比较**：`optimisation_package_comparison.csv`。LSTM、VAE、GAN 的“优化前/后”各自同时改变了多个因素，例如预训练、正则化、采样温度、长度控制或过滤。因此它们证明的是“该优化方案有效”，**不能**写成某一个单独组件导致了全部提升。
2. **固定 checkpoint 的采样参数对照**：`sampling_parameter_comparison.csv`。这些是更严格的生成阶段对照：checkpoint 固定，只调整 temperature/latent scale。可用于说明参数如何改变 fidelity--novelty--diversity 的取舍。

若课堂要求严格的训练阶段单因素消融，唯一仍可补的项目是：固定 LSTM 主训练设置后，仅去掉辅助预训练或仅去掉 dropout/weight decay，再重训一次。现有证据足以满足课程式的“组件/参数比较”，但不应把它包装成完整的论文级因果消融。

## 数据口径

- 图 1 和图 2 中的“新颖率”是相对于训练集、nearest identity `< 0.95` 的 evaluator 指标。
- 图 3 中的“严格新颖率”是相对于全部 165 条冻结蓝藻 GvpA，并同时要求双边 coverage ≥ 0.80，因此数值不可与前两图直接混用。
- 图 4 的柱为三个**生成采样种子**的均值，误差线为范围；它不是三个独立重新训练的模型。
- 所有 profile 指标使用 PF00741；profile 命中不等于功能验证。

## 复现

```bash
cd gvpa-sequence-generation
python3 evaluation/build_final_figures.py --output-dir runs/final_figures
```
