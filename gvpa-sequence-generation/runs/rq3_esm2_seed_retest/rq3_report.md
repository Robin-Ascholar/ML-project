# RQ3: PLM Embedding Distribution Analysis

## Question and scope

This analysis asks whether the generated sets occupy a representation-space distribution compatible with frozen cyanobacterial GvpA, rather than merely matching simple sequence statistics. It is computational evidence only: it does not prove protein function.

## Method

- PLM: `facebook/esm2_t6_8M_UR50D`; last-layer mean pooling over residue tokens; special and padding tokens excluded
- Target distribution: 165 frozen cyanobacterial GvpA sequences from `data/processed/dataset_v1/all.fasta`.
- Natural-variation baseline: repeated random real-vs-real half splits; MMD^2 median 0.0089, 95% interval [0.0044, 0.0206].
- Outlier threshold: 0.1489, the 95th percentile of held-out real-to-train embedding distances.
- The two baselines are negative controls: they remain in the plot/table even when their family-profile evidence is incomplete.

## Results

| Source | MMD^2 median | MMD beyond real q95 | Outlier rate | Assessment |
|---|---:|---:|---:|---|
| Real cyanobacterial GvpA | 0.0089 | 5.0% | 0.0% | Natural real-vs-real baseline; not a generated-model judgment. |
| LSTM / sample seed 42 | 0.0221 | 94.0% | 6.0% | Partly overlaps real GvpA, but distributional mismatch exceeds the real-vs-real baseline. |
| LSTM / sample seed 7 | 0.0240 | 96.5% | 8.0% | Partly overlaps real GvpA, but distributional mismatch exceeds the real-vs-real baseline. |
| LSTM / sample seed 123 | 0.0330 | 100.0% | 9.0% | Partly overlaps real GvpA, but distributional mismatch exceeds the real-vs-real baseline. |
| VAE / sample seed 42 | 0.2710 | 100.0% | 61.0% | Substantial embedding outlier rate; evidence of distributional drift. |
| VAE / sample seed 7 | 0.3075 | 100.0% | 70.0% | Substantial embedding outlier rate; evidence of distributional drift. |
| VAE / sample seed 123 | 0.2788 | 100.0% | 69.0% | Substantial embedding outlier rate; evidence of distributional drift. |
| GAN / sample seed 42 | 0.7216 | 100.0% | 97.0% | Substantial embedding outlier rate; evidence of distributional drift. |
| GAN / sample seed 7 | 0.7587 | 100.0% | 95.0% | Substantial embedding outlier rate; evidence of distributional drift. |
| GAN / sample seed 123 | 0.7382 | 100.0% | 97.0% | Substantial embedding outlier rate; evidence of distributional drift. |

## Interpretation rule

A small MMD relative to real-vs-real variation and a low calibrated outlier rate support distributional compatibility. A high profile hit rate without this support indicates that local family motifs can be present even when the global representation distribution drifts. UMAP is a visualization only; conclusions use the high-dimensional MMD and outlier statistics.

## Limitations

Each row is one sampling-seed replicate from the selected checkpoint; the companion `model_seed_retest.md` reports the mean and range across the three available seeds for each deep model. This is a sampling-stability check, not an independent model-retraining repeat. ESM-2 embeddings are not functional assays and should be interpreted alongside profile, conservation and composition evidence.
