# Diversity experiment results

All configurations generated 1,000 sequences with the frozen `dataset_v1` training split as reference.

- `baseline_results.csv`: AA-frequency and k-mer-3 baselines, seeds 42/43/44.
- `deep_model_results.csv`: 30-epoch LSTM temperature and VAE latent-scale experiments, seed 42.
- `*_eval/metrics.json`: unified evaluation output for each run.
- `*_eval/per_sequence_metrics.csv`: sequence-level metrics.

The LSTM temperature sweep shows the expected diversity/valid-length trade-off: temperature 0.7 is conservative and highly similar to training data, while 1.3 is more diverse but produces more out-of-range lengths. The VAE latent-scale sweep gives a smaller but consistent increase in pairwise diversity as latent scale increases.

These results are exploratory because profile/HMMER validation and multi-seed deep-model repeats are not yet included.
