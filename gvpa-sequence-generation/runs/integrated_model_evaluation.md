# Integrated GvpA Model Evaluation

## Decision rule

No overall score is used. A model must first pass the fidelity screen: 100% legal and in-range outputs, plus PF00741 hit rate >= 0.95. Novelty and diversity are compared only after that screen.

Novelty is measured against all frozen cyanobacterial GvpA sequences in `data/processed/dataset_v1/all.fasta`. A near duplicate requires aligned-residue identity >= the displayed threshold and bilateral coverage >= 0.80. Thus short fragments cannot become novel merely because they align poorly.

## Results

| Model | Fidelity screen | Profile | Strict novelty (<0.95) | Pairwise diversity | AA L1 | Interpretation |
|---|---|---:|---:|---:|---:|---|
| AA-frequency baseline | unassessed: profile_not_run | NA | 1.000 | 0.548 | 0.029 | Not comparable as a final candidate until the fidelity screen passes. |
| 3-mer baseline | unassessed: profile_not_run | NA | 1.000 | 0.379 | 0.082 | Not comparable as a final candidate until the fidelity screen passes. |
| LSTM (auxiliary pretrain + fine-tune) | pass | 0.980 | 0.420 | 0.131 | 0.040 | Eligible for fidelity--novelty--diversity Pareto comparison. |
| VAE (empirical length control) | pass | 0.970 | 0.620 | 0.176 | 0.098 | Eligible for fidelity--novelty--diversity Pareto comparison. |
| GAN (composition + hydrophobic filter) | pass | 1.000 | 1.000 | 0.157 | 0.170 | Profile passes, but composition drift makes this an exploratory candidate. |

## Reading the table

- Higher novelty and diversity are not automatically better: they must be interpreted together with the fidelity screen and composition distance.
- A missing profile value means unassessed, not failed. Such a baseline is retained only as a methodological comparison.
- The CSV retains the three novelty thresholds (0.90, 0.95 and 0.98), exact-copy rate and duplicate statistics for the final report or plots.
