# Conditional WGAN-GP initial run

This run is an initial GAN comparison for GV02-01, not a final multi-seed result.

## Setup

- Training data: 139 cyanobacterial GvpA training sequences from `dataset_v1/train.fasta`
- Model: length-conditioned WGAN-GP
- Epochs: 100
- Seed: 42
- Critic steps: 5
- Gradient-penalty weight: 10
- Position-wise composition stabilizer weight: 5
- Training temperature: linearly annealed from 1.0 to 0.5
- Hardware used for this run: NVIDIA RTX 4060 Laptop GPU

Generated lengths are sampled from the empirical training length distribution. Length validity is therefore controlled by design and must not be presented as learned GAN performance.

## Temperature sweep

| Sampling temperature | PF00741 hit rate | Mean nearest-train identity | Novel rate (<0.95 identity) | Unique ratio | Pairwise diversity | AA-composition L1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.85 | 0.773 | 1.00 | 1.00 | 0.331 | 0.173 |
| 0.15 | 0.83 | 0.768 | 1.00 | 1.00 | 0.338 | 0.173 |
| 0.20 | 0.79 | 0.761 | 1.00 | 1.00 | 0.347 | 0.177 |
| 0.50 | 0.44 | 0.696 | 1.00 | 1.00 | 0.420 | 0.203 |
| 0.80 | not run | 0.638 | 1.00 | 1.00 | 0.481 | 0.259 |
| 1.00 | not run | 0.604 | 1.00 | 1.00 | 0.506 | 0.304 |

The 0.10 sample is the current comparison candidate. It also achieved a 1.00 hit rate against the local `GvpA_core95` profile. PF00741 is the stricter and more informative result here.

## Interpretation and limitations

- No exact training copies or duplicate generated sequences were observed in the 100-sequence sample.
- Lower sampling temperature improves family-profile fidelity but reduces diversity, showing the expected fidelity/novelty trade-off.
- PF00741 fidelity (0.85) remains below the current LSTM (1.00) and VAE (0.95) runs.
- AA-composition distance (0.173) is worse than the 3-mer baseline and the existing LSTM/VAE runs.
- A direct sequence screen found 9/100 samples with a hydrophobic run of at least 8 residues (maximum 15), versus 0/139 in the training set. These samples must be treated as failures even when some pass PF00741.
- Results use one training seed and 100 generated sequences. Final claims require multiple seeds and the same full fidelity evaluator used for all models.
