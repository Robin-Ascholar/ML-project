# GV02-01 model optimization summary

This record preserves the original runs and adds optimized candidates. Every selected candidate generated 100 sequences for each of sampling seeds 42, 7, and 123. These are repeated sampling checks from one trained checkpoint per model, not independent multi-training-seed estimates.

## Shared evaluation

- Length range: 59--91 aa, determined from the 139 blue-GvpA training sequences.
- Family profile: PF00741 with `hmmsearch -E 1e-3`.
- Local GvpA-core profile: `GvpA_core95`; selected seed-42 samples passed at 0.99 (LSTM) or 1.00 (VAE and GAN).
- Novel sequence: nearest blue training sequence identity below 0.95.
- Hydrophobic-run screen: any run of at least 8 residues from `AILMFWVY`. The blue training set contains 0 such sequences.

## Selected candidates

| Model | Optimization selected | PF00741, mean (range) | Novel, mean (range) | Pairwise diversity, mean | AA L1, mean | Main interpretation |
|---|---|---:|---:|---:|---:|---|
| LSTM | 441-sequence auxiliary pretraining, blue fine-tuning, dropout 0.3, weight decay 1e-4, temperature 1.0 | 0.990 (0.980--1.000) | 0.320 (0.290--0.370) | 0.132 | 0.042 | Best match to blue-GvpA composition; novelty is improved substantially but still limited. |
| VAE | Original checkpoint, empirical-length forced EOS, latent scale 0.75, temperature 0.6 | 0.967 (0.960--0.970) | 0.330 (0.280--0.360) | 0.176 | 0.112 | Most balanced fidelity/diversity compromise; the pretraining variant did not outperform this checkpoint. |
| GAN | WGAN-GP with composition weight 20, hydrophobic-run weight 100, temperature 0.1 | 1.000 (1.000--1.000) | 0.997 (0.990--1.000) | 0.154 | 0.170 | Very high family pass and novelty, but composition drift remains; keep as a constrained candidate generator rather than the sole best model. |

All three selected seed-42 samples had legal amino-acid characters, lengths in range, and 0 sequences with a hydrophobic run of at least 8.

## What changed from the initial results

### LSTM

The initial LSTM had PF00741 hit rate 1.00 but only 0.01 novel rate and 0.06 exact-copy rate. Auxiliary pretraining, blue fine-tuning, dropout/weight decay, and a temperature increase removed exact copies and raised novel rate to about 0.32 while retaining about 0.99 PF00741 hit rate.

### VAE

The initial free-running VAE had 47/100 outputs at the 91-aa ceiling. `--length-mode empirical` removes this ceiling pile-up by applying a declared target-length constraint. A lower sampling temperature recovers profile fidelity: the selected setting has PF00741 0.967 and novelty about 0.33. The separately trained auxiliary-pretraining VAE was retained under `runs/opt_vae_pretrain_finetune_seed42/`, but its PF00741 range of 0.79--0.84 was not selected.

### GAN

The initial GAN had PF00741 0.85 and 9/100 seed-42 outputs with a hydrophobic run of at least 8, including one run of 15. Adding position-wise composition matching and a differentiable hydrophobic-run penalty raised PF00741 to 1.00 and reduced the abnormal run count to 0/100. Its pairwise diversity and AA composition still show that it is not automatically superior to LSTM or VAE.

## Files for reproduction

- LSTM checkpoint and outputs: `runs/opt_lstm_pretrain_finetune_seed42/`
- Selected VAE outputs: `runs/formal_vae_50ep_seed42/generated_empirical_s0.75_t0.6*.fasta`
- GAN checkpoint and outputs: `runs/opt_gan_composition_hydro_seed42/`
- Aggregate machine-readable result: `runs/optimized_model_comparison.csv`

## Remaining limitations

- Each model has one training seed; a final paper-style conclusion should retrain each selected configuration with at least two additional training seeds.
- PF00741 hit alone is not sufficient evidence of function. Use the core profile, conservation checks, and the planned ESM2 embedding analysis before treating a sequence as a candidate for biological work.
- The VAE and GAN length controls are intentional generation constraints and must be stated in the report.
