# Independent training-seed retest and submission decision

The selected optimization settings were retrained from scratch with training seed 7. Generation used seed 42. This is a true second training run, unlike the earlier sampling-seed checks.

| Model | Primary training result | Independent retrain result | Decision |
|---|---|---|---|
| LSTM | PF00741 0.98; novel rate 0.30; exact-copy rate 0.00 | PF00741 1.00; novel rate 0.28; exact-copy rate 0.00; AA L1 0.031; no hydrophobic run >=8 | Keep as the high-fidelity model. The improvement over the original memorizing LSTM is reproduced. |
| VAE | PF00741 0.97; novel rate 0.36; empirical-length generation | PF00741 0.83; novel rate 0.33; exact-copy rate 0.06; no hydrophobic run >=8 | Keep the length-control implementation and retain VAE as a comparison/instability result. Do not present the first run as a stable performance claim. Lower temperatures (0.3/0.4) reduced PF00741 further to 0.75. |
| GAN | PF00741 1.00; novel rate 0.99; no hydrophobic run >=8 | Raw output: PF00741 1.00; novel rate 1.00; one sequence with hydrophobic run 8. With `--max-hydrophobic-run 7`: PF00741 1.00; novel rate 1.00; no such sequences. | Keep as the constrained high-novelty model. Report the final output filter explicitly. |

## Interface correction discovered during retest

`<UNK>` was a tokenizer-only symbol, but previous LSTM/VAE generation did not mask it. FASTA decoding skipped that token and could silently shorten a sequence by one residue. Generation now masks `<UNK>` together with `<PAD>` and `<BOS>`. Target-length VAE generation was rechecked after the fix: all 100 retest sequences were within the 59--91 aa train range.

## Submission decision

Submit this version with the following interpretation:

- LSTM is the most reliable high-fidelity model after optimization.
- GAN is a useful high-novelty constrained generator, not an automatic overall winner because its AA-composition L1 remains about 0.169.
- VAE demonstrates the effect of length control but also the small-data sensitivity of latent-variable generation; include it in the comparison and failure analysis rather than selecting it as the sole preferred model.

The final candidate metrics are stored under `runs/opt_lstm_pretrain_finetune_seed42/`, `runs/formal_vae_50ep_seed42/`, and `runs/opt_gan_composition_hydro_seed42/`. Independent retraining artifacts are stored under `runs/retest_*_seed7/`.
