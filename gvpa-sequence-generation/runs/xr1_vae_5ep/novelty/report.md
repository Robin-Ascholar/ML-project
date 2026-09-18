# GvpA Novelty Evaluation

## Summary

- Generated sequences: 20
- Valid sequence rate: 20.0%
- Exact reference copy rate: 0.0%
- Near-duplicate rate: 0.0%
- Novel rate among valid sequences: 100.0%
- Mean sequence novelty among valid sequences: 0.6709
- Unique ratio: 95.0%
- Mean generated-set pairwise diversity: 0.8113
- HMM profile status: hmmsearch_unavailable
- HMM profile hit rate: NA
- Structure comparison rate: 0.0%
- Mean nearest-reference TM-score: NA

## Candidate Classes

- `invalid_sequence`: 16
- `sequence_novel`: 4

## Top Sequence-Novel Candidates

| Sequence | Class | Novelty | Identity | Query cov. | Target cov. | TM-score |
|---|---|---:|---:|---:|---:|---:|
| vae_scale1_temp0.8_seed42_0016 | sequence_novel | 0.7222 | 0.2985 | 0.9306 | 0.9306 | NA |
| vae_scale1_temp0.8_seed42_0005 | sequence_novel | 0.6667 | 0.4091 | 0.8148 | 0.8800 | NA |
| vae_scale1_temp0.8_seed42_0011 | sequence_novel | 0.6494 | 0.4154 | 0.8442 | 0.8904 | NA |
| vae_scale1_temp0.8_seed42_0000 | sequence_novel | 0.6452 | 0.4000 | 0.8871 | 0.9016 | NA |

Sequence novelty is `1 - nearest full-length identity`. A near duplicate
requires both coverage values to pass the configured threshold. Structure
novelty is reported separately from GvpA structure conservation.
