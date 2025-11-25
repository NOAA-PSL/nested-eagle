# 1.00 Degree - 15km HRRR
## Deterministic training with single step (6h) MSE loss
## Graph Encoder - Sliding Window Transformer Processor - Graph Decoder

Main features obtained from experimentation:
* Custom latent space, essentially inherited (coarsened) from data space
* Model channel width = 512
* Processor window size = 4320
* Training steps = 30k
* LAM loss fraction = 0.1
* "Empirical" loss weights per-variable group, essentially following AIFS
* Trimmed LAM edge = (10, 11, 10, 11) grid cells  (~50km)

For some empirical justification of these choices, see the `experiments/`
directory.
