# 1.00 Degree - 15km HRRR
## Pre Training: Deterministic training with single step (6h) MSE loss
## This stage: Fine tune with multi step rollout up to 24h

Pre Training: see `../mse06h`

Main features obtained from experimentation:
* Graph Encoder: 12 KNN determines graph encoding
* Graph Decoder: 3 KNN determines graph decoding
* Processor architecture: sliding window transformer
* Custom latent space, essentially inherited (coarsened) from data space
* Model channel width = 512
* Processor window size = 4320
* Training steps = 30k
* LAM loss fraction = 0.1
* "Empirical" loss weights per-variable group, essentially following AIFS
* Trimmed LAM edge = (10, 11, 10, 11) grid cells  (~150km)
