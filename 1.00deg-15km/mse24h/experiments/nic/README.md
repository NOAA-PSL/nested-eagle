# Num. Initial Condition Test

Test the impact of varying the number of initial conditions, this time after 24h
rollout training.

Fixing the following parameters (at least):
* Graph Encoder: 12 KNN determines graph encoding
* Graph Decoder: 3 KNN determines graph decoding
* Processor architecture: sliding window transformer
* Custom latent space, essentially inherited (coarsened) from data space
* Processor window size = 4320
* Model channel width = 512
* Training steps = 30k
* LAM loss fraction = 0.1
* "Empirical" loss weights per-variable group, essentially following AIFS
* Trimmed LAM edge = (10, 11, 10, 11) grid cells  (~150km)
* Loss = 24h, ideally with 1440 steps per epoch increment, but this is not
  panning out to be straightforward for some reason
