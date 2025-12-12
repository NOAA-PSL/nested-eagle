# Pressure scaling test

Test the impact of using a pressure level weighted loss scaling

Fixing the following parameters (at least):
* Graph Encoder: 12 KNN determines graph encoding
* Graph Decoder: 3 KNN determines graph decoding
* Processor architecture: sliding window transformer
* Custom latent space, essentially inherited (coarsened) from data space
* Processor window size = 4320
* Model channel width = 512 (except the experiments marked with nc1024)
* Training steps = 30k
* LAM loss fraction = 0.1
* "Empirical" loss weights per-variable group, essentially following AIFS
* Trimmed LAM edge = (10, 11, 10, 11) grid cells  (~150km)
