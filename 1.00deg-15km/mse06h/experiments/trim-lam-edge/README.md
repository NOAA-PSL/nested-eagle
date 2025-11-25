# Trim LAM Edge Test

Try to find how far to trim into the LAM boundary.

Vary trim parameter from 10-20, fixing the following parameters (at least):
* Graph Encoder: 12 KNN determines graph encoding
* Graph Decoder: 3 KNN determines graph decoding
* Processor architecture: sliding window transformer
* Custom latent space, essentially inherited (coarsened) from data space
* Processor window size = 4320
* Model channel width = 512
* Training steps = 30k
* LAM loss fraction = 0.5
* "Empirical" loss weights per-variable group, essentially following AIFS
