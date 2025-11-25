# LAM Loss Test

Find the "optimal" loss weighting for the LAM region of the nested model.

Vary LAM Loss from 0.01-0.5, fixing the following parameters (at least):
* Graph Encoder: 12 KNN determines graph encoding
* Graph Decoder: 3 KNN determines graph decoding
* Processor architecture: sliding window transformer
* Custom latent space, essentially inherited (coarsened) from data space
* Processor window size = 4320
* Model channel width = 512
* Training steps = 30k
* "Empirical" loss weights per-variable group, essentially following AIFS
* Trimmed LAM edge = (15, 16, 15, 16) grid cells  (~225km)
