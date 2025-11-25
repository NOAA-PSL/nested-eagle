# Window Size Test

Find the minimum necessary `window_size` for the sliding window processor.

Vary window size from 1080-4320, fixing the following parameters (at least):
* Graph Encoder: 12 KNN determines graph encoding
* Graph Decoder: 3 KNN determines graph decoding
* Processor architecture: sliding window transformer
* Custom latent space, essentially inherited (coarsened) from data space
* Model channel width = 512
* Training steps = 30k
* LAM loss fraction = 0.5
* "Empirical" loss weights per-variable group, essentially following AIFS
* Trimmed LAM edge = (15, 16, 15, 16) grid cells  (~225km)
