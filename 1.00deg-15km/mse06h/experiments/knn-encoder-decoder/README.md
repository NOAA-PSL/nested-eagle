# Encoder / Decoder KNN Test

Test the impact of increasing the number of connections in the encoder and
decoder.

Separately vary encoder KNN (12, 24) and decoder KNN (3, 6), fixing the following parameters (at least):
* Processor architecture: sliding window transformer
* Custom latent space, essentially inherited (coarsened) from data space
* Processor window size = 4320
* Model channel width = 512
* Training steps = 30k
* LAM loss fraction = 0.5
* "Empirical" loss weights per-variable group, essentially following AIFS
* Trimmed LAM edge = (15, 16, 15, 16) grid cells  (~225km)
