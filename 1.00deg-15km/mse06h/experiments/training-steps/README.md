# Training Steps Test

Check the benefit of training more, and see if we benefit from having a larger
latent space with more training.

The intent was to "lock in" good decisions (e.g., window size, lam loss, etc)
and then train longer. However, it turns out this test showed that we were
already training "the right amount". Which is really strange.

Vary training steps 15k - 180k. For a few of these, try with larger models
(num channels = 1024).
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
