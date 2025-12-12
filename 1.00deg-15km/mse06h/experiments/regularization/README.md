# Regularization Test

Test the impact of a larger regularization value.
This is motivated by Bo Huang's results, where he showed that using a larger
weight decay value (the value GraphCast used) resulted in more stable emulators.
We were seeing emulators trained on >20 years of data blow up in RMSE skill
beyond 5 days.
Increasing the weight decay resulted in emulators that performed better and did
not experience this "blow up", although more training was required to get to the
same skill.

Vary from `weight_decay`=0.01 to 0.1.
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

## wp01

This test should be a repeat of the `training-steps/steps030k` experiment, but
with an updated version of the code.
