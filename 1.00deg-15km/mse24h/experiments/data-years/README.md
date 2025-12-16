# Data Years Test

Test the impact of the years used in the fine tuning procedure.
Do we fine tune on all of the data (`base`) or just the most recent operational
version (`operational`). Note that at time of writing GFS and HRRR latest versions came online:

* GFS: 2021-03-22 12Z
* HRRR: 2020-12-02

So just to be safe I start the operational run on 2021-03-23

Vary:
* `config.dataloader.training.start` to reflect this start date (or not)
* `config.training.rollout.epoch_increment` to change how many epochs to iterate
  through during rollout training before incrementing the rollout, since the
  epoch size changes with dataset length

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
* Loss = 24h, ideally with 1440 steps per epoch increment, but this is not
  panning out to be straightforward for some reason
