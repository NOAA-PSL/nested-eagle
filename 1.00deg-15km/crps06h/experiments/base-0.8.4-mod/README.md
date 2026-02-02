# CRPS Experiments

Note: these are the same as `../base` except they work with an older version of
the code.
Specifically, they're designed to work with
[this
commit](https://github.com/ecmwf/anemoi-core/commit/20b2da6fc89c40de67d22eb33b7d29735299fac5)
of anemoi-core, which has a bugfix allowing for row-normalization of truncation
matrices (not that it actually helped though).
The biggest difference between this code and `../base` is the
[multi-dataset integration PR](https://github.com/ecmwf/anemoi-core/pull/594)
which was incorporated into `training-0.9.0` `models-0.12.0` and `graphs-0.8.4`.

* `o32/`: uses truncation of the skip connection to o32 resolution. It blows up
  during inference.
* `no-truncation/`: does not blow up, but carries around unphysical noise.

Note that these models will work by checking out anemoi-core commit `20b2da6`,
and then using even a recent version of anemoi-inference, e.g. 0.9.0.
