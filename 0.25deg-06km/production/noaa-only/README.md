# Nested-EAGLE Production
## 0.25deg-06km GFS-HRRR

* Stage 1a: 1 step training for 60k optimization steps
* Stage 1b: 2 step training for ~6k optimization steps (8 epochs)
* Stage 1c: 2-4 step training for ~6k optimization steps (8 epochs)

## Walltime

* Stage 1a: 30 hours (including restarting)
* Stage 1b: 6 hours 20 minutes
* Stage 1c: 8 hours 45 minutes

## Memory

All assume sharding model across 4 GPUs, prefetch factor = 1, batch size = 1
using the A100 nodes on Perlmutter (either 40GB or 80GB).

Note that with some minimal testing I didn't see a difference in compute time or
memory usage when changing the `num_chunks` parameter, which is the number of
adjoint checkpoints.

Number of workers refers to the number of training and validation workers, I
used the same number for both.

* Stage 1a: fits comfortably on 40GB A100s with 8 workers, using max ~26GB
* Stage 1b: uses ~30GB memory with 8 workers
* Stage 1c: tested with 4 rollout steps,
    * 4 workers fits on 40GB A100s
    * 8 workers crashes on 40GB, works on 80GB but is no faster than just using
      4 workers
