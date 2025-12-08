# Loss Scaling Tests

These are the global 1 degree versions, in order to simplify evaluation and make
training cheaper.

* `gmean-residual-stdev`: normalize using the temporal residual statistics,
  including geometric mean factor, as in ACE
* `default`: is what p0 uses, the suggested values already in Anemoi, which are
  essentially the same as what's used in AIFS. There are just a couple of
  minor differences to note (even though the second two are not super important):
    * this configuration does not weight the loss function by pressure level,
      whereas AIFS and default Anemoi do (higher weight at lower levels)
    * there are more variables in AIFS, so the "relative" weighting will be
      different
    * here, and in anemoi default, 10m u and 10m v are weighted as 0.1, whereas
      AIFS gives the 0.5
* `ones`: just make it all equal
* `ensemble-spread`: use the standard deviation from an ensemble and normalize
  by these values
