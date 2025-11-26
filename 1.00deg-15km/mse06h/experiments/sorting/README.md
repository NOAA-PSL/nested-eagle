# Test if sorting or unsorting nodes or edges matters

This was initially launched because the sliding window transformer was producing
terrible results, and the nested region seemed to be missed completely in
forecasts (storms would stop or propagate around CONUS).
This initial is captured by the `nodeunsorted.yaml` setup.

Sorting the nodes while generating the latent mesh fixed the issue, and is a
critical step to mesh generation.
See more in the `mesh-gen` directories.

The other experiment was testing the effect of sorting the edges or keeping them unsorted.
It does not matter.
