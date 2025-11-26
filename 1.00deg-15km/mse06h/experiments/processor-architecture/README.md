# Test Processor Architecture

Here we have directories which contain three different setups:

* mmgt: MultiMesh Graph Transformer. This is the architecture that Met Norway
  used for Bris, it's essentially a direct translation to our setup with high
  resolution over CONUS using HRRR data.
* csmswt: Custom Single Mesh Sliding Window Transformer. This uses a "custom"
  latent mesh derived by essentially coarsening and combining the global and LAM
  data grids. See more in `mesh-gen`.
  The directory here is a copy of the `window-size` experiment, so all the tests
  here can be compared to that setup.
* smswt: Single Mesh Sliding Window Transformer. The idea here was to use a
  single resolution mesh across the globe, without having high resolution in the
  latent space over the LAM region.
  So far, I haven't had time to chase this down.

In summary, it was found that with a large enough window size, the `csmswt`
setup can produce results that are as accurate as the `mmgt` architecture, while
eliminating artifacts that show up in the mmgt solutions at the global/LAM
boundary.
Note that there are a couple of setups in the `mmgt` directory aimed at trying
to eliminate these artifacts (manage hanging nodes, change the boundary
trimming).
But I could not find a systematic way to handle these issues, whereas the
sliding window transformer processor produced clean results.

Additionally, thanks to flash attention, the sliding window processor is ~30-40%
more efficient.
