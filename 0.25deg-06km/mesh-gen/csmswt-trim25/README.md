# Single stage graph

Differences from the 1.00deg-15km mesh generation
* used `min_distance_km=6` for data nodes here, 0 before
* used `min_distance_km=12` for mesh nodes here, 0 before
  * using 24km, the distance between nodes in the HRRR mesh, seemed
    too large for the regions between HRRR and global mesh. It seems like
    having more latent mesh nodes here, rather than fewer, will reduce the
    amount of isolated nodes.
* latent mesh is a x4 coarsening rather than x2. Because of grid node
  convergence at the poles in the lat/lon mesh, it is nearly impossible to
  eliminate all isolated nodes at the pole.
  The tests below show how many isolated nodes there are for a given KNN or
  cutoff value.
  Note that for reference in GraphCast there are ~92k isolated nodes starting at
  about +/- 70 degrees latitude.
  The 1.00deg-15km setup used encoder KNN=12, and here there are ~4x more data
  nodes per latent node.

## KNN Sweep

* KNN=16, isolated nodes = 196108 all along HRRR border ...
* KNN=24, isolated nodes =  98076, all along HRRR border and +/-~70deg
* KNN=32, isolated nodes =  67726, still some along HRRR border and +/-75deg
* KNN=48, isolated nodes =  41760, +/- 80deg
