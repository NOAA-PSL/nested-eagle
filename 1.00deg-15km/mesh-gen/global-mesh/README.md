# Global Mesh Generation

1. Use the notebook to create the mesh (ufs2arco env for xesmf, also take a look
   at data node / mesh node layout)
2. Use the python script to convert to npz file, sorting the anemoi way along
   the way

## Encoder: KNN vs CutOff

Key question: What minimizes the number of edges?

* KNN Edges: Lowest `nearest_neighbours`= 4, which makes sense given the design
  (2 degree global latent mesh nodes on 1 degree data nodes)
    * Results in 48600 edges
* Cutoff: Smallest `cutoff_factor` = 0.4
    * Results in 134640 edges


Given the design, KNN maybe makes the most sense. Number of edges is
dramatically reduced, so hopefully more efficient?
