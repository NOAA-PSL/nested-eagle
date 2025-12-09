# Global mesh

Seems like a data / mesh ratio of 2 is really the best.
With 2, we can use KNN=4 and have no hanging nodes
With 4, we have
* KNN = 16, isolated nodes = 200160 (these are everywhere)
* KNN = 24, isolated nodes = 100800 (all poleward of ~65 degrees)
* KNN = 32, isolated nodes =  70560 (all poleward of ~75 degrees)

Note that this is still better than when I was creating the latent mesh from the
wrong data node locations (i.e., I was creating them from xesmf's global grid
generation rather than taking the average of the actual data coordinates).

