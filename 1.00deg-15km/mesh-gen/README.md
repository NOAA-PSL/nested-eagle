# Mesh Generation

Given that the first successfully trained model had GFS/HRRR boundary artifacts,
it seemed like a good idea to take a closer look at the graph definitions.
It turns out that the encoder had a lot of isolated data nodes, i.e. disconnected
from the latent space.
Some of these isolated nodes were at the GFS/HRRR boundary, but many were across
the poles.
Each of these directories show different ways to handle the effects.

Note that between the two methods for creating edges from nodes, KNN and Cutoff,
cutoff is typically used for global models.
However, for nested it doesn't make sense, since a cutoff distance that works
for the coarse-resolution component wouldn't work for the high-resolution
subdomain, and vice versa.


## MMGT

All the graph definitions targeted at the multimesh graph transformer type
Note that I did not play with the `margin_radius_km` parameter, kept it at 11km.
This changes
the max threshold used to figure out neighbors, only used for a LAM setup.
I don't fully understand this parameter, but figured it wasn't worth it until
it becomes necessary.

* `mmgt`: a recreation of what I had trained, since the yaml has to be written
  differently to use anemoi-graphs standalone.
* `mmgt-nbd`: same as `mmgt`, except modified so that there are no disconnected
  graph nodes around the GFS/HRRR boundary.
  Note that there are still disconnected nodes close to the poles, but we ignore
  these for now.
  The only real parameter to modify so that we don't just discard hanging nodes is
  the number of nearest neighbors used to define the encode/decode graphs.
  So, this just has a larger KNN parameter, where this was chosen to approximately
  be the smallest without hanging nodes near CONUS.
  18 didn't work, 20 did. I didn't try 19.
* `mmgt-tnbd`: same as `mmgt-nbd` except with the trimmed boundary of 15-16
  grid cells (75-80 km), based on visual inspection of several fields and trying
  to minimize the trimming.
  See `plot_trim_and_pad.ipynb`.
  Once again, encoder KNN was the key parameter. 18 didn't work, 20 did.

## Single Mesh Designs

The idea here was to try creating a mesh that would work with the sliding window
transformer.
Since we're making a custom mesh here, I was able to address the issue of
hanging nodes at the poles.

* `global-mesh`: figured out ways to make graphs that deal with the hanging
  nodes at the poles.
  There are basically two options:
  1. use an x2 coarsening in latitude and longitude
  2. use a graduated coarsening, from x2 at the poles to x4 at mid
     latitudes+tropics.
     See the notebook for this creation
  3. using a coarsening factor of x4 in latitude and longitude did not work
     toward the poles, without a high encoder KNN value (something like >25,
     IDK).
* `custom-single-mesh`: combine the `global-mesh` with a coarsened HRRR mesh.
  Here all the isolated nodes were at the GFS/HRRR boundary.
  1. First, I tried using a coarsening to a factor of 4 in latitude and longitude,
     using the graduated approach for the global mesh.
     Basically I tried up to ~32 nearest neighbors or so and still had hanging
     nodes.
  2. Then I added a "sponge layer" by creating mesh points in the area of the
     HRRR domain that got trimmed out.
     This had a reduced number of isolated nodes, but still there were a
     significant number (50 or so at least) even with 20-30 nearest neighbors.
  3. Refining to x2 coarsening in latitude longitude with no sponge layer worked
     with just 14 nearest neighbors, no additional trouble.
     This increases the number of nodes by a factor of 3 or so from the
     graduated setup, but it's still about an order of magnitude smaller than the
     multimesh setup.

What has mattered for the custom mesh?

1. Needed to sort the latent mesh (graph) nodes, using the same block of code
   that's in anemoi-graphs generate/.
2. In the graph generation, the "sponge" layer helped reduce the number of
   disconnected nodes, and reduced the number of nearest neighbors
    * with 1 and 2, the boundary issues were resolved and the model looked OK
      but had poor RMSE (losing to HRRR) and developed very large scale
      instabilities
3. Making the transformer processor window size bigger brought the RMSE to
   somewhere between MMGT and HRRR for most fields except surface pressure and
   geopotential, and removed the stability issue.
   Note there are ~360 (180 + 166 = 346) longitudinal points in the latent mesh,
   which dictates the window size
    * window size = 3 * 360 = 1080 looks good, not quite as good as MMGT

What did not matter?

1. using the graph post processor to sort by edges made no discernable
   difference in skill or speed


Custom Encoder:

![custom_encoder](custom_encoder.png)

Custom Decoder:

![custom_decoder](custom_decoder.png)
