# Find Dates with Missing Data

The very first training experiments were going off the rails.
As in prediction distributions were shifting with more epochs, and eventually
the model would produce NaNs.
By training on each year of data individually we were able to track down that
the issue was due to missing data that was not initially reported by ufs2arco.
Since then, checks have been put in place in ufs2arco to catch this, see [this
point](https://github.com/NOAA-PSL/ufs2arco/commit/4b1d401091d3bf9478c611ebea6cc4575f6e6a45)
in the commit history and related PR.

These setups run with the `processor-architecture/mmgt` setup.
