# Nested-EAGLE Deterministic
## Resolution: 1 degree global - 15km CONUS

The general process:
1. Create data, see `data/submit_ufs2arco.sh`
2. Train the model, see `training/submit_training.sh`
3. Inference, evaluation, etc
    * For inference, see `inference-and-beyond/submit_inference.sh`
    * For CPU based evaluation using eagle-tools, see
      `inference-and-beyond/submit_metrics.sh`

Right now, the last step computes evaluation metrics with respect to HRRR analysis.
This can be generalized to compute
* spatial error maps (`eagle-tools spatial`)
* spectra (`eagle-tools spectra`)
* visualizations (`eagle-tools figures` or `eagle-tools movies`)

Check out [eagle-tools](https://github.com/NOAA-PSL/eagle-tools) for more,
and use `eagle-tools --help` or e.g. `eagle-tools metrics --help` for more info.

## TODO:

These items are either done or basically done, but need some cleaning up.

- [ ] global eval vs GFS analysis
- [ ] global eval vs conventional observations
- [ ] CONUS eval vs conventional observations


## Installation

The module load statements work for perlmutter, but may not be appropriate for
other machines.

```
conda env create -n eagle python=3.11
conda install -c conda-forge ufs2arco matplotlib cartopy cmocean
module load gcc cudnn nccl
pip install 'torch<2.7' anemoi-datasets==0.5.26 anemoi-graphs==0.6.4 anemoi-models==0.9.2 anemoi-training==0.6.2 anemoi-inference==0.7.1 anemoi-utils==0.4.35 anemoi-transform==0.1.16
pip install 'flash-attn<2.8' --no-build-isolation
pip install git+https://github.com/timothyas/xmovie.git@feature/gif-scale
pip install eagle-tools
```
