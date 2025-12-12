# 1.00 Degree - 15km HRRR
## Deterministic training with single step (6h) MSE loss


Main features obtained from experimentation:
* Graph Encoder: 12 KNN determines graph encoding
* Graph Decoder: 3 KNN determines graph decoding
* Processor architecture: sliding window transformer
* Custom latent space, essentially inherited (coarsened) from data space
* Model channel width = 512
* Processor window size = 4320
* Training steps = 30k
* LAM loss fraction = 0.1
* "Empirical" loss weights per-variable group, essentially following AIFS
* Trimmed LAM edge = (10, 11, 10, 11) grid cells  (~150km)
* weight decay = 0.01 (doesn't matter though)

For some empirical justification of these choices, see the `experiments/`
directory.

The config for the current "best" model is a soft link to the appropriate yaml
and inference-evaluation directory within the experiments.
Right now it is the same setup as in:

* `experiments/training-steps/steps030k.yaml`
* `experiments/regularization/wp01.yaml`

except the latter has the updated package versions that are used here.


## Package stack

```
eagle-tools==0.5.0
ufs2arco==0.18.0
```

See eagle-tools
[installation instructions](https://github.com/NOAA-PSL/eagle-tools?tab=readme-ov-file#installation)
for more details on other packages, and
the anemoi versions are as defined by the submodules
[here](https://github.com/timothyas/aneml).

## Installation on Perlmutter

The module load statements work for perlmutter, but may not be appropriate for
other machines.

Note that most of this should work with the versions noted here, but some of the
experiments may not...

```
conda env create -n eagle python=3.11
conda install -c conda-forge ufs2arco>=0.17.2 matplotlib cartopy cmocean
module load gcc cudnn nccl
pip install 'torch<2.7' anemoi-datasets==0.5.26 anemoi-graphs==0.6.4 anemoi-models==0.9.2 anemoi-training==0.6.2 anemoi-inference==0.7.1 anemoi-utils==0.4.35 anemoi-transform==0.1.16
pip install 'flash-attn<2.8' --no-build-isolation
pip install git+https://github.com/timothyas/xmovie.git@feature/gif-scale
pip install eagle-tools==0.4.0
```
