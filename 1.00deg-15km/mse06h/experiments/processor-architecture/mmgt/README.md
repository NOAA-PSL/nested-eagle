# Training

Multi Mesh Graph transformer Architecture.

## Package stack

```
pip install anemoi-datasets==0.5.23 anemoi-graphs==0.5.2 anemoi-models==0.5.0 anemoi-training==0.4.0
pip install 'earthkit-data<0.14.0'

conda uninstall mlflow mlflow-skinny mlflow-ui

pip install mlflow azureml-core azureml-mlflow
```

With the exception that anemoi-core uses
[this feature branch](https://github.com/timothyas/anemoi-core/tree/feature/azure-mlflow)
with Azure ML MLFlow logging capabilities.


## Run it

```
source ~/.azure.sh
conda activate anemoi
srun --jobid $SLURM_JOBID ~/anemoi-house/slurm2ddp.sh anemoi-training train --config-name=config
```

## Base

Defined in `config.yaml`
Key hyperparameters here are the latent graph resolutions. Right now with
1degree + 15km, using refinements


```
  | Name    | Type                 | Params | Mode
---------------------------------------------------------
0 | model   | AnemoiModelInterface | 71.5 M | train
1 | loss    | WeightedMSELoss      | 0      | train
2 | metrics | ModuleList           | 0      | train
---------------------------------------------------------
71.5 M    Trainable params
0         Non-trainable params
71.5 M    Total params
286.135   Total estimated model params size (MB)
309       Modules in train mode
0         Modules in eval mode
```

## NBD

Defined in `nbd.yaml`.
Multi Mesh Graph Transformer with No Boundary Disconnected nodes.
This is a copy of `../mmgt` except that the graph definition was modified so
that there are no disconnected graph nodes around the GFS/HRRR boundary.
Note that there are still disconnected nodes close to the poles, but we ignore
these.

The only real parameter to modify so that we don't just discard hanging nodes is
the number of nearest neighbors used to define the encode/decode graphs.
So, this just has a larger KNN parameter, where this was chosen to approximately
be the smallest without hanging nodes near CONUS.

```
  | Name    | Type                 | Params | Mode
---------------------------------------------------------
0 | model   | AnemoiModelInterface | 73.5 M | train
1 | loss    | WeightedMSELoss      | 0      | train
2 | metrics | ModuleList           | 0      | train
---------------------------------------------------------
73.5 M    Trainable params
0         Non-trainable params
73.5 M    Total params
294.179   Total estimated model params size (MB)
309       Modules in train mode
0         Modules in eval mode
```

## TNBD

Defined in `tnbd.yaml`.
This is the same as the `nbd` setup, but with a trimmed boundary to see if this
would help alleviate boundary artifacts.
