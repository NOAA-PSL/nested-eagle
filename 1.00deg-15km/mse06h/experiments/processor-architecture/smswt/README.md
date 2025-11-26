# Training


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

```
  | Name    | Type                 | Params | Mode
---------------------------------------------------------
0 | model   | AnemoiModelInterface | 66.1 M | train
1 | loss    | WeightedMSELoss      | 0      | train
2 | metrics | ModuleList           | 0      | train
---------------------------------------------------------
66.1 M    Trainable params
0         Non-trainable params
66.1 M    Total params
264.555   Total estimated model params size (MB)
244       Modules in train mode
0         Modules in eval mode
```
