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

For some empirical justification of these choices, see the `experiments/`
directory.

## Package stack

```
pip install anemoi-datasets==0.5.23 anemoi-graphs==0.5.2 anemoi-models==0.5.0 anemoi-training==0.4.0
pip install 'earthkit-data<0.14.0'

conda uninstall mlflow mlflow-skinny mlflow-ui

pip install mlflow azureml-core azureml-mlflow
```
