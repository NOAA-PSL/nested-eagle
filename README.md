# Nested-EAGLE

A data driven forecast model with high resolution over CONUS nested in a coarser global model.

## Phase 1: Nested-ERA5

Using ERA5 data for training & evaluation, with:
* 1/4 degree resolution over CONUS
* 1 degree resolution globally

## Phase 2: Coarse Nested-EAGLE

Pretrain with ERA5 globally, then:
* HRRR conservatively regridded to 12km over CONUS
* GFS conservatively regridded to 1 degree globally

## Phase 3: Nested-EAGLE version 1

Pretrain with ERA5 globally, then:
* HRRR at 3km over CONUS
* GFS at 1/4 degree globally

## More details

Check out the
[Development Document](https://docs.google.com/document/d/12UQUZzQTdlnIwQ-DMKPyCPTsgUyKeorhJM1WLtvorI8/edit?usp=sharing)
for more details.
