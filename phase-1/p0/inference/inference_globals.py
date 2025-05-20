################################
################################
##### for run_inference.py #####
################################
################################

# model checkpoint to load
CHECKPOINT = "/path/to/checkpoint/checkpoint.ckpt"

# how far out do you want to save? in hrs.
LEAD_TIME = 48

# where to save output of inference (nc files)
OUTPUT_PATH = "/enter/desired/path/here"

# file paths to what trained your model
# if not using LAM - just leave blank or with ""
LAM = True
LAM_PATH = "/path/to/data/conus.validation.zarr"
GLOBAL_PATH = "/path/to/data/global.validation.zarr"

# if you are not using a LAM (e.g. just one global source for training)- otherwise leave blank or with ""
INPUT_DATA_PATH = ""

##################################
##################################
##### for create_wbx_zarr.py #####
##################################
##################################

# target grid that you will regrid inference to for wbx
WBX_TARGET_PATH = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

# vars to put through verification (assuming we will usually only grab a few variables here)
VARS_OF_INTEREST = [
    "10m_u_component_of_wind",
    "2m_temperature",
]

# target grid for regridding the LAM
# this will be a static dataset of your global domain, which is used to regrid the lam area to match global.
# e.g. era5 1-deg or GFS etc
# what I used here is the output created from create_global_grid.py before data creation with era5 nested model.
# this file is used to regrid conus to global res so they can then be combined on the same grid afterwards.
LAM_TARGET_PATH = "global_one_degree.nc"

# now this one is your lam file in its native resolution (e.g. 0.25deg era5 or hrrr)
# it is used as a mask so we can extract lam from the nested domain.
# a few options for this file ---
# 1) run inference with "extract_lam" = True for one timestep. then a file is saved that is named "lam.nc".
# 2) I also built functionality to run inference like this on the fly within the create_wbx_zarr.py,
#       so if you leave this str blank the create_wbx_zarr.py. Should save out a file "lam.nc" on the fly to then use.
# 3) You could seprately just download your own lam file (e.g. static hrrr file for when we do hrrr/gfs)
# tldr --- you just need a static lam (conus) file here that will be used to clip the nested file to the conus domain.
PATH_TO_LAM_FILE = "lam.nc"

# path to save final zarr that will then go through wbx
PATH_TO_OUTPUT_ZARR = "test.zarr"
