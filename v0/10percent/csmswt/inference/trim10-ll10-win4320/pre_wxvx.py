import logging

import pandas as pd
import xarray as xr
from eagle.tools.log import setup_simple_log
from eagle.tools.data import open_anemoi_inference_dataset

logger = logging.getLogger("eagle.tools")

if __name__ == "__main__":

    setup_simple_log()
    forecast_dir = "/pscratch/sd/t/timothys/nested-eagle/v0/10percent/csmswt/logoffline-trim10-ll10-win4320/inference-validation"


    dates = pd.date_range("2023-02-01T06", "2024-01-20T12", freq="54h")
    for t0 in dates:
        st0 = t0.strftime("%Y-%m-%dT%H")
        logger.info(f"Processing {st0}")

        path_in = f"{forecast_dir}/{st0}.240h.nc"
        path_out= f"{forecast_dir}/lam.{st0}.240h.nc"

        logger.info(f"Opening {path_in}")
        lds = open_anemoi_inference_dataset(
            path=path_in,
            model_type="nested-lam",
            lam_index=64220,
            levels=[250, 500, 850],
            vars_of_interest=[
                "t2m",
                "gh",
                "sp",
                "t",
                "u",
                "u10",
                "v",
                "v10",
            ],
            load=True,
            reshape_cell_to_2d=True,
            lcc_info={
                "n_y": 211-11-10,
                "n_x": 359-11-10,
            },
        )
        for key in ["x", "y"]:
            if key in lds.coords:
                lds = lds.drop_vars(key)

        lds = lds.rename({"x": "longitude", "y": "latitude"})
        lds.attrs["forecast_reference_time"] = str(lds.time.values[0])
        lds.to_netcdf(path_out)
        logger.info(f"Wrote to {path_out}")
