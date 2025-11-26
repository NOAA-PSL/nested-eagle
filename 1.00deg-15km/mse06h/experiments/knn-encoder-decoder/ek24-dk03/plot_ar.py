import os
import sys
_nested_eagle = os.path.expandvars("${HOME}/nested-eagle")
sys.path.append(_nested_eagle)

from eagle.visualize import main

if __name__ == "__main__":
    inference_dir = "/pscratch/sd/t/timothys/nested-eagle/v0/10percent/csmswt/ek24-win4320/inference"

    main(
        read_path=f"{inference_dir}/2023-03-09T00.240h.nc",
        store_dir=inference_dir,
        t0="2023-03-09T00",
        tf="2023-03-10T00",
        mode="figure",
        trim_lam_edge=[15, 16, 15, 16],
        name="csmswt-ek24-win4320",
    )
    main(
        read_path=f"{inference_dir}/2023-03-09T00.240h.nc",
        store_dir=inference_dir,
        t0="2023-03-09T00",
        tf="2023-03-19T00",
        mode="movie",
        trim_lam_edge=[15, 16, 15, 16],
        name="csmswt-ek24-win4320",
    )
