from visualize import main

if __name__ == "__main__":
    inference_dir = "/pscratch/sd/t/timothys/aneml/nested-conus/era5-nest/inference/c080c4bf-7c5a-4f8d-ae86-b070d4e432e1"
    main(
        read_path=f"{inference_dir}/forecast.4320hr.2019-01-01T06.nc",
        store_dir=inference_dir,
        t0="2019-01-01T06",
        tf="2019-01-01T12",
        mode="figure",
    )

    main(
        read_path=f"{inference_dir}/forecast.4320hr.2019-01-01T06.nc",
        store_dir=inference_dir,
        t0="2019-01-01T06",
        tf="2019-06-30T00",
        mode="movie",
    )
