# Wxvx timing

* postprocessing the forecasts: ~200 sec
* wxvx:
    * pull in obs: 35min, via `wxvx -c obs.validation.yaml -t obs -n 4`
    * compute stats: ran serially for ~8 minutes and killed it, then
      pulled this up on an interactive node with 16 cpus per task
      `wxvx -c obs.validation.yaml -t stats -n 16` ... started at
      over two jobs... took something like 1.5-2 hours

## Parallel

```
find ./test-cycles-wxvx -maxdepth 1 -type f -name "*.yaml" > input.txt
parallel -j 8 ./parallel_wxvx.sh {} :::: input.txt
```

Suggestions:
* it would be nice to be able to create the prepbufr nc file in its own workflow
* seems like somewhere there's some bad parallelism? IDK. running stats in
  serial seems to be the only way to not see a busy text file
* unclear if this is why, but lots of missing data points in resulting stats
  runs
