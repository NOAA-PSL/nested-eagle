# Nested-EAGLE
## Production Grade Models

To create an environment that should work with this setup,

```
conda env create -f environment.yaml
conda run -n eagle pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1
```

However, there are some complications here.
First, the anemoi-datasets version is somewhere between 0.5.28 and 0.5.29.
More importantly though, these models were developed using
[this feature branch](https://github.com/timothyas/anemoi-core/tree/feature/simple-noise).
This branch sits on top of:

```
anemoi-graphs==0.8.4
anemoi-models==0.12.0
anemoi-training==0.9.0
```

which are the versions listed in `environment.yaml`.
However, some of the that feature branch includes some bugfixes that have not yet been
released to pypi at time of writing this note.
The fix is only important for transfer learning or forking, which we do here
in the multi-stage training.
There are a number of ways to make the fix necessary.
My feature branch includes
[this fix](https://github.com/ecmwf/anemoi-core/pull/950)
but
[this PR](https://github.com/ecmwf/anemoi-core/pull/867)
also appears to fix the problem, and it was merged into the codebase.

All that said, hopefully these versions should work for inference using the
existing checkpoint.
However, if any issues arise, then the versions listed in
`strict.requirements.txt` should be helpful, since this lists packages with the
specific github commits that were used for development.
