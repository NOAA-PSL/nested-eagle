# End of day notes to self

HRRR data is good to go, except both analysis and forecast could not grab
anything on date 2016-03-19

GFS is a nightmare.
This work spurred a number of PRs in ufs2arco:
* [#89](https://github.com/NOAA-PSL/ufs2arco/pull/89) was encountered when
  reading failed, so we only had forcings that had incomplete data
* [#87](https://github.com/NOAA-PSL/ufs2arco/pull/87) made reading error
  messages impossible, so with this fixed hopefully things are faster, but at
  least we can read the messages
* [#88](https://github.com/NOAA-PSL/ufs2arco/pull/88) to avoid RDA when possible
* ... still to come, the big one with all attempts to make errors passive

Some of these are independent, but all of
these were toward the end goal of making data reading errors passive.
So just collect missing data dimensions and move on, patch it up at the end.

This never really worked fully, even after putting those PRs in I would get
weird errors like this:

```python
skipping corrupted Message
Traceback (most recent call last):
  File "/global/homes/t/timothys/.conda/envs/ufs2arco/lib/python3.11/site-packages/cfgrib/messages.py", line 274, in itervalues
    yield self.filestream.message_from_file(file, errors=errors)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/global/homes/t/timothys/.conda/envs/ufs2arco/lib/python3.11/site-packages/cfgrib/messages.py", line 341, in message_from_file
    return Message.from_file(file, offset, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/global/homes/t/timothys/.conda/envs/ufs2arco/lib/python3.11/site-packages/cfgrib/messages.py", line 97, in from_file
    codes_id = eccodes.codes_grib_new_from_file(file)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/global/homes/t/timothys/.conda/envs/ufs2arco/lib/python3.11/site-packages/gribapi/gribapi.py", line 409, in grib_new_from_file
    GRIB_CHECK(err)
  File "/global/homes/t/timothys/.conda/envs/ufs2arco/lib/python3.11/site-packages/gribapi/gribapi.py", line 226, in GRIB_CHECK
    errors.raise_grib_error(errid)
  File "/global/homes/t/timothys/.conda/envs/ufs2arco/lib/python3.11/site-packages/gribapi/errors.py", line 381, in raise_grib_error
    raise ERROR_MAP[errid](errid)
gribapi.errors.WrongLengthError: Wrong message length
skipping corrupted Message
Traceback (most recent call last):
  File "/global/homes/t/timothys/.conda/envs/ufs2arco/lib/python3.11/site-packages/cfgrib/messages.py", line 274, in itervalues
    yield self.filestream.message_from_file(file, errors=errors)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/global/homes/t/timothys/.conda/envs/ufs2arco/lib/python3.11/site-packages/cfgrib/messages.py", line 341, in message_from_file
    return Message.from_file(file, offset, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/global/homes/t/timothys/.conda/envs/ufs2arco/lib/python3.11/site-packages/cfgrib/messages.py", line 97, in from_file
    codes_id = eccodes.codes_grib_new_from_file(file)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/global/homes/t/timothys/.conda/envs/ufs2arco/lib/python3.11/site-packages/gribapi/gribapi.py", line 409, in grib_new_from_file
    GRIB_CHECK(err)
  File "/global/homes/t/timothys/.conda/envs/ufs2arco/lib/python3.11/site-packages/gribapi/gribapi.py", line 226, in GRIB_CHECK
    errors.raise_grib_error(errid)
  File "/global/homes/t/timothys/.conda/envs/ufs2arco/lib/python3.11/site-packages/gribapi/errors.py", line 381, in raise_grib_error
    raise ERROR_MAP[errid](errid)
gribapi.errors.WrongLengthError: Wrong message length
```

even though I was catching for this error with `try/except`!
So here's how it went down:
1. First pass, got a bunch of warnings with the data that couldn't be read
2. Reran starting at the last batch possible to do one full sweep (most ranks
   redid some data transfer)

This part is TODO:
3. Collected missing data dimensions from the warning messages
    * Note that I had code logic to do this, but since it waits until the end to
      spit out a yaml of missed dates, it never got there...
4. Rerun with patch workflow
5. For ufs2arco, should we keep track of and report missing dates at the end, or just check for NaNs at the end?
