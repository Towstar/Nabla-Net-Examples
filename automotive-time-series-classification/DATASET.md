# FordA dataset

This project contains the official [FordA archive](https://www.timeseriesclassification.com/description.php?Dataset=FordA), downloaded on 2026-08-28.

`data/FordA.zip` is retained as downloaded. Its SHA-256 checksum is:

```text
4d78a72e5165e8da40c51115a8219c9d01978aafac6334ed8dfabd29da359ffb
```

The archive was extracted without changing its contents. The raw UCR-style
files for the planned FFT experiment are:

| File | Rows | Layout |
| --- | ---: | --- |
| `data/FordA_TRAIN.txt` | 3,601 | label followed by 500 engine-noise samples |
| `data/FordA_TEST.txt` | 1,320 | label followed by 500 engine-noise samples |

The original `.ts` and `.arff` representations are also retained. No rows have
been reordered, re-split, standardized, or otherwise transformed, preserving
FordA's predefined training/test division.

The source describes FordA as an automotive-subsystem diagnosis dataset. It
does not state a dataset license on the linked description page; retain source
attribution and confirm its terms before redistribution.
