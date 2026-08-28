# UCI Bank Marketing dataset

This project contains the official [UCI Bank Marketing archive](https://archive.ics.uci.edu/dataset/222/bank+marketing), downloaded on 2026-08-28.

The outer archive and its `bank-additional` bundle are retained unmodified:

| File | SHA-256 |
| --- | --- |
| `data/bank-marketing.zip` | `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4` |
| `data/bank-additional.zip` | `a607b5edab6c6c75ce09c39142a77702c38123bd5aa7ae89a63503bbe17d65cd` |

The planned campaign-prioritization experiment should read the original,
chronologically ordered full data at:

```text
data/bank-additional/bank-additional/bank-additional-full.csv
```

It contains 41,188 records. No rows have been reordered, re-split, encoded, or
otherwise transformed; `duration` remains present in the raw data and must be
removed only in the later leakage-safe modeling pipeline.

UCI lists this dataset under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). Preserve the UCI attribution when redistributing it.
