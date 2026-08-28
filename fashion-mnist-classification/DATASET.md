# Fashion-MNIST dataset

This project contains the four original compressed IDX files from the official
[Zalando Research Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist), downloaded on 2026-08-28.

The files remain compressed and preserve the official split: 60,000 training
images and labels, plus 10,000 test images and labels. The data is stored in
`data/fashion/` and totals 30,878,645 bytes (about 29.4 MiB).

| File | Contents | Official MD5 |
| --- | --- | --- |
| `train-images-idx3-ubyte.gz` | 60,000 training images | `8d4fb7e6c68d591d4c3dfef9ec88bf0d` |
| `train-labels-idx1-ubyte.gz` | 60,000 training labels | `25c81989df183df01b3e8a0aad5dffbe` |
| `t10k-images-idx3-ubyte.gz` | 10,000 test images | `bef4ecab320f06d8554ea6380940ec79` |
| `t10k-labels-idx1-ubyte.gz` | 10,000 test labels | `bb300cfdad3c16e7a12a480ee83cd310` |

All four checksums were verified after download. Fashion-MNIST is released
under the MIT License; retain the upstream attribution and license when
redistributing this data.
