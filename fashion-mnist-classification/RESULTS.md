# Fashion-MNIST SVD classifier run notes

This is a library-usage example, not a hyperparameter-search framework. It
uses the one fixed configuration declared in
`fashion-mnist-classification.cpp`; the validation partition is reported before
the official test partition is transformed and evaluated.

## Data lineage and split policy

- Raw inputs are the four original gzip-compressed IDX files documented in
  `DATASET.md`. No processed data or results are written under `data/`.
- The official 60,000/10,000 train/test split is preserved.
- A seeded stratified holdout selects 500 images from each training class for a
  5,000-example validation partition. The remaining 55,000 images train the
  model.
- The SVD mean and rank-k component basis are fit only on those 55,000
  training images. Validation and test images use that frozen basis.

## Fixed run configuration

| Setting | Value |
| --- | --- |
| Split seed | `20260828` |
| SVD rank | `64` |
| Network seed | `20260829` |
| Architecture | `64 -> 256 ReLU -> 128 GELU -> 10 Linear logits` |
| Initialization | Kaiming He |
| Optimizer | Adam, learning rate `0.001` |
| Training | 8 epochs, mini-batches of 128, shuffle seed `20260830` |
| Objective | Softmax cross-entropy plus L2 `0.0001` on weights only |

The executable reports retained singular-value energy, reconstruction RMSE,
losses, validation/test accuracy, runtimes, parameter count, and parameter
storage size. Record the output from the exact machine used for a result;
runtime is inherently hardware- and build-dependent.

## Verified local run

The fixed configuration above was run on 2026-08-28 as a Release x64 CMake
C++20 target using the local Visual Studio build environment. The complete
program output is retained under the generated `build/` directory during local
development; only this compact, reproducible summary is committed.

| Metric | Result |
| --- | ---: |
| Retained SVD energy | 88.1287% |
| Reconstruction RMSE (train / validation / test) | 0.1017 / 0.1014 / 0.1018 |
| Model parameters / parameter storage | 50,826 / 397.0781 KiB |
| Training objective loss | 3.3974 -> 0.2993 |
| Training runtime | 153.2803 s |
| Validation objective loss / accuracy | 0.3559 / 89.6400% (4,482 / 5,000) |
| Official-test objective loss / accuracy | 0.4002 / 88.1800% (8,818 / 10,000) |
| Final test evaluation runtime | 1.0334 s |

## Build and run

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release --target nablanet_fashion_mnist_classification
.\build\Release\nablanet_fashion_mnist_classification.exe
```

From the aggregate `examples/` directory, add
`-DNABLANET_EXAMPLES_BUILD_FASHION_MNIST=ON` while configuring.
