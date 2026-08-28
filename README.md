# NablaNet examples

This directory contains the finalized public example layout for the sibling
NablaNet library. Each project owns its source and `CMakeLists.txt`; the
shared helper automatically includes the adjacent `../lib` checkout when the
library target is not already available.

The current `.cpp` files are temporary scaffolding while the experiments are
implemented. Their directory names, targets, datasets, and documentation
contracts below are the source of truth for the final examples.

| Directory | CMake target | Final experiment |
| --- | --- | --- |
| `xor-bce/` | `nablanet_xor_bce` | Deterministic XOR binary classification with BCE |
| `margin-loss-robustness/` | `nablanet_margin_loss_robustness` | Label-corruption comparison of BCE, exponential, and hinge loss |
| `automotive-time-series-classification/` | `nablanet_forda` | FordA raw versus FFT feature classification with L-BFGS/Wolfe diagnostics |
| `bank-marketing/` | `nablanet_bank_marketing` | Leakage-safe chronological campaign prioritization |
| `fashion-mnist-classification/` | `nablanet_fashion_mnist_classification` | Fashion-MNIST rank-k SVD features with softmax classification |

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release --parallel
.\build\xor-bce\Release\nablanet_xor_bce.exe
```

For a single-configuration generator, add
`-DCMAKE_BUILD_TYPE=Release` to the configure command and run the executable
from `build/`.

To build FordA and Bank Marketing as well, enable the data-backed projects:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 `
  -DNABLANET_EXAMPLES_BUILD_DATASET_EXAMPLES=ON
cmake --build build --config Release --parallel
```

Fashion-MNIST remains separate because it requires OpenCV 4 for its rank-k SVD
feature extraction. Install OpenCV through your normal package manager, then
configure with:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 `
  -DNABLANET_EXAMPLES_BUILD_FASHION_MNIST=ON
cmake --build build --config Release --target nablanet_fashion_mnist_classification
```
