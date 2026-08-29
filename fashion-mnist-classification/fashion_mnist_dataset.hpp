#pragma once

#include "fashion_mnist_idx.hpp"
#include "fashion_mnist_svd.hpp"

#include <nablanet/mlp.hpp>

namespace fashion_mnist {

// Combines contiguous SVD features and one-hot labels into NablaNet samples.
nablanet::Dataset make_classification_dataset(
    const ProjectedFeatures& features,
    const OneHotTargets& targets
);

} // namespace fashion_mnist
