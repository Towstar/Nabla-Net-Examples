#pragma once

#include "fashion_mnist_idx.hpp"

#include <cstddef>
#include <vector>

namespace fashion_mnist {

// A contiguous row-major matrix of rank-k SVD features. Values are stored as
// doubles so they can be used directly as NablaNet inputs later.
struct ProjectedFeatures {
    std::size_t sample_count{};
    std::size_t feature_count{};
    std::vector<double> values;
};

// The learned, training-only rank-k basis. Components are laid out one
// component at a time: components[component * input_dimensions + pixel].
struct SvdFeatureBasis {
    std::size_t input_dimensions{};
    std::size_t rank{};
    std::vector<float> mean;
    std::vector<float> components;
    std::vector<float> singular_values;
    double retained_energy{};
};

// Fits a centered truncated SVD basis to normalized training images. The
// caller must ensure the images came only from the training partition.
SvdFeatureBasis fit_svd_basis(
    const NormalizedImages& training_images,
    std::size_t rank
);

// Centers images with the training mean and projects them into the learned
// rank-k feature space. This function never refits the basis.
ProjectedFeatures project_images(
    const NormalizedImages& images,
    const SvdFeatureBasis& basis
);

// Measures pixel-space RMSE after projection and reconstruction through the
// learned basis. It is useful for reporting compression quality by rank.
double reconstruction_root_mean_square_error(
    const NormalizedImages& images,
    const SvdFeatureBasis& basis
);

} // namespace fashion_mnist
