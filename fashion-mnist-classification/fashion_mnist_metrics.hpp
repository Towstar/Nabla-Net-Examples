#pragma once

#include <nablanet/mlp.hpp>

#include <cstddef>

namespace fashion_mnist {

struct ClassificationMetrics {
    std::size_t sample_count{};
    std::size_t correct_predictions{};
    double accuracy{};
};

// Applies the network to one-hot labelled examples. Predictions use softmax
// over the final linear logits, matching the softmax cross-entropy objective.
ClassificationMetrics evaluate_classifier(
    const nablanet::MLP& network,
    const nablanet::Dataset& dataset
);

} // namespace fashion_mnist
