#include "fashion_mnist_metrics.hpp"

#include <nablanet/objective_functions.hpp>

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fashion_mnist {
namespace {

std::size_t argmax(
    const nablanet::Values& values,
    const std::string_view description
)
{
    if (values.empty()) {
        throw std::invalid_argument(
            std::string(description) + " must not be empty"
        );
    }

    return static_cast<std::size_t>(std::distance(
        values.begin(),
        std::max_element(values.begin(), values.end())
    ));
}

} // namespace

ClassificationMetrics evaluate_classifier(
    const nablanet::MLP& network,
    const nablanet::Dataset& dataset
)
{
    if (dataset.empty()) {
        throw std::invalid_argument("Cannot evaluate an empty dataset");
    }

    ClassificationMetrics metrics;
    metrics.sample_count = dataset.size();

    for (const nablanet::Sample& sample : dataset) {
        const nablanet::ForwardCache cache =
            nablanet::forward_pass(network, sample.input);
        const nablanet::Values probabilities = nablanet::stable_softmax(
            cache.pre_activations.back()
        );

        if (probabilities.size() != sample.target.size()) {
            throw std::invalid_argument(
                "Network output size does not match the target size"
            );
        }

        if (argmax(probabilities, "Network probabilities")
            == argmax(sample.target, "One-hot target")) {
            ++metrics.correct_predictions;
        }
    }

    metrics.accuracy = static_cast<double>(metrics.correct_predictions)
        / static_cast<double>(metrics.sample_count);
    return metrics;
}

} // namespace fashion_mnist
