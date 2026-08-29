#include "fashion_mnist_dataset.hpp"

#include <stdexcept>

namespace fashion_mnist {

nablanet::Dataset make_classification_dataset(
    const ProjectedFeatures& features,
    const OneHotTargets& targets
) {
    if (features.sample_count == 0 || features.feature_count == 0
        || targets.count == 0 || targets.class_count == 0
        || features.sample_count != targets.count
        || features.values.size()
            != features.sample_count * features.feature_count
        || targets.values.size()
            != targets.count * targets.class_count) {
        throw std::invalid_argument(
            "Features and one-hot targets cannot form a valid dataset"
        );
    }

    nablanet::Dataset dataset;
    dataset.reserve(features.sample_count);

    for (std::size_t sample = 0; sample < features.sample_count; ++sample) {
        const auto feature_begin =
            features.values.begin() + sample * features.feature_count;
        const auto target_begin =
            targets.values.begin() + sample * targets.class_count;

        dataset.push_back({
            nablanet::Values(
                feature_begin,
                feature_begin + features.feature_count
            ),
            nablanet::Values(
                target_begin,
                target_begin + targets.class_count
            )
        });
    }

    return dataset;
}

}
