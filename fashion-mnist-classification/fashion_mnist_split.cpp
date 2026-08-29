#include "fashion_mnist_split.hpp"

#include <algorithm>
#include <array>
#include <random>
#include <stdexcept>

namespace fashion_mnist {
namespace {

constexpr std::size_t kClassCount = 10;

void validate_source_labels(const Split& data)
{
    if (data.labels.size() != data.count) {
        throw std::invalid_argument(
            "Fashion-MNIST split has a mismatched label count"
        );
    }

    const auto invalid_label = std::find_if(
        data.labels.begin(),
        data.labels.end(),
        [](const std::uint8_t label) {
            return label >= kClassCount;
        }
    );

    if (invalid_label != data.labels.end()) {
        throw std::invalid_argument(
            "Fashion-MNIST split contains an out-of-range class label"
        );
    }
}

} // namespace

IndexSplit make_stratified_split(
    const Split& data,
    const std::uint32_t seed,
    const std::size_t validation_examples_per_class
)
{
    validate_source_labels(data);

    std::array<std::vector<std::size_t>, kClassCount> indices_by_class;
    for (std::size_t index = 0; index < data.count; ++index) {
        indices_by_class[data.labels[index]].push_back(index);
    }

    std::mt19937 generator(seed);
    IndexSplit split;
    split.validation_indices.reserve(
        kClassCount * validation_examples_per_class
    );

    for (std::vector<std::size_t>& class_indices : indices_by_class) {
        std::shuffle(class_indices.begin(), class_indices.end(), generator);

        if (class_indices.size() < validation_examples_per_class) {
            throw std::invalid_argument(
                "A Fashion-MNIST class has too few examples for validation"
            );
        }

        split.validation_indices.insert(
            split.validation_indices.end(),
            class_indices.begin(),
            class_indices.begin() + validation_examples_per_class
        );
        split.training_indices.insert(
            split.training_indices.end(),
            class_indices.begin() + validation_examples_per_class,
            class_indices.end()
        );
    }

    std::shuffle(
        split.training_indices.begin(),
        split.training_indices.end(),
        generator
    );
    std::shuffle(
        split.validation_indices.begin(),
        split.validation_indices.end(),
        generator
    );
    return split;
}

void validate_stratified_split(
    const Split& data,
    const IndexSplit& split,
    const std::size_t validation_examples_per_class
)
{
    validate_source_labels(data);

    const std::size_t expected_validation_count =
        kClassCount * validation_examples_per_class;
    if (split.validation_indices.size() != expected_validation_count
        || split.training_indices.size()
            != data.count - expected_validation_count) {
        throw std::logic_error("Fashion-MNIST split has unexpected sizes");
    }

    std::vector<bool> seen(data.count, false);
    const auto mark_indices = [&seen, &data](
                                  const std::vector<std::size_t>& indices) {
        for (const std::size_t index : indices) {
            if (index >= data.count || seen[index]) {
                throw std::logic_error(
                    "Fashion-MNIST split has an out-of-range or duplicate index"
                );
            }

            seen[index] = true;
        }
    };

    mark_indices(split.training_indices);
    mark_indices(split.validation_indices);

    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        throw std::logic_error(
            "Fashion-MNIST split does not cover every training image"
        );
    }

    std::array<std::size_t, kClassCount> validation_counts{};
    for (const std::size_t index : split.validation_indices) {
        ++validation_counts[data.labels[index]];
    }

    if (std::any_of(
            validation_counts.begin(),
            validation_counts.end(),
            [validation_examples_per_class](const std::size_t count) {
                return count != validation_examples_per_class;
            }
        )) {
        throw std::logic_error(
            "Fashion-MNIST validation set is not stratified by class"
        );
    }
}

} // namespace fashion_mnist
