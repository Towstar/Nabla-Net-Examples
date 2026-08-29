#pragma once

#include "fashion_mnist_idx.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fashion_mnist {

struct IndexSplit {
    std::vector<std::size_t> training_indices;
    std::vector<std::size_t> validation_indices;
};

// Selects an equal number of examples from each class for validation. The
// returned index vectors are shuffled but always reproduce for the same seed.
IndexSplit make_stratified_split(
    const Split& data,
    std::uint32_t seed,
    std::size_t validation_examples_per_class
);

// Ensures that a split covers the source data exactly once and that validation
// contains the requested number of examples from every class.
void validate_stratified_split(
    const Split& data,
    const IndexSplit& split,
    std::size_t validation_examples_per_class
);

} // namespace fashion_mnist
