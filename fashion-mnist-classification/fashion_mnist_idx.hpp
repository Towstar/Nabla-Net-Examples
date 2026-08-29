#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace fashion_mnist {

// Raw image bytes and labels from one official Fashion-MNIST IDX split.
// Keeping the bytes compact allows callers to create only the normalized rows
// required for a train, validation, or test partition.
struct Split {
    std::size_t count{};
    std::size_t rows{};
    std::size_t columns{};
    std::vector<std::uint8_t> pixels;
    std::vector<std::uint8_t> labels;
};

// A contiguous row-major feature matrix. Every value is normalized to [0, 1].
struct NormalizedImages {
    std::size_t count{};
    std::size_t features_per_image{};
    std::vector<double> values;
};

// A contiguous row-major matrix of one-hot Fashion-MNIST class targets.
struct OneHotTargets {
    std::size_t count{};
    std::size_t class_count{};
    std::vector<double> values;
};

// Opens the original gzip-compressed IDX files and validates their headers,
// payload lengths, image dimensions, and labels.
Split load_training_data(const std::filesystem::path& data_directory);
Split load_test_data(const std::filesystem::path& data_directory);

// Returns every valid row index in source order. This keeps full-split test
// preprocessing out of the example's main program.
std::vector<std::size_t> make_all_image_indices(const Split& split);

// Creates normalized feature rows only for the selected image indices. This
// keeps normalization and data layout out of the experiment's main program
// without allocating a second copy of every raw image.
NormalizedImages normalize_images(
    const Split& split,
    std::span<const std::size_t> image_indices
);

// Produces matching one-hot class targets for the selected image indices.
OneHotTargets make_one_hot_targets(
    const Split& split,
    std::span<const std::size_t> image_indices
);

} // namespace fashion_mnist
