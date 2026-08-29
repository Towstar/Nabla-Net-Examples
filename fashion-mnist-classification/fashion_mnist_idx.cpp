#include "fashion_mnist_idx.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <limits>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <zlib.h>

namespace fashion_mnist {
namespace {

constexpr std::uint32_t kImageMagic = 2'051;
constexpr std::uint32_t kLabelMagic = 2'049;
constexpr std::size_t kRows = 28;
constexpr std::size_t kColumns = 28;
constexpr std::size_t kClassCount = 10;
constexpr std::size_t kTrainingExampleCount = 60'000;
constexpr std::size_t kTestExampleCount = 10'000;
constexpr double kPixelScale = 1.0 / 255.0;

class GzipInput {
public:
    explicit GzipInput(std::filesystem::path path)
        : path_(std::move(path)),
          stream_(gzopen(path_.string().c_str(), "rb"))
    {
        if (stream_ == nullptr) {
            throw std::runtime_error(
                "Could not open compressed IDX file: " + path_.string()
            );
        }
    }

    ~GzipInput()
    {
        if (stream_ != nullptr) {
            gzclose(stream_);
        }
    }

    GzipInput(const GzipInput&) = delete;
    GzipInput& operator=(const GzipInput&) = delete;

    [[nodiscard]] gzFile stream() const noexcept
    {
        return stream_;
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
    gzFile stream_{};
};

[[noreturn]] void throw_read_error(const GzipInput& input)
{
    int gzip_error_code = Z_OK;
    const char* gzip_message = gzerror(input.stream(), &gzip_error_code);

    std::ostringstream message;
    message << "Failed while reading " << input.path().string() << ": ";

    if (gzip_error_code == Z_ERRNO) {
        message << std::strerror(errno);
    } else if (gzip_message != nullptr && *gzip_message != '\0') {
        message << gzip_message;
    } else {
        message << "unexpected end of compressed IDX data";
    }

    throw std::runtime_error(message.str());
}

void read_exact(GzipInput& input, std::span<std::uint8_t> destination)
{
    std::size_t offset = 0;

    while (offset < destination.size()) {
        const std::size_t remaining = destination.size() - offset;
        const auto request_size = static_cast<unsigned int>(
            std::min(
                remaining,
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max()
                )
            )
        );

        const int bytes_read = gzread(
            input.stream(),
            destination.data() + offset,
            request_size
        );

        if (bytes_read <= 0) {
            throw_read_error(input);
        }

        offset += static_cast<std::size_t>(bytes_read);
    }
}

std::uint32_t read_big_endian_u32(GzipInput& input)
{
    std::array<std::uint8_t, 4> bytes{};
    read_exact(
        input,
        std::span<std::uint8_t>(bytes.data(), bytes.size())
    );

    return (static_cast<std::uint32_t>(bytes[0]) << 24U)
        | (static_cast<std::uint32_t>(bytes[1]) << 16U)
        | (static_cast<std::uint32_t>(bytes[2]) << 8U)
        | static_cast<std::uint32_t>(bytes[3]);
}

void validate_images(
    GzipInput& input,
    const std::size_t expected_count,
    Split& split
)
{
    const std::uint32_t magic = read_big_endian_u32(input);
    const std::uint32_t count = read_big_endian_u32(input);
    const std::uint32_t rows = read_big_endian_u32(input);
    const std::uint32_t columns = read_big_endian_u32(input);

    if (magic != kImageMagic) {
        throw std::runtime_error(
            input.path().string() + " has image magic number "
            + std::to_string(magic) + "; expected "
            + std::to_string(kImageMagic)
        );
    }

    if (count != expected_count) {
        throw std::runtime_error(
            input.path().string() + " contains " + std::to_string(count)
            + " images; expected " + std::to_string(expected_count)
        );
    }

    if (rows != kRows || columns != kColumns) {
        throw std::runtime_error(
            input.path().string() + " has image dimensions "
            + std::to_string(rows) + "x" + std::to_string(columns)
            + "; expected 28x28"
        );
    }

    split.count = count;
    split.rows = rows;
    split.columns = columns;
    split.pixels.resize(expected_count * kRows * kColumns);
    read_exact(
        input,
        std::span<std::uint8_t>(split.pixels.data(), split.pixels.size())
    );
}

void validate_labels(
    GzipInput& input,
    const std::size_t expected_count,
    Split& split
)
{
    const std::uint32_t magic = read_big_endian_u32(input);
    const std::uint32_t count = read_big_endian_u32(input);

    if (magic != kLabelMagic) {
        throw std::runtime_error(
            input.path().string() + " has label magic number "
            + std::to_string(magic) + "; expected "
            + std::to_string(kLabelMagic)
        );
    }

    if (count != expected_count) {
        throw std::runtime_error(
            input.path().string() + " contains " + std::to_string(count)
            + " labels; expected " + std::to_string(expected_count)
        );
    }

    split.labels.resize(expected_count);
    read_exact(
        input,
        std::span<std::uint8_t>(split.labels.data(), split.labels.size())
    );

    const auto invalid_label = std::find_if(
        split.labels.begin(),
        split.labels.end(),
        [](const std::uint8_t label) {
            return label >= kClassCount;
        }
    );

    if (invalid_label != split.labels.end()) {
        throw std::runtime_error(
            input.path().string() + " contains an out-of-range class label"
        );
    }
}

Split load_split(
    const std::filesystem::path& image_file,
    const std::filesystem::path& label_file,
    const std::size_t expected_count
)
{
    GzipInput images(image_file);
    GzipInput labels(label_file);

    Split split;
    validate_images(images, expected_count, split);
    validate_labels(labels, expected_count, split);
    return split;
}

void validate_indices(
    const Split& split,
    const std::span<const std::size_t> image_indices
)
{
    const std::size_t pixels_per_image = split.rows * split.columns;

    if (split.count == 0 || pixels_per_image == 0
        || split.pixels.size() != split.count * pixels_per_image
        || split.labels.size() != split.count) {
        throw std::invalid_argument("Fashion-MNIST split is not valid");
    }

    const auto invalid_index = std::find_if(
        image_indices.begin(),
        image_indices.end(),
        [&split](const std::size_t image_index) {
            return image_index >= split.count;
        }
    );

    if (invalid_index != image_indices.end()) {
        throw std::out_of_range("Fashion-MNIST image index is out of range");
    }
}

} // namespace

Split load_training_data(const std::filesystem::path& data_directory)
{
    return load_split(
        data_directory / "train-images-idx3-ubyte.gz",
        data_directory / "train-labels-idx1-ubyte.gz",
        kTrainingExampleCount
    );
}

Split load_test_data(const std::filesystem::path& data_directory)
{
    return load_split(
        data_directory / "t10k-images-idx3-ubyte.gz",
        data_directory / "t10k-labels-idx1-ubyte.gz",
        kTestExampleCount
    );
}

std::vector<std::size_t> make_all_image_indices(const Split& split)
{
    std::vector<std::size_t> indices(split.count);
    std::iota(indices.begin(), indices.end(), std::size_t{ 0 });
    validate_indices(split, indices);
    return indices;
}

NormalizedImages normalize_images(
    const Split& split,
    const std::span<const std::size_t> image_indices
)
{
    validate_indices(split, image_indices);

    const std::size_t pixels_per_image = split.rows * split.columns;
    NormalizedImages normalized;
    normalized.count = image_indices.size();
    normalized.features_per_image = pixels_per_image;
    normalized.values.resize(normalized.count * pixels_per_image);

    for (std::size_t destination_row = 0;
         destination_row < image_indices.size();
         ++destination_row) {
        const std::size_t source_offset =
            image_indices[destination_row] * pixels_per_image;
        const std::size_t destination_offset =
            destination_row * pixels_per_image;

        for (std::size_t pixel = 0; pixel < pixels_per_image; ++pixel) {
            normalized.values[destination_offset + pixel] =
                static_cast<double>(split.pixels[source_offset + pixel])
                * kPixelScale;
        }
    }

    return normalized;
}

OneHotTargets make_one_hot_targets(
    const Split& split,
    const std::span<const std::size_t> image_indices
)
{
    validate_indices(split, image_indices);

    OneHotTargets targets;
    targets.count = image_indices.size();
    targets.class_count = kClassCount;
    targets.values.assign(targets.count * targets.class_count, 0.0);

    for (std::size_t row = 0; row < image_indices.size(); ++row) {
        const std::size_t label = split.labels[image_indices[row]];
        targets.values[row * targets.class_count + label] = 1.0;
    }

    return targets;
}

} // namespace fashion_mnist
