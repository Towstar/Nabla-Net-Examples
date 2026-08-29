#include "fashion_mnist_svd.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fashion_mnist {
    namespace {

void validate_normalized_images(
    const NormalizedImages& images,
    const std::string_view name
)
{
    if (images.count == 0 || images.features_per_image == 0
        || images.values.size()
            != images.count * images.features_per_image) {
        throw std::invalid_argument(
            std::string(name) + " does not contain a valid image matrix"
        );
    }

    if (!std::all_of(
            images.values.begin(),
            images.values.end(),
            [](const double value) {
                return std::isfinite(value) && value >= 0.0 && value <= 1.0;
            }
        )) {
        throw std::invalid_argument(
            std::string(name) + " contains a non-normalized pixel value"
        );
    }
}

void validate_basis(const SvdFeatureBasis& basis)
{
    if (basis.input_dimensions == 0 || basis.rank == 0
        || basis.mean.size() != basis.input_dimensions
        || basis.components.size()
            != basis.rank * basis.input_dimensions
        || basis.singular_values.size() < basis.rank
        || basis.retained_energy < 0.0
        || basis.retained_energy > 1.0) {
        throw std::invalid_argument("SVD basis is not valid");
    }
}

Eigen::MatrixXf make_float_matrix(const NormalizedImages& images)
{
    Eigen::MatrixXf result(
        static_cast<Eigen::Index>(images.count),
        static_cast<Eigen::Index>(images.features_per_image)
    );

    for (std::size_t row = 0; row < images.count; ++row) {
        for (std::size_t column = 0;
             column < images.features_per_image;
             ++column) {
            result(
                static_cast<Eigen::Index>(row),
                static_cast<Eigen::Index>(column)
            ) = static_cast<float>(
                images.values[row * images.features_per_image + column]
            );
        }
    }

    return result;
}

Eigen::MatrixXf make_centered_matrix(
    const NormalizedImages& images,
    const std::vector<float>& mean
)
{
    Eigen::MatrixXf centered = make_float_matrix(images);

    Eigen::Map<const Eigen::RowVectorXf> mean_vector(
        mean.data(),
        static_cast<Eigen::Index>(mean.size())
    );
    centered.rowwise() -= mean_vector;

    return centered;
}

Eigen::Map<const Eigen::Matrix<
    float,
    Eigen::Dynamic,
    Eigen::Dynamic,
    Eigen::RowMajor
>> component_matrix(const SvdFeatureBasis& basis)
{
    return {
        basis.components.data(),
        static_cast<Eigen::Index>(basis.rank),
        static_cast<Eigen::Index>(basis.input_dimensions)
    };
}

} // namespace

    SvdFeatureBasis fit_svd_basis(
        const NormalizedImages& training_images,
        const std::size_t rank
    )
    {
        validate_normalized_images(training_images, "training images");

        const std::size_t maximum_rank = std::min(
            training_images.count,
            training_images.features_per_image
        );
        if (rank == 0 || rank > maximum_rank) {
            throw std::invalid_argument("Requested SVD rank is out of range");
        }

        Eigen::MatrixXf centered = make_float_matrix(training_images);
        const Eigen::RowVectorXf mean = centered.colwise().mean();
        centered.rowwise() -= mean;

        const Eigen::BDCSVD<Eigen::MatrixXf> svd(
            centered,
            Eigen::ComputeThinV
        );
        if (svd.info() != Eigen::Success) {
            throw std::runtime_error("SVD decomposition did not converge");
        }

        const Eigen::VectorXf singular_values = svd.singularValues();

        double total_energy = 0.0;
        double retained_energy = 0.0;
        for (Eigen::Index index = 0; index < singular_values.size(); ++index) {
            const double energy =
                static_cast<double>(singular_values(index))
                * static_cast<double>(singular_values(index));

            total_energy += energy;
            if (index < static_cast<Eigen::Index>(rank)) {
                retained_energy += energy;
            }
        }

        if (total_energy == 0.0) {
            throw std::runtime_error(
                "Training images contain no variance for SVD"
            );
        }

        SvdFeatureBasis basis;
        basis.input_dimensions = training_images.features_per_image;
        basis.rank = rank;
        basis.mean.assign(mean.data(), mean.data() + mean.size());
        basis.singular_values.assign(
            singular_values.data(),
            singular_values.data() + singular_values.size()
        );
        basis.retained_energy = retained_energy / total_energy;
        basis.components.resize(rank * basis.input_dimensions);

        Eigen::Map<Eigen::Matrix<
            float,
            Eigen::Dynamic,
            Eigen::Dynamic,
            Eigen::RowMajor
        >> components(
            basis.components.data(),
            static_cast<Eigen::Index>(rank),
            static_cast<Eigen::Index>(basis.input_dimensions)
        );

        // V's columns are principal directions; storing Vᵀ gives one component
        // per contiguous row and makes projection/reconstruction direct.
        components = svd.matrixV().leftCols(
            static_cast<Eigen::Index>(rank)
        ).transpose();

        return basis;
    }

    ProjectedFeatures project_images(
        const NormalizedImages& images,
        const SvdFeatureBasis& basis
    )
    {
        validate_normalized_images(images, "images to project");
        validate_basis(basis);

        if (images.features_per_image != basis.input_dimensions) {
            throw std::invalid_argument(
                "Image dimensions do not match the SVD basis"
            );
        }

        const Eigen::MatrixXf centered = make_centered_matrix(
            images,
            basis.mean
        );
        const Eigen::MatrixXf projected =
            centered * component_matrix(basis).transpose();

        ProjectedFeatures result;
        result.sample_count = images.count;
        result.feature_count = basis.rank;
        result.values.resize(result.sample_count * result.feature_count);

        for (std::size_t row = 0; row < result.sample_count; ++row) {
            for (std::size_t column = 0;
                column < result.feature_count;
                ++column) {
                result.values[row * result.feature_count + column] =
                    static_cast<double>(
                        projected(
                            static_cast<Eigen::Index>(row),
                            static_cast<Eigen::Index>(column)
                        )
                    );
            }
        }

        return result;
    }

    double reconstruction_root_mean_square_error(
        const NormalizedImages& images,
        const SvdFeatureBasis& basis
    )
{
    validate_normalized_images(images, "images to reconstruct");
    validate_basis(basis);

    if (images.features_per_image != basis.input_dimensions) {
        throw std::invalid_argument(
            "Image dimensions do not match the SVD basis"
        );
    }

    const Eigen::MatrixXf centered = make_centered_matrix(
        images,
        basis.mean
    );
    const Eigen::MatrixXf projected =
        centered * component_matrix(basis).transpose();
    const Eigen::MatrixXf reconstruction =
        projected * component_matrix(basis);

    double squared_error = 0.0;
    for (Eigen::Index row = 0; row < centered.rows(); ++row) {
        for (Eigen::Index column = 0; column < centered.cols(); ++column) {
            const double error = static_cast<double>(
                centered(row, column) - reconstruction(row, column)
            );
            squared_error += error * error;
        }
    }

    return std::sqrt(
        squared_error
        / static_cast<double>(centered.rows() * centered.cols())
    );
}

}
