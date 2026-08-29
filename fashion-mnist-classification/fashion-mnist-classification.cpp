#include "fashion_mnist_dataset.hpp"
#include "fashion_mnist_idx.hpp"
#include "fashion_mnist_metrics.hpp"
#include "fashion_mnist_split.hpp"
#include "fashion_mnist_svd.hpp"

#include <nablanet/nablanet.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::uint32_t kSplitSeed = 20'260'828;
constexpr std::uint32_t kNetworkSeed = 20'260'829;
constexpr std::uint32_t kShuffleSeed = 20'260'830;

constexpr std::size_t kValidationExamplesPerClass = 500;
constexpr std::size_t kSvdRank = 64;
constexpr std::size_t kHiddenWidthOne = 256;
constexpr std::size_t kHiddenWidthTwo = 128;
constexpr std::size_t kClassCount = 10;

constexpr std::size_t kMiniBatchSize = 128;
constexpr std::size_t kTrainingEpochs = 8;
constexpr double kLearningRate = 0.001;
constexpr double kL2Coefficient = 1e-4;

fs::path parse_data_directory(const int argc, char* argv[])
{
    if (argc == 1) {
        return fs::path{ NABLANET_FASHION_DEFAULT_DATA_DIR };
    }

    if (argc == 2) {
        return fs::path{ argv[1] };
    }

    throw std::runtime_error(
        "Usage: nablanet_fashion_mnist_classification [data-directory]"
    );
}

double elapsed_seconds(const std::chrono::steady_clock::time_point started)
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        const fs::path data_directory = parse_data_directory(argc, argv);

        const auto data_loading_started = std::chrono::steady_clock::now();
        const fashion_mnist::Split training_data =
            fashion_mnist::load_training_data(data_directory);
        const fashion_mnist::Split test_data =
            fashion_mnist::load_test_data(data_directory);
        const fashion_mnist::IndexSplit training_validation_split =
            fashion_mnist::make_stratified_split(
                training_data,
                kSplitSeed,
                kValidationExamplesPerClass
            );
        fashion_mnist::validate_stratified_split(
            training_data,
            training_validation_split,
            kValidationExamplesPerClass
        );
        const double data_loading_seconds = elapsed_seconds(data_loading_started);

        const auto preprocessing_started = std::chrono::steady_clock::now();

        const fashion_mnist::NormalizedImages training_images =
            fashion_mnist::normalize_images(
                training_data,
                training_validation_split.training_indices
            );
        const fashion_mnist::NormalizedImages validation_images =
            fashion_mnist::normalize_images(
                training_data,
                training_validation_split.validation_indices
            );

        // Fit the preprocessing transform on training pixels only. The
        // validation images are centered and projected with this frozen basis.
        const fashion_mnist::SvdFeatureBasis svd_basis =
            fashion_mnist::fit_svd_basis(training_images, kSvdRank);
        const fashion_mnist::ProjectedFeatures training_features =
            fashion_mnist::project_images(training_images, svd_basis);
        const fashion_mnist::ProjectedFeatures validation_features =
            fashion_mnist::project_images(validation_images, svd_basis);

        const fashion_mnist::OneHotTargets training_targets =
            fashion_mnist::make_one_hot_targets(
                training_data,
                training_validation_split.training_indices
            );
        const fashion_mnist::OneHotTargets validation_targets =
            fashion_mnist::make_one_hot_targets(
                training_data,
                training_validation_split.validation_indices
            );

        const nablanet::Dataset training_dataset =
            fashion_mnist::make_classification_dataset(
                training_features,
                training_targets
            );
        const nablanet::Dataset validation_dataset =
            fashion_mnist::make_classification_dataset(
                validation_features,
                validation_targets
            );
        const double preprocessing_seconds = elapsed_seconds(
            preprocessing_started
        );

        // One activation is declared for each dense layer. The final layer
        // remains linear because the objective consumes logits directly.
        nablanet::NetworkSpec network_spec;
        network_spec.layer_sizes = {
            kSvdRank, kHiddenWidthOne, kHiddenWidthTwo, kClassCount
        };
        network_spec.seed = kNetworkSeed;
        network_spec.layer_activations = {
            nablanet::Activations::ReLU,
            nablanet::Activations::GELU,
            nablanet::Activations::Linear
        };
        network_spec.initialization_type =
            nablanet::InitializationType::KaimingHe;

        nablanet::MLP network = nablanet::make_mlp(network_spec);

        nablanet::AdamOptions adam_options;
        adam_options.learning_rate_schedule = [](std::size_t) {
            return kLearningRate;
        };

        nablanet::TrainingConfig training_config;
        training_config.batch_mode = nablanet::BatchMode::MiniBatch;
        training_config.batch_size = kMiniBatchSize;
        training_config.max_epochs = kTrainingEpochs;
        training_config.shuffle = true;
        training_config.shuffle_seed = kShuffleSeed;
        training_config.record_history = false;
        training_config.parameter_change_tolerance = 10e-8;
        training_config.gradient_tolerance = 10e-8;
        training_config.max_epochs = kTrainingEpochs;

        const nablanet::ObjectiveConfig objective =
            nablanet::make_objective_config(
                nablanet::make_softmax_cross_entropy_objective(),
                nablanet::make_l2_regularization(kL2Coefficient)
            );

        std::cout << "Training Fashion-MNIST classifier..." << std::endl;
        const auto training_started = std::chrono::steady_clock::now();
        const nablanet::TrainingReport training_report = nablanet::train(
            network,
            training_dataset,
            nablanet::make_adam(adam_options),
            training_config,
            objective
        );
        const double training_seconds = elapsed_seconds(training_started);

        const double training_reconstruction_rmse =
            fashion_mnist::reconstruction_root_mean_square_error(
                training_images,
                svd_basis
            );
        const double validation_reconstruction_rmse =
            fashion_mnist::reconstruction_root_mean_square_error(
                validation_images,
                svd_basis
            );
        const double validation_loss = nablanet::objective_loss(
            network,
            validation_dataset,
            objective
        );
        const fashion_mnist::ClassificationMetrics validation_metrics =
            fashion_mnist::evaluate_classifier(network, validation_dataset);

        // The configuration above is now frozen. The official test split is
        // transformed and evaluated only in this final section.
        const auto final_evaluation_started = std::chrono::steady_clock::now();
        const std::vector<std::size_t> test_indices =
            fashion_mnist::make_all_image_indices(test_data);
        const fashion_mnist::NormalizedImages test_images =
            fashion_mnist::normalize_images(test_data, test_indices);
        const fashion_mnist::ProjectedFeatures test_features =
            fashion_mnist::project_images(test_images, svd_basis);
        const fashion_mnist::OneHotTargets test_targets =
            fashion_mnist::make_one_hot_targets(test_data, test_indices);
        const nablanet::Dataset test_dataset =
            fashion_mnist::make_classification_dataset(
                test_features,
                test_targets
            );
        const double test_reconstruction_rmse =
            fashion_mnist::reconstruction_root_mean_square_error(
                test_images,
                svd_basis
            );
        const double test_loss = nablanet::objective_loss(
            network,
            test_dataset,
            objective
        );
        const fashion_mnist::ClassificationMetrics test_metrics =
            fashion_mnist::evaluate_classifier(network, test_dataset);
        const double final_evaluation_seconds = elapsed_seconds(
            final_evaluation_started
        );

        const std::size_t model_parameters = nablanet::parameter_count(network);
        const double model_size_kib = static_cast<double>(
            model_parameters * sizeof(double)
        ) / 1024.0;

        std::cout << std::fixed << std::setprecision(4)
                  << "\nFashion-MNIST data lineage\n"
                  << "  source: official IDX gzip files in "
                  << data_directory.string() << '\n'
                  << "  official split: " << training_data.count
                  << " training / " << test_data.count << " test\n"
                  << "  validation split: "
                  << training_validation_split.validation_indices.size()
                  << " stratified examples (" << kValidationExamplesPerClass
                  << " per class), seed " << kSplitSeed << '\n'
                  << "  data loading and split runtime: "
                  << data_loading_seconds << " s\n"
                  << "\nSVD feature extraction\n"
                  << "  rank: " << svd_basis.rank << '\n'
                  << "  retained singular-value energy: "
                  << svd_basis.retained_energy * 100.0 << "%\n"
                  << "  reconstruction RMSE: train "
                  << training_reconstruction_rmse << ", validation "
                  << validation_reconstruction_rmse << ", test "
                  << test_reconstruction_rmse << '\n'
                  << "  preprocessing runtime: " << preprocessing_seconds
                  << " s\n"
                  << "\nNablaNet model\n"
                  << "  architecture: " << kSvdRank << " -> "
                  << kHiddenWidthOne << " (ReLU) -> " << kHiddenWidthTwo
                  << " (GELU) -> " << kClassCount << " (Linear logits)\n"
                  << "  initialization: Kaiming He, seed " << kNetworkSeed
                  << '\n'
                  << "  trainable parameters: " << model_parameters
                  << " (" << model_size_kib << " KiB of double parameters)\n"
                  << "\nTraining\n"
                  << "  optimizer: Adam, learning rate " << kLearningRate
                  << ", mini-batch size " << kMiniBatchSize
                  << ", epochs " << kTrainingEpochs
                  << ", shuffle seed " << kShuffleSeed << '\n'
                  << "  objective: softmax cross-entropy + L2 "
                  << kL2Coefficient << " (weights only)\n"
                  << "  loss: " << training_report.initial_loss << " -> "
                  << training_report.final_loss << '\n'
                  << "  completed epochs / steps: "
                  << training_report.epochs_completed << " / "
                  << training_report.steps << '\n'
                  << "  training runtime: " << training_seconds << " s\n"
                  << "\nValidation (used before the final test evaluation)\n"
                  << "  objective loss: " << validation_loss << '\n'
                  << "  accuracy: " << validation_metrics.accuracy * 100.0
                  << "% (" << validation_metrics.correct_predictions << " / "
                  << validation_metrics.sample_count << ")\n"
                  << "\nOfficial test evaluation (frozen configuration)\n"
                  << "  objective loss: " << test_loss << '\n'
                  << "  accuracy: " << test_metrics.accuracy * 100.0
                  << "% (" << test_metrics.correct_predictions << " / "
                  << test_metrics.sample_count << ")\n"
                  << "  final evaluation runtime: "
                  << final_evaluation_seconds << " s\n"
                  << "  build: Release CMake C++20 target, sizeof(double) = "
                  << sizeof(double) << " bytes\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Fashion-MNIST example failed: " << error.what() << '\n';
        return 1;
    }
}
