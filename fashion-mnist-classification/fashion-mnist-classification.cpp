#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace {

int parse_rank(const char* text)
{
    std::size_t consumed{};
    const int rank = std::stoi(text, &consumed);
    if (consumed != std::string(text).size() || rank <= 0) {
        throw std::invalid_argument("rank must be a positive integer");
    }
    return rank;
}

void write_csv(const cv::Mat& feature_vectors, const std::string& output_path)
{
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("could not open output file: " + output_path);
    }

    output << std::setprecision(17);
    for (int row = 0; row < feature_vectors.rows; ++row) {
        for (int column = 0; column < feature_vectors.cols; ++column) {
            if (column != 0) {
                output << ',';
            }
            output << feature_vectors.at<double>(row, column);
        }
        output << '\n';
    }
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        if (argc < 2 || argc > 4) {
            std::cerr << "Usage: nablanet_image_svd_features <image> [rank] [output.csv]\n";
            return 2;
        }

        const cv::Mat grayscale = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        if (grayscale.empty()) {
            throw std::runtime_error("could not read image: " + std::string(argv[1]));
        }

        cv::Mat pixels;
        grayscale.convertTo(pixels, CV_64F, 1.0 / 255.0);

        const int maximum_rank = std::min(pixels.rows, pixels.cols);
        const int rank = argc >= 3
            ? parse_rank(argv[2])
            : std::min(32, maximum_rank);
        if (rank > maximum_rank) {
            throw std::invalid_argument(
                "rank must not exceed min(image height, image width)"
            );
        }

        cv::Mat singular_values;
        cv::Mat left_vectors;
        cv::Mat right_vectors_transposed;
        cv::SVD::compute(
            pixels,
            singular_values,
            left_vectors,
            right_vectors_transposed
        );

        // A_k = U_k Sigma_k V_k^T is the Eckart-Young best rank-k
        // approximation of this image matrix. U_k Sigma_k gives one
        // rank-k feature vector for every image row.
        const cv::Mat sigma = cv::Mat::diag(singular_values.rowRange(0, rank));
        const cv::Mat feature_vectors =
            left_vectors.colRange(0, rank) * sigma;
        const cv::Mat approximation =
            feature_vectors * right_vectors_transposed.rowRange(0, rank);

        double total_energy{};
        double retained_energy{};
        for (int index = 0; index < singular_values.rows; ++index) {
            const double value = singular_values.at<double>(index);
            total_energy += value * value;
            if (index < rank) {
                retained_energy += value * value;
            }
        }

        const std::string output_path = argc == 4
            ? argv[3]
            : "image_svd_row_features.csv";
        write_csv(feature_vectors, output_path);

        const double retained_percent = total_energy > 0.0
            ? 100.0 * retained_energy / total_energy
            : 100.0;

        std::cout << std::fixed << std::setprecision(6)
                  << "Image: " << pixels.cols << " x " << pixels.rows << "\n"
                  << "Rank: " << rank << "\n"
                  << "Retained energy: " << retained_percent << "%\n"
                  << "Rank-k reconstruction error (Frobenius): "
                  << cv::norm(pixels, approximation, cv::NORM_L2) << "\n";
        std::cout << "Singular-value descriptor:";
        for (int index = 0; index < rank; ++index) {
            std::cout << ' ' << singular_values.at<double>(index);
        }
        std::cout << "\nWrote " << feature_vectors.rows << " feature vectors of length "
                  << feature_vectors.cols << " to " << output_path << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 1;
    }
}
