#include <algorithm>
#include <iomanip>
#include <iostream>

#include <nablanet/nablanet.hpp>

int main()
{
    // Three simple, linearly separated regions represented with one-hot targets.
    const nablanet::Dataset data{
        { { -1.0, -1.0 }, { 1.0, 0.0, 0.0 } },
        { { -1.2, -0.7 }, { 1.0, 0.0, 0.0 } },
        { { 1.0, -1.0 }, { 0.0, 1.0, 0.0 } },
        { { 1.2, -0.7 }, { 0.0, 1.0, 0.0 } },
        { { 0.0, 1.0 }, { 0.0, 0.0, 1.0 } },
        { { 0.2, 1.3 }, { 0.0, 0.0, 1.0 } }
    };

    nablanet::MLP network = nablanet::make_mlp({ 2, 6, 3 }, 31);
    network.layer_activations = {
        nablanet::Activations::Tanh,
        nablanet::Activations::Linear
    };

    nablanet::AdamOptions adam_options;
    adam_options.learning_rate_schedule = [](std::size_t) {
        return 0.04;
    };

    nablanet::TrainingConfig training;
    training.max_epochs = 1'000;
    training.record_history = false;

    const nablanet::TrainingReport report = nablanet::train(
        network,
        data,
        nablanet::make_adam(adam_options),
        training,
        nablanet::make_objective_config(
            nablanet::make_softmax_cross_entropy_objective()
        )
    );

    std::cout << std::fixed << std::setprecision(3)
              << "Softmax loss: " << report.initial_loss
              << " -> " << report.final_loss << "\n";
    for (const nablanet::Sample& sample : data) {
        const nablanet::ForwardCache cache =
            nablanet::forward_pass(network, sample.input);
        const nablanet::Values probabilities =
            nablanet::stable_softmax(cache.pre_activations.back());
        const std::size_t predicted_class = static_cast<std::size_t>(
            std::distance(
                probabilities.begin(),
                std::max_element(probabilities.begin(), probabilities.end())
            )
        );
        std::cout << "(" << sample.input[0] << ", " << sample.input[1]
                  << ") -> class " << predicted_class
                  << " with probability " << probabilities[predicted_class]
                  << "\n";
    }

    return 0;
}
