#include <iomanip>
#include <iostream>

#include <nablanet/nablanet.hpp>

int main()
{
    // Learn y = 0.75x - 0.25 from a small, deterministic data set.
    const nablanet::Dataset data{
        { { -2.0 }, { -1.75 } },
        { { -1.0 }, { -1.00 } },
        { { 0.0 }, { -0.25 } },
        { { 1.0 }, { 0.50 } },
        { { 2.0 }, { 1.25 } }
    };

    nablanet::MLP network = nablanet::make_mlp({ 1, 8, 1 }, 23);
    network.layer_activations = {
        nablanet::Activations::Tanh,
        nablanet::Activations::Linear
    };

    nablanet::AdamOptions adam_options;
    adam_options.learning_rate_schedule = [](std::size_t) {
        return 0.02;
    };

    nablanet::TrainingConfig training;
    training.batch_mode = nablanet::BatchMode::MiniBatch;
    training.batch_size = 2;
    training.max_epochs = 800;
    training.shuffle = true;
    training.shuffle_seed = 5;
    training.record_history = false;

    const nablanet::TrainingReport report = nablanet::train(
        network,
        data,
        nablanet::make_adam(adam_options),
        training,
        nablanet::make_objective_config(
            nablanet::make_mean_squared_error_objective()
        )
    );

    std::cout << std::fixed << std::setprecision(3)
              << "Regression loss: " << report.initial_loss
              << " -> " << report.final_loss << "\n";
    for (const double input : { -1.5, 0.5, 1.5 }) {
        const nablanet::ForwardCache cache =
            nablanet::forward_pass(network, { input });
        std::cout << "x = " << input
                  << ", prediction = " << cache.activations.back().front()
                  << "\n";
    }

    return 0;
}
