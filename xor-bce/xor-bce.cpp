#include <iomanip>
#include <iostream>

#include <nablanet/nablanet.hpp>

int main()
{
    const nablanet::Dataset xor_data{
        { { 0.0, 0.0 }, { 0.0 } },
        { { 0.0, 1.0 }, { 1.0 } },
        { { 1.0, 0.0 }, { 1.0 } },
        { { 1.0, 1.0 }, { 0.0 } }
    };

    nablanet::MLP network = nablanet::make_mlp({ 2, 4, 1 }, 11);

    nablanet::AdamOptions adam_options;
    adam_options.learning_rate_schedule = [](std::size_t) {
        return 0.03;
    };

    nablanet::TrainingConfig training;
    training.max_epochs = 1'500;
    training.record_history = false;

    const nablanet::TrainingReport report = nablanet::train(
        network,
        xor_data,
        nablanet::make_adam(adam_options),
        training,
        nablanet::make_objective_config(
            nablanet::make_binary_cross_entropy_objective()
        )
    );

    std::cout << std::fixed << std::setprecision(3)
              << "XOR training loss: " << report.initial_loss
              << " -> " << report.final_loss << "\n";
    for (const nablanet::Sample& sample : xor_data) {
        const nablanet::ForwardCache cache =
            nablanet::forward_pass(network, sample.input);
        std::cout << sample.input[0] << " xor " << sample.input[1]
                  << " = " << cache.activations.back().front() << "\n";
    }

    return 0;
}
