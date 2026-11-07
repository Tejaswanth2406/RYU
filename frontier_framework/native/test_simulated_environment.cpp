#include "simulated_environment.h"
#include "simulated_environment.hpp"
#include "neural_brain/learning_controller.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

int main() {
    FrontierSimulationState state = {
        4, 50, 0.7, 0.28, 100, 0.0, 20, 0.2, 0
    };

    assert(frontier_simulation_validate(&state) == FRONTIER_SIMULATION_OK);
    const double initial_score = frontier_environment_score(&state);
    assert(initial_score > 0.0 && initial_score < 1.0);

    FrontierSimulationStepResult step_result = {};
    assert(frontier_simulation_step(&state, 10.0, &step_result) ==
           FRONTIER_SIMULATION_OK);
    assert(step_result.cycle == 1);
    assert(step_result.agents_before == 50);
    assert(step_result.agents_after == 53);
    assert(state.uptime_s == 10.0);
    assert(state.avg_fitness < 0.7);
    assert(state.memory_alive < 100);

    FrontierSimulationState invalid = state;
    invalid.risk_level = std::numeric_limits<double>::quiet_NaN();
    assert(frontier_simulation_validate(&invalid) ==
           FRONTIER_SIMULATION_INVALID_STATE);
    assert(frontier_simulation_step(&state, 0.0, &step_result) ==
           FRONTIER_SIMULATION_INVALID_ARGUMENT);

    frontier::SimulatedEnvironment environment(state);
    const FrontierSimulationStepResult cpp_result = environment.step(2.0);
    assert(cpp_result.cycle == state.cycle_count + 1);
    assert(std::abs(environment.score() - cpp_result.score) < 1e-12);

    bool rejected_invalid_state = false;
    try {
        frontier::SimulatedEnvironment invalid_environment(invalid);
    } catch (const std::invalid_argument &) {
        rejected_invalid_state = true;
    }
    assert(rejected_invalid_state);

    neural_brain::BrainConfig brain_config;
    brain_config.graph.embedding_size = 4;
    brain_config.graph.message_passing_steps = 2;
    brain_config.models.forest_size = 4;
    brain_config.models.boosting_rounds = 3;
    brain_config.models.mlp_hidden_size = 4;
    brain_config.models.mlp_epochs = 5;
    brain_config.learning.minimum_training_examples = 4;
    neural_brain::LearningController brain(brain_config);
    brain.ensemble().add_model(std::make_shared<neural_brain::XGBoostAdapter>(
        [](const std::vector<neural_brain::TrainingExample>&) {},
        [](const neural_brain::Vector&) { return 0.4; }
    ));
    brain.add_cell(1, "planner");
    brain.add_cell(2, "researcher");
    auto divided = brain.graph().cell(1).divide(3);
    assert(divided.id() == 3);
    assert(std::abs(divided.energy() - brain.graph().cell(1).energy()) < 1e-12);
    brain.graph().add_cell(std::move(divided));
    brain.connect(1, 2, 0.7);
    brain.connect(2, 1, 0.4);
    brain.graph().reinforce(1, 2, 0.8, 0.7);
    brain.connect(1, 3, 0.005);
    assert(brain.graph().prune() == 1);
    brain.graph().cell(2).accumulate_damage(0.5);
    brain.graph().cell(2).repair(1.0);
    assert(brain.graph().cell(2).damage() < 0.5);
    for (int i = 0; i < 4; ++i) {
        const double feature = static_cast<double>(i) / 3.0;
        brain.learn({feature, 1.0 - feature}, feature * 2.0, "native test");
    }
    const auto brain_result = brain.infer({0.25, 0.75});
    assert(std::isfinite(brain_result.prediction.value));
    assert(brain_result.prediction.confidence >= 0.0);
    assert(brain_result.prediction.confidence <= 1.0);
    assert(brain_result.graph_embedding.size() == 4);
    assert(brain_result.retrieved_memories > 0);
    assert(brain.memory().episodic_size() > 0);
    assert(brain.ensemble().model_names().size() == 5);
    neural_brain::Experience semantic;
    semantic.features = brain_result.encoded_features;
    semantic.outcome = {1.0};
    semantic.importance = 0.9;
    brain.remember(std::move(semantic), neural_brain::MemoryKind::semantic);
    assert(brain.memory().semantic_size() == 1);
    brain.feedback({0.25, 0.75}, 0.5);
    return 0;
}
