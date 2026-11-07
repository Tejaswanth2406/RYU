#include "simulated_environment.h"

#include "simulated_environment.hpp"

#include <stdexcept>

namespace frontier {

SimulatedEnvironment::SimulatedEnvironment(
    const FrontierSimulationState &initial_state
) : state_(initial_state) {
    if (frontier_simulation_validate(&state_) != FRONTIER_SIMULATION_OK) {
        throw std::invalid_argument("invalid initial simulation state");
    }
}

const FrontierSimulationState &SimulatedEnvironment::state() const noexcept {
    return state_;
}

double SimulatedEnvironment::score() const noexcept {
    return frontier_environment_score(&state_);
}

FrontierSimulationStepResult SimulatedEnvironment::step(
    double elapsed_seconds
) {
    FrontierSimulationStepResult result = {};
    const FrontierSimulationStatus status =
        frontier_simulation_step(&state_, elapsed_seconds, &result);
    if (status == FRONTIER_SIMULATION_INVALID_ARGUMENT) {
        throw std::invalid_argument("simulation step rejected invalid input or state");
    }
    if (status == FRONTIER_SIMULATION_INVALID_STATE) {
        throw std::runtime_error("simulation state became invalid");
    }
    if (status == FRONTIER_SIMULATION_NUMERIC_ERROR) {
        throw std::overflow_error("simulation step exceeded numeric limits");
    }
    return result;
}

}  // namespace frontier
