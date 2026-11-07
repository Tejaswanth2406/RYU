#include "simulated_environment.h"
#include "simulated_environment.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

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
    assert(cpp_result.cycle == 1);
    assert(std::abs(environment.score() - cpp_result.score) < 1e-12);

    bool rejected_invalid_state = false;
    try {
        frontier::SimulatedEnvironment invalid_environment(invalid);
    } catch (const std::invalid_argument &) {
        rejected_invalid_state = true;
    }
    assert(rejected_invalid_state);
    return 0;
}
