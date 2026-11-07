#ifndef FRONTIER_SIMULATED_ENVIRONMENT_HPP
#define FRONTIER_SIMULATED_ENVIRONMENT_HPP

#include "simulated_environment.h"

namespace frontier {

class SimulatedEnvironment {
public:
    explicit SimulatedEnvironment(const FrontierSimulationState &initial_state);

    const FrontierSimulationState &state() const noexcept;
    double score() const noexcept;
    FrontierSimulationStepResult step(double elapsed_seconds);

private:
    FrontierSimulationState state_;
};

}  // namespace frontier

#endif
