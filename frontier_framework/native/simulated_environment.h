#ifndef FRONTIER_SIMULATED_ENVIRONMENT_H
#define FRONTIER_SIMULATED_ENVIRONMENT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum FrontierSimulationStatus {
    FRONTIER_SIMULATION_OK = 0,
    FRONTIER_SIMULATION_INVALID_ARGUMENT = 1,
    FRONTIER_SIMULATION_INVALID_STATE = 2,
    FRONTIER_SIMULATION_NUMERIC_ERROR = 3
} FrontierSimulationStatus;

typedef struct FrontierSimulationState {
    int node_count;
    int total_agents;
    double avg_fitness;
    double emergence_score;
    int memory_alive;
    double uptime_s;
    int knowledge_total;
    double risk_level;
    int cycle_count;
} FrontierSimulationState;

typedef struct FrontierSimulationStepResult {
    int cycle;
    int agents_before;
    int agents_after;
    double score;
} FrontierSimulationStepResult;

FrontierSimulationStatus frontier_simulation_validate(
    const FrontierSimulationState *state
);

FrontierSimulationStatus frontier_simulation_step(
    FrontierSimulationState *state,
    double elapsed_seconds,
    FrontierSimulationStepResult *result
);

double frontier_environment_score(const FrontierSimulationState *state);

#ifdef __cplusplus
}
#endif

#endif
