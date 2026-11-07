#include "simulated_environment.h"

#include <math.h>
#include <limits.h>
#include <stddef.h>

static double clamp_unit(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

FrontierSimulationStatus frontier_simulation_validate(
    const FrontierSimulationState *state
) {
    if (state == NULL) {
        return FRONTIER_SIMULATION_INVALID_ARGUMENT;
    }

    if (state->node_count < 0 || state->total_agents < 0 ||
        state->memory_alive < 0 || state->knowledge_total < 0 ||
        state->cycle_count < 0) {
        return FRONTIER_SIMULATION_INVALID_STATE;
    }

    if (!isfinite(state->avg_fitness) || state->avg_fitness < 0.0 ||
        state->avg_fitness > 1.0 || !isfinite(state->emergence_score) ||
        state->emergence_score < 0.0 || !isfinite(state->uptime_s) ||
        state->uptime_s < 0.0 || !isfinite(state->risk_level) ||
        state->risk_level < 0.0 || state->risk_level > 1.0) {
        return FRONTIER_SIMULATION_INVALID_STATE;
    }

    return FRONTIER_SIMULATION_OK;
}

double frontier_environment_score(const FrontierSimulationState *state) {
    if (frontier_simulation_validate(state) != FRONTIER_SIMULATION_OK) {
        return 0.0;
    }

    const double agent_pressure =
        (double)state->total_agents / ((double)state->total_agents + 30.0);
    const double memory_factor =
        (double)state->memory_alive / ((double)state->memory_alive + 40.0);
    const double knowledge_factor =
        (double)state->knowledge_total / ((double)state->knowledge_total + 30.0);

    return clamp_unit(
        state->avg_fitness * 0.35 +
        (1.0 - state->risk_level) * 0.25 +
        agent_pressure * 0.15 +
        memory_factor * 0.10 +
        knowledge_factor * 0.05 +
        clamp_unit(state->emergence_score) * 0.10
    );
}

FrontierSimulationStatus frontier_simulation_step(
    FrontierSimulationState *state,
    double elapsed_seconds,
    FrontierSimulationStepResult *result
) {
    FrontierSimulationStatus status = frontier_simulation_validate(state);
    if (status != FRONTIER_SIMULATION_OK) {
        return status;
    }

    if (!isfinite(elapsed_seconds) || elapsed_seconds <= 0.0 ||
        !isfinite(state->uptime_s + elapsed_seconds)) {
        return FRONTIER_SIMULATION_INVALID_ARGUMENT;
    }
    if (state->cycle_count == INT_MAX) {
        return FRONTIER_SIMULATION_NUMERIC_ERROR;
    }

    const int agents_before = state->total_agents;
    FrontierSimulationState next_state = *state;
    const double agent_delta =
        (double)state->total_agents *
        (state->avg_fitness - state->risk_level) *
        0.01 * elapsed_seconds;
    const double rounded_delta = round(agent_delta);

    if (!isfinite(rounded_delta) ||
        rounded_delta > (double)(INT_MAX - state->total_agents) ||
        rounded_delta < -(double)state->total_agents) {
        return FRONTIER_SIMULATION_NUMERIC_ERROR;
    }

    next_state.total_agents += (int)rounded_delta;
    next_state.avg_fitness = clamp_unit(
        next_state.avg_fitness +
        (0.7 - state->risk_level - state->avg_fitness) *
        0.01 * elapsed_seconds
    );
    next_state.risk_level = clamp_unit(
        next_state.risk_level +
        (0.5 - next_state.avg_fitness) * 0.01 * elapsed_seconds
    );
    next_state.emergence_score = clamp_unit(
        next_state.avg_fitness * (double)next_state.node_count * 0.1
    );
    next_state.memory_alive = (int)round(
        (double)next_state.memory_alive * exp(-0.001 * elapsed_seconds)
    );
    next_state.uptime_s += elapsed_seconds;
    next_state.cycle_count += 1;

    if (frontier_simulation_validate(&next_state) != FRONTIER_SIMULATION_OK) {
        return FRONTIER_SIMULATION_NUMERIC_ERROR;
    }
    *state = next_state;

    if (result != NULL) {
        result->cycle = next_state.cycle_count;
        result->agents_before = agents_before;
        result->agents_after = next_state.total_agents;
        result->score = frontier_environment_score(&next_state);
    }

    return FRONTIER_SIMULATION_OK;
}
