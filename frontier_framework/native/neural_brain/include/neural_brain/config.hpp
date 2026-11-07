#pragma once

#include "types.hpp"

#include <cstddef>
#include <cstdint>

namespace neural_brain {

enum class AggregationMode { weighted_mean, attention };

struct CellConfig {
    std::size_t state_size{8};
    Activation activation{Activation::tanh};
    double initial_energy{1.0};
    double energy_cost{0.02};
    double recovery_rate{0.01};
    double synapse_decay{0.001};
    double hebbian_rate{0.05};
    double repair_rate{0.02};
    double damage_rate{0.001};
    double senescence_age{10000.0};
    double mutation_rate{0.01};
    double termination_energy{0.02};
    std::uint64_t random_seed{42};
};

struct GraphConfig {
    std::size_t embedding_size{8};
    std::size_t message_passing_steps{2};
    double pruning_threshold{0.01};
    double learning_rate{0.05};
    double synapse_decay{0.001};
    std::uint64_t random_seed{42};
    AggregationMode aggregation{AggregationMode::weighted_mean};
};

struct MemoryConfig {
    std::size_t working_capacity{64};
    std::size_t episodic_capacity{4096};
    std::size_t semantic_capacity{2048};
    double decay_rate{0.001};
    double consolidation_threshold{0.5};
    double recency_scale{1000.0};
};

struct ModelConfig {
    std::size_t tree_depth{4};
    std::size_t forest_size{32};
    std::size_t boosting_rounds{32};
    std::size_t mlp_hidden_size{16};
    std::size_t mlp_epochs{100};
    double learning_rate{0.03};
    std::size_t min_samples_leaf{2};
    std::uint64_t random_seed{42};
};

struct LearningConfig {
    TaskType task{TaskType::regression};
    double minimum_confidence{0.0};
    double feedback_rate{0.1};
    double latency_cost_scale_microseconds{1000.0};
    std::size_t minimum_training_examples{4};
    std::size_t max_training_examples{10000};
};

struct BrainConfig {
    CellConfig cell;
    GraphConfig graph;
    MemoryConfig memory;
    ModelConfig models;
    LearningConfig learning;
};

}  // namespace neural_brain
