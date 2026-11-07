#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace neural_brain {

using CellId = std::uint64_t;
using Vector = std::vector<double>;

enum class MemoryKind { working, episodic, semantic };
enum class TaskType { regression, classification };
enum class Activation { identity, tanh, sigmoid, relu };

// 7 canonical input feature indices (aligned with Python data contract)
enum class FeatureIndex : std::size_t {
    avg_fitness = 0,
    active_ratio = 1,
    avg_energy = 2,
    avg_damage = 3,
    graph_density = 4,
    risk_level = 5,
    emergence_score = 6,
    COUNT = 7
};

inline constexpr std::array<const char*, 7> FEATURE_NAMES = {
    "avg_fitness",
    "active_ratio",
    "avg_energy",
    "avg_damage",
    "graph_density",
    "risk_level",
    "emergence_score"
};

struct Experience {
    Vector features;
    Vector outcome;
    std::string context;
    double importance{0.5};
    double timestamp{0.0};
    std::uint64_t retrievals{0};
};

struct DualTarget {
    double avg_fitness{0.0};      // Continuous regression target [0.0, 1.0]
    std::int32_t is_collapse{0};  // Binary classification target {0, 1}
};

struct SnapshotMetadata {
    std::string run_id;
    std::uint64_t cycle{0};
    std::uint32_t horizon{1};
};

struct TrainingExample {
    Vector features;              // 7 normalized features
    double target{0.0};           // Backward-compatible primary target (avg_fitness)
    DualTarget targets;           // Explicit dual targets (regression + classification)
    SnapshotMetadata metadata;    // Run ID, cycle, horizon
};

struct Prediction {
    double value{0.0};
    double confidence{0.0};
    std::string model;
};

struct RetrievedMemory {
    Experience experience;
    MemoryKind kind{MemoryKind::working};
    double similarity{0.0};
    double retention_score{0.0};
};

double clamp(double value, double lower, double upper);
double cosine_similarity(const Vector& lhs, const Vector& rhs);
bool is_finite(const Vector& values);

}  // namespace neural_brain
