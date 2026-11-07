#pragma once

#include "ensemble.hpp"
#include "feature_encoder.hpp"
#include "gnn_layer.hpp"
#include "memory_layer.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace neural_brain {

struct LearningCycleResult {
    Prediction prediction;
    Vector encoded_features;
    Vector graph_embedding;
    std::size_t retrieved_memories{0};
    std::size_t cycle{0};
};

class LearningController {
public:
    explicit LearningController(BrainConfig config = {});

    BrainGraph& graph() noexcept;
    MemoryLayer& memory() noexcept;
    Ensemble& ensemble() noexcept;
    const Ensemble& ensemble() const noexcept;

    NeuralCell& add_cell(CellId id, const std::string& specialization = "general");
    void connect(CellId source, CellId target, double weight);
    void remember(Experience experience, MemoryKind kind = MemoryKind::episodic);
    void learn(const Vector& features, double target, std::string context = {});
    LearningCycleResult infer(const Vector& features, double elapsed_time = 1.0);
    void feedback(const Vector& features, double observed);

private:
    Vector model_features(const Vector& features, const Vector& embedding);
    void activate_cells(const Vector& features, double elapsed_time);

    BrainConfig config_;
    BrainGraph graph_;
    GNNLayer gnn_;
    MemoryLayer memory_;
    FeatureEncoder encoder_;
    Ensemble ensemble_;
    std::vector<TrainingExample> training_;
    std::vector<Vector> encoder_samples_;
    std::vector<Vector> embedding_history_;
    std::vector<double> target_history_;
    std::size_t cycle_{0};
};

}  // namespace neural_brain
