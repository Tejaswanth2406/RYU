#include "neural_brain/learning_controller.hpp"

#include "neural_brain/types.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace neural_brain {

LearningController::LearningController(BrainConfig config)
    : config_(config),
      graph_(config_.graph),
      gnn_(config_.graph),
      memory_(config_.memory),
      ensemble_(config_.learning.latency_cost_scale_microseconds) {
    if (!std::isfinite(config_.learning.minimum_confidence) ||
        config_.learning.minimum_confidence < 0.0 ||
        config_.learning.minimum_confidence > 1.0 ||
        !std::isfinite(config_.learning.feedback_rate) ||
        config_.learning.feedback_rate < 0.0 ||
        config_.learning.feedback_rate > 1.0 ||
        !std::isfinite(config_.learning.latency_cost_scale_microseconds) ||
        config_.learning.latency_cost_scale_microseconds <= 0.0) {
        throw std::invalid_argument("invalid learning controller configuration");
    }
    ensemble_.add_model(std::make_shared<DecisionTreeRegressor>(config_.models));
    ensemble_.add_model(std::make_shared<RandomForestRegressor>(config_.models));
    ensemble_.add_model(std::make_shared<GradientBoostingRegressor>(config_.models));
    ensemble_.add_model(std::make_shared<MLPRegressor>(config_.models));
    if (config_.learning.minimum_training_examples == 0 ||
        config_.learning.max_training_examples <
            config_.learning.minimum_training_examples) {
        throw std::invalid_argument("invalid learning dataset limits");
    }
}

BrainGraph& LearningController::graph() noexcept { return graph_; }
MemoryLayer& LearningController::memory() noexcept { return memory_; }
Ensemble& LearningController::ensemble() noexcept { return ensemble_; }
const Ensemble& LearningController::ensemble() const noexcept { return ensemble_; }

NeuralCell& LearningController::add_cell(CellId id, const std::string& specialization) {
    return graph_.add_cell(id, config_.cell, specialization);
}

void LearningController::connect(CellId source, CellId target, double weight) {
    graph_.connect(source, target, weight);
}

void LearningController::remember(Experience experience, MemoryKind kind) {
    switch (kind) {
        case MemoryKind::working:
            memory_.store_working(std::move(experience));
            break;
        case MemoryKind::episodic:
            memory_.store_episode(std::move(experience));
            break;
        case MemoryKind::semantic:
            memory_.store_semantic(std::move(experience));
            break;
    }
}

void LearningController::activate_cells(const Vector& features, double elapsed_time) {
    for (const auto& [id, cell] : graph_.cells()) {
        (void)cell;
        Vector input(config_.cell.state_size, 0.0);
        for (std::size_t i = 0; i < features.size(); ++i) {
            input[i % input.size()] += features[i];
        }
        graph_.cell(id).activate(input, elapsed_time);
    }
}

Vector LearningController::model_features(const Vector& features,
                                          const Vector& embedding) {
    return encoder_.combine(features, embedding);
}

void LearningController::learn(const Vector& features, double target,
                               std::string context) {
    if (features.empty() || !is_finite(features) || !std::isfinite(target)) {
        throw std::invalid_argument("learning sample must be finite and non-empty");
    }
    activate_cells(features, 1.0);
    const Vector embedding = gnn_.forward(graph_).graph_embedding;
    encoder_samples_.push_back(features);
    embedding_history_.push_back(embedding);
    target_history_.push_back(target);
    if (encoder_samples_.size() > config_.learning.max_training_examples) {
        encoder_samples_.erase(encoder_samples_.begin());
        embedding_history_.erase(embedding_history_.begin());
        target_history_.erase(target_history_.begin());
    }
    encoder_.fit(encoder_samples_);

    training_.clear();
    training_.reserve(encoder_samples_.size());
    for (std::size_t i = 0; i < encoder_samples_.size(); ++i) {
        training_.push_back({
            model_features(encoder_samples_[i], embedding_history_[i]),
            target_history_[i]
        });
    }

    Experience experience;
    experience.features = training_.back().features;
    experience.outcome = {target};
    experience.context = std::move(context);
    experience.importance = 0.7;
    memory_.store_working(experience);
    memory_.consolidate();

    if (training_.size() >= config_.learning.minimum_training_examples) {
        ensemble_.fit(training_);
    }
    ++cycle_;
}

LearningCycleResult LearningController::infer(const Vector& features,
                                              double elapsed_time) {
    if (features.empty() || !is_finite(features) ||
        !std::isfinite(elapsed_time) || elapsed_time <= 0.0) {
        throw std::invalid_argument("inference input is invalid");
    }
    if (!encoder_.fitted() || !ensemble_.trained()) {
        throw std::logic_error("train the learning controller before inference");
    }
    activate_cells(features, elapsed_time);
    const Vector embedding = gnn_.forward(graph_).graph_embedding;
    const Vector encoded = model_features(features, embedding);
    const auto memories = memory_.retrieve(encoded, 5);
    const Prediction prediction = ensemble_.predict(encoded);
    Prediction output = prediction;
    if (config_.learning.task == TaskType::classification) {
        const double probability = 1.0 /
            (1.0 + std::exp(-std::clamp(prediction.value, -60.0, 60.0)));
        output.value = probability >= 0.5 ? 1.0 : 0.0;
        output.confidence = std::max(probability, 1.0 - probability);
    }
    if (output.confidence < config_.learning.minimum_confidence) {
        throw std::runtime_error("prediction confidence is below configured minimum");
    }
    graph_.tick(elapsed_time);
    memory_.decay(elapsed_time);
    ++cycle_;
    return {output, encoded, embedding, memories.size(), cycle_};
}

void LearningController::feedback(const Vector& features, double observed) {
    if (!encoder_.fitted() || !is_finite(features) || !std::isfinite(observed)) {
        throw std::invalid_argument("feedback features or target are invalid");
    }
    const Vector embedding = gnn_.forward(graph_).graph_embedding;
    const Vector encoded = model_features(features, embedding);
    ensemble_.feedback(encoded, observed, config_.learning.feedback_rate);
    Experience experience;
    experience.features = encoded;
    experience.outcome = {observed};
    experience.importance = 0.9;
    memory_.store_episode(std::move(experience));
}

}  // namespace neural_brain
