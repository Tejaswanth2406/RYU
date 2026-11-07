#include "neural_brain/ensemble.hpp"

#include "neural_brain/types.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace neural_brain {

Ensemble::Ensemble(double latency_cost_scale_microseconds)
    : latency_cost_scale_microseconds_(latency_cost_scale_microseconds) {
    if (!std::isfinite(latency_cost_scale_microseconds_) ||
        latency_cost_scale_microseconds_ <= 0.0) {
        throw std::invalid_argument("ensemble latency cost scale must be positive");
    }
}

void Ensemble::add_model(std::shared_ptr<Regressor> model) {
    if (!model) {
        throw std::invalid_argument("ensemble model cannot be null");
    }
    for (const auto& expert : experts_) {
        if (expert.model->name() == model->name()) {
            throw std::invalid_argument("ensemble model names must be unique");
        }
    }
    experts_.push_back({std::move(model), 1.0, 0.0, 0, 0.0, 0});
}

void Ensemble::fit(const std::vector<TrainingExample>& samples) {
    if (experts_.empty()) {
        throw std::logic_error("ensemble has no registered models");
    }
    for (auto& expert : experts_) {
        expert.model->fit(samples);
    }
}

Prediction Ensemble::predict(const Vector& features) const {
    double weighted_total = 0.0;
    double total_weight = 0.0;
    std::vector<double> predictions;
    std::vector<double> effective_weights;
    for (const auto& expert : experts_) {
        if (!expert.model->trained() || expert.weight <= 0.0) {
            continue;
        }
        const auto started = std::chrono::steady_clock::now();
        const double value = expert.model->predict(features);
        const auto finished = std::chrono::steady_clock::now();
        if (!std::isfinite(value)) {
            throw std::runtime_error("ensemble expert returned a non-finite result");
        }
        const double elapsed = std::chrono::duration<double, std::micro>(
            finished - started
        ).count();
        expert.average_latency_microseconds =
            expert.prediction_count == 0 ? elapsed :
            0.8 * expert.average_latency_microseconds + 0.2 * elapsed;
        ++expert.prediction_count;
        const double effective_weight = expert.weight /
            (1.0 + expert.average_latency_microseconds /
                       latency_cost_scale_microseconds_);
        predictions.push_back(value);
        effective_weights.push_back(effective_weight);
        weighted_total += effective_weight * value;
        total_weight += effective_weight;
    }
    if (predictions.empty() || total_weight <= 0.0) {
        throw std::logic_error("ensemble has no trained experts");
    }
    const double mean = weighted_total / total_weight;
    double weighted_variance = 0.0;
    for (std::size_t i = 0; i < predictions.size(); ++i) {
        const double delta = predictions[i] - mean;
        weighted_variance += effective_weights[i] * delta * delta;
    }
    weighted_variance /= total_weight;
    const double confidence = 1.0 / (1.0 + std::sqrt(weighted_variance));
    std::string best_model;
    double best_weight = -1.0;
    std::size_t prediction_index = 0;
    for (const auto& expert : experts_) {
        if (!expert.model->trained() || expert.weight <= 0.0) {
            continue;
        }
        const double effective_weight = effective_weights[prediction_index++];
        if (effective_weight > best_weight) {
            best_weight = effective_weight;
            best_model = expert.model->name();
        }
    }
    return {mean, clamp(confidence, 0.0, 1.0), best_model};
}

void Ensemble::feedback(const Vector& features, double observed, double rate) {
    if (!is_finite(features) || !std::isfinite(observed) ||
        !std::isfinite(rate) || rate < 0.0 || rate > 1.0) {
        throw std::invalid_argument("invalid ensemble feedback");
    }
    for (auto& expert : experts_) {
        if (!expert.model->trained()) {
            continue;
        }
        const double error = std::abs(expert.model->predict(features) - observed);
        expert.absolute_error += error;
        ++expert.feedback_count;
        expert.weight *= std::exp(-rate * error);
        expert.weight = clamp(expert.weight, 1e-6, 1e6);
    }
}

std::vector<std::string> Ensemble::model_names() const {
    std::vector<std::string> result;
    result.reserve(experts_.size());
    for (const auto& expert : experts_) {
        result.push_back(expert.model->name());
    }
    return result;
}

bool Ensemble::trained() const noexcept {
    for (const auto& expert : experts_) {
        if (expert.model->trained()) {
            return true;
        }
    }
    return false;
}

}  // namespace neural_brain
