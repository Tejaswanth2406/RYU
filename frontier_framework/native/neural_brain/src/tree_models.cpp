#include "neural_brain/tree_models.hpp"

#include "neural_brain/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace neural_brain {
namespace {

std::size_t validate_samples(const std::vector<TrainingExample>& samples) {
    if (samples.empty() || samples.front().features.empty()) {
        throw std::invalid_argument("model training requires non-empty samples");
    }
    const std::size_t dimensions = samples.front().features.size();
    for (const auto& sample : samples) {
        if (sample.features.size() != dimensions || !is_finite(sample.features) ||
            !std::isfinite(sample.target)) {
            throw std::invalid_argument("training examples are invalid or inconsistent");
        }
        for (const double value : sample.features) {
            if (std::abs(value) > 1e6) {
                throw std::invalid_argument("training features must be scaled to a safe range");
            }
        }
        if (std::abs(sample.target) > 1e6) {
            throw std::invalid_argument("training targets exceed the supported numeric range");
        }
    }
    return dimensions;
}

double average_target(const std::vector<TrainingExample>& samples,
                      const std::vector<std::size_t>& indices) {
    double sum = 0.0;
    for (const auto index : indices) {
        sum += samples[index].target;
    }
    return sum / static_cast<double>(indices.size());
}

}  // namespace

DecisionTreeRegressor::DecisionTreeRegressor(ModelConfig config) : config_(config) {
    if (config_.tree_depth == 0 || config_.min_samples_leaf == 0) {
        throw std::invalid_argument("tree depth and leaf size must be positive");
    }
}

std::string DecisionTreeRegressor::name() const { return "decision_tree"; }

void DecisionTreeRegressor::fit(const std::vector<TrainingExample>& samples) {
    feature_count_ = validate_samples(samples);
    nodes_.clear();
    std::vector<std::size_t> indices(samples.size());
    std::iota(indices.begin(), indices.end(), 0);
    build(samples, indices, 0);
    if (nodes_.empty() || feature_count_ == 0) {
        throw std::runtime_error("decision tree failed to build");
    }
}

bool DecisionTreeRegressor::trained() const noexcept { return !nodes_.empty(); }

double DecisionTreeRegressor::predict(const Vector& features) const {
    if (!trained() || features.size() != feature_count_ || !is_finite(features)) {
        throw std::logic_error("decision tree is untrained or input is invalid");
    }
    return predict_node(0, features);
}

int DecisionTreeRegressor::build(const std::vector<TrainingExample>& samples,
                                 const std::vector<std::size_t>& indices,
                                 std::size_t depth) {
    const int node_index = static_cast<int>(nodes_.size());
    nodes_.push_back({0, 0.0, average_target(samples, indices), -1, -1, true});
    if (depth >= config_.tree_depth ||
        indices.size() < 2 * config_.min_samples_leaf) {
        return node_index;
    }

    const std::size_t dimensions = samples.front().features.size();
    double best_loss = std::numeric_limits<double>::infinity();
    std::size_t best_feature = 0;
    double best_threshold = 0.0;
    std::vector<std::size_t> best_left;
    std::vector<std::size_t> best_right;
    double total_sum = 0.0;
    double total_square_sum = 0.0;
    for (const auto index : indices) {
        const double target = samples[index].target;
        total_sum += target;
        total_square_sum += target * target;
    }
    if (!std::isfinite(total_sum) || !std::isfinite(total_square_sum)) {
        throw std::invalid_argument("tree target statistics exceeded numeric range");
    }
    const double parent_loss = std::max(
        0.0, total_square_sum -
            total_sum * total_sum / static_cast<double>(indices.size())
    );

    for (std::size_t feature = 0; feature < dimensions; ++feature) {
        auto ordered = indices;
        std::sort(ordered.begin(), ordered.end(), [&](std::size_t lhs, std::size_t rhs) {
            return samples[lhs].features[feature] < samples[rhs].features[feature];
        });
        double left_sum = 0.0;
        double left_square_sum = 0.0;
        for (std::size_t split = 1; split < ordered.size(); ++split) {
            const double added_target = samples[ordered[split - 1]].target;
            left_sum += added_target;
            left_square_sum += added_target * added_target;
            if (split < config_.min_samples_leaf ||
                ordered.size() - split < config_.min_samples_leaf) {
                continue;
            }
            const double lower = samples[ordered[split - 1]].features[feature];
            const double upper = samples[ordered[split]].features[feature];
            if (lower == upper) {
                continue;
            }
            const double threshold = lower * 0.5 + upper * 0.5;
            const std::size_t right_count = ordered.size() - split;
            const double right_sum = total_sum - left_sum;
            const double right_square_sum = total_square_sum - left_square_sum;
            const double loss = std::max(
                0.0, left_square_sum - left_sum * left_sum /
                         static_cast<double>(split)
            ) + std::max(
                0.0, right_square_sum - right_sum * right_sum /
                         static_cast<double>(right_count)
            );
            if (loss < best_loss) {
                best_loss = loss;
                best_feature = feature;
                best_threshold = threshold;
                best_left.assign(ordered.begin(), ordered.begin() + split);
                best_right.assign(ordered.begin() + split, ordered.end());
            }
        }
    }

    if (best_left.empty() || best_right.empty() ||
        best_loss >= parent_loss - 1e-12) {
        return node_index;
    }
    const int left_index = build(samples, best_left, depth + 1);
    const int right_index = build(samples, best_right, depth + 1);
    nodes_[node_index] = {
        best_feature, best_threshold, nodes_[node_index].value,
        left_index, right_index, false
    };
    return node_index;
}

double DecisionTreeRegressor::predict_node(int node, const Vector& features) const {
    if (static_cast<std::size_t>(node) >= nodes_.size()) {
        throw std::runtime_error("decision tree contains an invalid node");
    }
    const Node& current = nodes_[static_cast<std::size_t>(node)];
    if (current.leaf) {
        return current.value;
    }
    if (current.feature >= features.size()) {
        throw std::invalid_argument("prediction feature count differs from training");
    }
    return predict_node(
        features[current.feature] <= current.threshold ? current.left : current.right,
        features
    );
}

RandomForestRegressor::RandomForestRegressor(ModelConfig config) : config_(config) {}
std::string RandomForestRegressor::name() const { return "random_forest"; }

void RandomForestRegressor::fit(const std::vector<TrainingExample>& samples) {
    validate_samples(samples);
    if (config_.forest_size == 0) {
        throw std::invalid_argument("random forest must contain at least one tree");
    }
    std::mt19937_64 random(config_.random_seed);
    std::uniform_int_distribution<std::size_t> choose(0, samples.size() - 1);
    trees_.clear();
    trees_.reserve(config_.forest_size);
    for (std::size_t tree_index = 0; tree_index < config_.forest_size; ++tree_index) {
        std::vector<TrainingExample> bootstrap;
        bootstrap.reserve(samples.size());
        for (std::size_t i = 0; i < samples.size(); ++i) {
            bootstrap.push_back(samples[choose(random)]);
        }
        trees_.emplace_back(config_);
        trees_.back().fit(bootstrap);
    }
}

double RandomForestRegressor::predict(const Vector& features) const {
    if (trees_.empty()) {
        throw std::logic_error("random forest is untrained");
    }
    double total = 0.0;
    for (const auto& tree : trees_) {
        total += tree.predict(features);
    }
    return total / static_cast<double>(trees_.size());
}

bool RandomForestRegressor::trained() const noexcept { return !trees_.empty(); }

GradientBoostingRegressor::GradientBoostingRegressor(ModelConfig config)
    : config_(config) {}
std::string GradientBoostingRegressor::name() const { return "gradient_boosting"; }

void GradientBoostingRegressor::fit(const std::vector<TrainingExample>& samples) {
    validate_samples(samples);
    if (config_.boosting_rounds == 0 || !std::isfinite(config_.learning_rate) ||
        config_.learning_rate <= 0.0) {
        throw std::invalid_argument("invalid gradient boosting configuration");
    }
    baseline_ = 0.0;
    for (const auto& sample : samples) {
        baseline_ += sample.target;
    }
    baseline_ /= static_cast<double>(samples.size());
    std::vector<double> predictions(samples.size(), baseline_);
    trees_.clear();
    trees_.reserve(config_.boosting_rounds);
    ModelConfig stump_config = config_;
    stump_config.tree_depth = 1;
    for (std::size_t round = 0; round < config_.boosting_rounds; ++round) {
        std::vector<TrainingExample> residuals;
        residuals.reserve(samples.size());
        for (std::size_t i = 0; i < samples.size(); ++i) {
            residuals.push_back({samples[i].features, samples[i].target - predictions[i]});
        }
        trees_.emplace_back(stump_config);
        trees_.back().fit(residuals);
        for (std::size_t i = 0; i < samples.size(); ++i) {
            predictions[i] += config_.learning_rate *
                trees_.back().predict(samples[i].features);
        }
    }
}

double GradientBoostingRegressor::predict(const Vector& features) const {
    if (!trained()) {
        throw std::logic_error("gradient boosting model is untrained");
    }
    double result = baseline_;
    for (const auto& tree : trees_) {
        result += config_.learning_rate * tree.predict(features);
    }
    return result;
}

bool GradientBoostingRegressor::trained() const noexcept { return !trees_.empty(); }

XGBoostAdapter::XGBoostAdapter(XGBoostFit fit_backend,
                               XGBoostPredict predict_backend)
    : fit_backend_(std::move(fit_backend)),
      predict_backend_(std::move(predict_backend)) {
    if (!fit_backend_ || !predict_backend_) {
        throw std::invalid_argument("XGBoost adapter requires backend callbacks");
    }
}

std::string XGBoostAdapter::name() const { return "xgboost_external"; }

void XGBoostAdapter::fit(const std::vector<TrainingExample>& samples) {
    validate_samples(samples);
    fit_backend_(samples);
    trained_ = true;
}

double XGBoostAdapter::predict(const Vector& features) const {
    if (!trained_ || !is_finite(features)) {
        throw std::logic_error("XGBoost adapter is untrained or input is invalid");
    }
    const double result = predict_backend_(features);
    if (!std::isfinite(result)) {
        throw std::runtime_error("XGBoost backend returned a non-finite prediction");
    }
    return result;
}

bool XGBoostAdapter::trained() const noexcept { return trained_; }

MLPRegressor::MLPRegressor(ModelConfig config) : config_(config) {}
std::string MLPRegressor::name() const { return "mlp"; }

void MLPRegressor::fit(const std::vector<TrainingExample>& samples) {
    input_size_ = validate_samples(samples);
    if (config_.mlp_hidden_size == 0 || config_.mlp_epochs == 0 ||
        !std::isfinite(config_.learning_rate) || config_.learning_rate <= 0.0) {
        throw std::invalid_argument("invalid MLP configuration");
    }
    const std::size_t hidden = config_.mlp_hidden_size;
    std::mt19937_64 random(config_.random_seed);
    std::uniform_real_distribution<double> initialise(
        -1.0 / std::sqrt(static_cast<double>(input_size_)),
         1.0 / std::sqrt(static_cast<double>(input_size_))
    );
    weights_.assign(2, {});
    weights_[0].resize(hidden * input_size_);
    weights_[1].resize(hidden);
    biases_.assign(hidden + 1, 0.0);
    for (auto& weight : weights_[0]) {
        weight = initialise(random);
    }
    for (auto& weight : weights_[1]) {
        weight = initialise(random);
    }

    for (std::size_t epoch = 0; epoch < config_.mlp_epochs; ++epoch) {
        for (const auto& sample : samples) {
            Vector hidden_values(hidden, 0.0);
            for (std::size_t h = 0; h < hidden; ++h) {
                double sum = biases_[h];
                for (std::size_t i = 0; i < input_size_; ++i) {
                    sum += weights_[0][h * input_size_ + i] * sample.features[i];
                }
                hidden_values[h] = std::tanh(sum);
            }
            double prediction = biases_[hidden];
            for (std::size_t h = 0; h < hidden; ++h) {
                prediction += weights_[1][h] * hidden_values[h];
            }
            if (!std::isfinite(prediction)) {
                throw std::runtime_error("MLP training diverged numerically");
            }
            const double output_gradient = clamp(
                2.0 * (prediction - sample.target), -1000.0, 1000.0
            );
            for (std::size_t h = 0; h < hidden; ++h) {
                const double old_output_weight = weights_[1][h];
                weights_[1][h] -= config_.learning_rate *
                    output_gradient * hidden_values[h];
                const double hidden_gradient = output_gradient *
                    old_output_weight * (1.0 - hidden_values[h] * hidden_values[h]);
                for (std::size_t i = 0; i < input_size_; ++i) {
                    weights_[0][h * input_size_ + i] -= config_.learning_rate *
                        hidden_gradient * sample.features[i];
                }
                biases_[h] -= config_.learning_rate * hidden_gradient;
            }
            biases_[hidden] -= config_.learning_rate * output_gradient;
        }
    }
    trained_ = true;
}

double MLPRegressor::predict(const Vector& features) const {
    if (!trained_ || features.size() != input_size_ || !is_finite(features)) {
        throw std::logic_error("MLP is untrained or input dimensions differ");
    }
    double result = biases_[config_.mlp_hidden_size];
    for (std::size_t h = 0; h < config_.mlp_hidden_size; ++h) {
        double sum = biases_[h];
        for (std::size_t i = 0; i < input_size_; ++i) {
            sum += weights_[0][h * input_size_ + i] * features[i];
        }
        result += weights_[1][h] * std::tanh(sum);
    }
    return result;
}

bool MLPRegressor::trained() const noexcept { return trained_; }

}  // namespace neural_brain
