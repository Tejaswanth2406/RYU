#include "neural_brain/neural_cell.hpp"

#include "neural_brain/types.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace neural_brain {

NeuralCell::NeuralCell(CellId id, CellConfig config, std::string specialization)
    : id_(id),
      config_(config),
      state_(config.state_size, 0.0),
      energy_(config.initial_energy),
      specialization_(std::move(specialization)) {
    if (id_ == 0 || config_.state_size == 0 || !std::isfinite(energy_) ||
        energy_ <= 0.0 || !std::isfinite(config_.energy_cost) ||
        config_.energy_cost < 0.0 || !std::isfinite(config_.recovery_rate) ||
        config_.recovery_rate < 0.0 || !std::isfinite(config_.hebbian_rate) ||
        config_.hebbian_rate < 0.0 || !std::isfinite(config_.synapse_decay) ||
        config_.synapse_decay < 0.0 || !std::isfinite(config_.damage_rate) ||
        config_.damage_rate < 0.0 || !std::isfinite(config_.repair_rate) ||
        config_.repair_rate < 0.0 || !std::isfinite(config_.mutation_rate) ||
        config_.mutation_rate < 0.0 || !std::isfinite(config_.termination_energy) ||
        config_.termination_energy < 0.0 ||
        config_.termination_energy > config_.initial_energy ||
        !std::isfinite(config_.senescence_age) || config_.senescence_age <= 0.0 ||
        specialization_.empty()) {
        throw std::invalid_argument("invalid neural cell configuration");
    }
}

CellId NeuralCell::id() const noexcept { return id_; }
const Vector& NeuralCell::state() const noexcept { return state_; }
double NeuralCell::energy() const noexcept { return energy_; }
double NeuralCell::damage() const noexcept { return damage_; }
double NeuralCell::learning_trace() const noexcept { return learning_trace_; }
double NeuralCell::age() const noexcept { return age_; }
bool NeuralCell::active() const noexcept { return active_; }
const std::string& NeuralCell::specialization() const noexcept { return specialization_; }

void NeuralCell::activate(const Vector& input, double elapsed_time) {
    if (!active_) {
        return;
    }
    if (input.size() != state_.size() || !is_finite(input) ||
        !std::isfinite(elapsed_time) || elapsed_time <= 0.0) {
        throw std::invalid_argument("invalid neural cell activation input");
    }
    energy_ = std::max(0.0, energy_ - config_.energy_cost * elapsed_time);
    age_ += elapsed_time;
    damage_ = std::min(1.0, damage_ + config_.damage_rate * elapsed_time);
    if (energy_ <= config_.termination_energy || damage_ >= 1.0 ||
        age_ >= config_.senescence_age) {
        active_ = false;
        return;
    }
    const double health = 1.0 - damage_;
    for (std::size_t i = 0; i < state_.size(); ++i) {
        state_[i] = apply_activation(input[i] * health);
    }
    repair(elapsed_time);
}

void NeuralCell::reinforce(double pre_activity, double post_activity, double elapsed_time) {
    if (!std::isfinite(pre_activity) || !std::isfinite(post_activity) ||
        !std::isfinite(elapsed_time) || elapsed_time < 0.0) {
        throw std::invalid_argument("invalid synaptic reinforcement values");
    }
    learning_trace_ = clamp(
        learning_trace_ * std::exp(-config_.synapse_decay * elapsed_time) +
            config_.hebbian_rate * pre_activity * post_activity * elapsed_time,
        -1.0, 1.0
    );
    damage_ = clamp(damage_ - config_.repair_rate * elapsed_time, 0.0, 1.0);
}

void NeuralCell::accumulate_damage(double amount) {
    if (!std::isfinite(amount) || amount < 0.0) {
        throw std::invalid_argument("damage must be finite and non-negative");
    }
    damage_ = std::min(1.0, damage_ + amount);
    if (damage_ >= 1.0) {
        active_ = false;
    }
}

void NeuralCell::repair(double elapsed_time) {
    if (!std::isfinite(elapsed_time) || elapsed_time < 0.0) {
        throw std::invalid_argument("repair time must be finite and non-negative");
    }
    damage_ = std::max(0.0, damage_ - config_.repair_rate * elapsed_time);
    energy_ = std::min(
        config_.initial_energy,
        energy_ + config_.recovery_rate * elapsed_time
    );
}

void NeuralCell::age(double elapsed_time) {
    if (!std::isfinite(elapsed_time) || elapsed_time < 0.0) {
        throw std::invalid_argument("age increment must be finite and non-negative");
    }
    age_ += elapsed_time;
    learning_trace_ *= std::exp(-config_.synapse_decay * elapsed_time);
    if (age_ >= config_.senescence_age || energy_ <= config_.termination_energy) {
        active_ = false;
    }
}

void NeuralCell::deactivate() noexcept { active_ = false; }

NeuralCell NeuralCell::divide(CellId child_id) {
    if (!active_ || child_id == 0 || child_id == id_ || energy_ < 2.0 * config_.termination_energy) {
        throw std::logic_error("cell cannot divide in its current state");
    }
    std::mt19937_64 random(config_.random_seed ^ id_ ^ child_id);
    std::normal_distribution<double> mutation(0.0, config_.mutation_rate);
    CellConfig child_config = config_;
    child_config.hebbian_rate = std::max(
        0.0, child_config.hebbian_rate + mutation(random)
    );
    child_config.recovery_rate = std::max(
        0.0, child_config.recovery_rate + mutation(random)
    );
    NeuralCell child(child_id, child_config, specialization_);
    child.state_ = state_;
    for (double& value : child.state_) {
        value = std::tanh(value + mutation(random));
    }
    child.energy_ = energy_ * 0.5;
    child.damage_ = damage_;
    child.learning_trace_ = learning_trace_;
    energy_ *= 0.5;
    return child;
}

void NeuralCell::set_specialization(std::string specialization) {
    if (specialization.empty()) {
        throw std::invalid_argument("cell specialization cannot be empty");
    }
    specialization_ = std::move(specialization);
}

CellSnapshot NeuralCell::snapshot() const {
    return {id_, state_, energy_, damage_, learning_trace_, age_, active_, specialization_};
}

double NeuralCell::apply_activation(double value) const {
    switch (config_.activation) {
        case Activation::identity:
            return value;
        case Activation::tanh:
            return std::tanh(value);
        case Activation::sigmoid:
            return 1.0 / (1.0 + std::exp(-std::clamp(value, -60.0, 60.0)));
        case Activation::relu:
            return std::max(0.0, value);
    }
    throw std::invalid_argument("unknown activation function");
}

}  // namespace neural_brain
