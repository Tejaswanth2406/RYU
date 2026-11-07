#include "neural_brain/brain_graph.hpp"

#include "neural_brain/types.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace neural_brain {

BrainGraph::BrainGraph(GraphConfig config) : config_(config) {
    if (!std::isfinite(config_.pruning_threshold) ||
        config_.pruning_threshold < 0.0 || config_.pruning_threshold > 1.0 ||
        !std::isfinite(config_.learning_rate) || config_.learning_rate < 0.0 ||
        !std::isfinite(config_.synapse_decay) || config_.synapse_decay < 0.0) {
        throw std::invalid_argument("invalid graph configuration");
    }
}

NeuralCell& BrainGraph::add_cell(CellId id, const CellConfig& cell_config,
                                 const std::string& specialization) {
    if (cells_.contains(id)) {
        throw std::invalid_argument("cell identifier already exists");
    }
    return cells_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(id),
        std::forward_as_tuple(id, cell_config, specialization)
    ).first->second;
}

NeuralCell& BrainGraph::add_cell(NeuralCell neural_cell) {
    const CellId id = neural_cell.id();
    if (cells_.contains(id)) {
        throw std::invalid_argument("cell identifier already exists");
    }
    return cells_.emplace(id, std::move(neural_cell)).first->second;
}

bool BrainGraph::remove_cell(CellId id) {
    const bool removed = cells_.erase(id) != 0;
    if (removed) {
        std::erase_if(synapses_, [id](const Synapse& edge) {
            return edge.source == id || edge.target == id;
        });
    }
    return removed;
}

void BrainGraph::connect(CellId source, CellId target, double weight) {
    if (!cells_.contains(source) || !cells_.contains(target) ||
        !std::isfinite(weight) || weight < -1.0 || weight > 1.0) {
        throw std::invalid_argument("invalid synapse endpoints or weight");
    }
    const auto existing = std::find_if(
        synapses_.begin(), synapses_.end(), [source, target](const Synapse& edge) {
            return edge.source == source && edge.target == target;
        });
    if (existing != synapses_.end()) {
        existing->weight = weight;
        existing->age = 0.0;
        return;
    }
    synapses_.push_back({source, target, weight, 0.0});
}

bool BrainGraph::disconnect(CellId source, CellId target) {
    const auto old_size = synapses_.size();
    std::erase_if(synapses_, [source, target](const Synapse& edge) {
        return edge.source == source && edge.target == target;
    });
    return synapses_.size() != old_size;
}

bool BrainGraph::reinforce(CellId source, CellId target, double pre_activity,
                           double post_activity, double elapsed_time) {
    if (!std::isfinite(pre_activity) || !std::isfinite(post_activity) ||
        !std::isfinite(elapsed_time) || elapsed_time < 0.0) {
        throw std::invalid_argument("invalid synapse reinforcement values");
    }
    const auto edge = std::find_if(
        synapses_.begin(), synapses_.end(), [source, target](const Synapse& item) {
            return item.source == source && item.target == target;
        });
    if (edge == synapses_.end()) {
        return false;
    }
    const double potentiation =
        config_.learning_rate * pre_activity * post_activity * elapsed_time;
    const double depression = config_.synapse_decay * edge->weight * elapsed_time;
    edge->weight = clamp(edge->weight + potentiation - depression, -1.0, 1.0);
    cells_.at(source).reinforce(pre_activity, post_activity, elapsed_time);
    return true;
}

std::size_t BrainGraph::prune() {
    const auto previous_size = synapses_.size();
    std::erase_if(synapses_, [this](const Synapse& edge) {
        return std::abs(edge.weight) < config_.pruning_threshold;
    });
    return previous_size - synapses_.size();
}

void BrainGraph::tick(double elapsed_time) {
    if (!std::isfinite(elapsed_time) || elapsed_time < 0.0) {
        throw std::invalid_argument("graph time must be finite and non-negative");
    }
    for (auto& [id, neural_cell] : cells_) {
        (void)id;
        neural_cell.age(elapsed_time);
        neural_cell.repair(elapsed_time);
    }
    for (auto& edge : synapses_) {
        edge.age += elapsed_time;
        edge.weight *= std::exp(-config_.synapse_decay * elapsed_time);
    }
    prune();
}

NeuralCell& BrainGraph::cell(CellId id) {
    const auto found = cells_.find(id);
    if (found == cells_.end()) {
        throw std::out_of_range("unknown neural cell identifier");
    }
    return found->second;
}

const NeuralCell& BrainGraph::cell(CellId id) const {
    const auto found = cells_.find(id);
    if (found == cells_.end()) {
        throw std::out_of_range("unknown neural cell identifier");
    }
    return found->second;
}

const std::unordered_map<CellId, NeuralCell>& BrainGraph::cells() const noexcept {
    return cells_;
}

const std::vector<Synapse>& BrainGraph::synapses() const noexcept {
    return synapses_;
}

std::vector<Synapse> BrainGraph::outgoing(CellId source) const {
    std::vector<Synapse> edges;
    for (const auto& edge : synapses_) {
        if (edge.source == source) {
            edges.push_back(edge);
        }
    }
    return edges;
}

const GraphConfig& BrainGraph::config() const noexcept { return config_; }

}  // namespace neural_brain
