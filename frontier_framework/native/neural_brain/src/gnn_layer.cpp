#include "neural_brain/gnn_layer.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace neural_brain {

GNNLayer::GNNLayer(GraphConfig config) : config_(config) {
    if (config_.embedding_size == 0 || !std::isfinite(config_.learning_rate) ||
        config_.learning_rate < 0.0) {
        throw std::invalid_argument("invalid graph neural network configuration");
    }
}

GraphRepresentation GNNLayer::forward(const BrainGraph& graph) const {
    GraphRepresentation output;
    if (graph.cells().empty()) {
        output.graph_embedding.assign(config_.embedding_size, 0.0);
        return output;
    }

    std::unordered_map<CellId, Vector> states;
    for (const auto& [id, cell] : graph.cells()) {
        Vector projected(config_.embedding_size, 0.0);
        const Vector& state = cell.state();
        if (!is_finite(state)) {
            throw std::runtime_error("graph contains a non-finite cell state");
        }
        for (std::size_t i = 0; i < state.size(); ++i) {
            projected[i % projected.size()] += state[i];
        }
        for (double& value : projected) {
            value = std::tanh(value);
        }
        states.emplace(id, std::move(projected));
    }

    for (std::size_t step = 0; step < config_.message_passing_steps; ++step) {
        auto next = states;
        for (const auto& [id, own_state] : states) {
            Vector message(config_.embedding_size, 0.0);
            double total_weight = 0.0;
            for (const auto& edge : graph.synapses()) {
                if (edge.target != id || edge.weight == 0.0) {
                    continue;
                }
                const auto neighbor = states.find(edge.source);
                if (neighbor == states.end()) {
                    continue;
                }
                double attention = 1.0;
                if (config_.aggregation == AggregationMode::attention) {
                    attention = std::exp(cosine_similarity(own_state, neighbor->second));
                }
                const double weight = std::abs(edge.weight) * attention;
                total_weight += weight;
                for (std::size_t i = 0; i < message.size(); ++i) {
                    message[i] += neighbor->second[i] *
                        std::copysign(weight, edge.weight);
                }
            }
            if (total_weight > 0.0) {
                for (std::size_t i = 0; i < message.size(); ++i) {
                    message[i] /= total_weight;
                    next[id][i] = std::tanh(
                        own_state[i] + config_.learning_rate * message[i]
                    );
                }
            }
        }
        states = std::move(next);
    }

    output.node_embeddings = states;
    output.graph_embedding.assign(config_.embedding_size, 0.0);
    for (const auto& [id, embedding] : states) {
        (void)id;
        for (std::size_t i = 0; i < embedding.size(); ++i) {
            output.graph_embedding[i] += embedding[i];
        }
    }
    for (double& value : output.graph_embedding) {
        value /= static_cast<double>(states.size());
    }
    return output;
}

}  // namespace neural_brain
