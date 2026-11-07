#pragma once

#include "brain_graph.hpp"

#include <unordered_map>

namespace neural_brain {

struct GraphRepresentation {
    std::unordered_map<CellId, Vector> node_embeddings;
    Vector graph_embedding;
};

class GNNLayer {
public:
    explicit GNNLayer(GraphConfig config = {});
    GraphRepresentation forward(const BrainGraph& graph) const;

private:
    GraphConfig config_;
};

}  // namespace neural_brain
