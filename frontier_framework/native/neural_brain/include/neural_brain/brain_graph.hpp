#pragma once

#include "config.hpp"
#include "neural_cell.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace neural_brain {

struct Synapse {
    CellId source{0};
    CellId target{0};
    double weight{0.0};
    double age{0.0};
};

class BrainGraph {
public:
    explicit BrainGraph(GraphConfig config = {});

    NeuralCell& add_cell(CellId id, const CellConfig& cell_config,
                         const std::string& specialization = "general");
    NeuralCell& add_cell(NeuralCell cell);
    bool remove_cell(CellId id);
    void connect(CellId source, CellId target, double weight);
    bool disconnect(CellId source, CellId target);
    bool reinforce(CellId source, CellId target, double pre_activity,
                   double post_activity, double elapsed_time = 1.0);
    std::size_t prune();
    void tick(double elapsed_time = 1.0);

    NeuralCell& cell(CellId id);
    const NeuralCell& cell(CellId id) const;
    const std::unordered_map<CellId, NeuralCell>& cells() const noexcept;
    const std::vector<Synapse>& synapses() const noexcept;
    std::vector<Synapse> outgoing(CellId source) const;
    const GraphConfig& config() const noexcept;

private:
    GraphConfig config_;
    std::unordered_map<CellId, NeuralCell> cells_;
    std::vector<Synapse> synapses_;
};

}  // namespace neural_brain
