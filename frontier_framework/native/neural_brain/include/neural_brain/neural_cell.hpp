#pragma once

#include "config.hpp"

#include <cstdint>
#include <string>

namespace neural_brain {

struct CellSnapshot {
    CellId id{0};
    Vector state;
    double energy{0.0};
    double damage{0.0};
    double learning_trace{0.0};
    double age{0.0};
    bool active{true};
    std::string specialization;
};

class NeuralCell {
public:
    NeuralCell(CellId id, CellConfig config, std::string specialization = "general");

    CellId id() const noexcept;
    const Vector& state() const noexcept;
    double energy() const noexcept;
    double damage() const noexcept;
    double learning_trace() const noexcept;
    double age() const noexcept;
    bool active() const noexcept;
    const std::string& specialization() const noexcept;

    void activate(const Vector& input, double elapsed_time = 1.0);
    void reinforce(double pre_activity, double post_activity, double elapsed_time = 1.0);
    void accumulate_damage(double amount);
    void repair(double elapsed_time = 1.0);
    void age(double elapsed_time = 1.0);
    void deactivate() noexcept;
    NeuralCell divide(CellId child_id);
    void set_specialization(std::string specialization);
    CellSnapshot snapshot() const;

private:
    double apply_activation(double value) const;

    CellId id_;
    CellConfig config_;
    Vector state_;
    double energy_;
    double damage_{0.0};
    double learning_trace_{0.0};
    double age_{0.0};
    bool active_{true};
    std::string specialization_;
};

}  // namespace neural_brain
