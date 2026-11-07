#pragma once

#include "tree_models.hpp"

#include <memory>
#include <vector>

namespace neural_brain {

class Ensemble {
public:
    explicit Ensemble(double latency_cost_scale_microseconds = 1000.0);
    void add_model(std::shared_ptr<Regressor> model);
    void fit(const std::vector<TrainingExample>& samples);
    Prediction predict(const Vector& features) const;
    void feedback(const Vector& features, double observed, double rate);
    std::vector<std::string> model_names() const;
    bool trained() const noexcept;

private:
    struct Expert {
        std::shared_ptr<Regressor> model;
        double weight{1.0};
        double absolute_error{0.0};
        std::size_t feedback_count{0};
        mutable double average_latency_microseconds{0.0};
        mutable std::size_t prediction_count{0};
    };
    double latency_cost_scale_microseconds_;
    std::vector<Expert> experts_;
};

}  // namespace neural_brain
