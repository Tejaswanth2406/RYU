#pragma once

#include "config.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace neural_brain {

class Regressor {
public:
    virtual ~Regressor() = default;
    virtual std::string name() const = 0;
    virtual void fit(const std::vector<TrainingExample>& samples) = 0;
    virtual double predict(const Vector& features) const = 0;
    virtual bool trained() const noexcept = 0;
};

class DecisionTreeRegressor final : public Regressor {
public:
    explicit DecisionTreeRegressor(ModelConfig config = {});
    std::string name() const override;
    void fit(const std::vector<TrainingExample>& samples) override;
    double predict(const Vector& features) const override;
    bool trained() const noexcept override;

private:
    struct Node {
        std::size_t feature{0};
        double threshold{0.0};
        double value{0.0};
        int left{-1};
        int right{-1};
        bool leaf{true};
    };
    int build(const std::vector<TrainingExample>& samples,
              const std::vector<std::size_t>& indices, std::size_t depth);
    double predict_node(int node, const Vector& features) const;

    ModelConfig config_;
    std::vector<Node> nodes_;
    std::size_t feature_count_{0};
};

class RandomForestRegressor final : public Regressor {
public:
    explicit RandomForestRegressor(ModelConfig config = {});
    std::string name() const override;
    void fit(const std::vector<TrainingExample>& samples) override;
    double predict(const Vector& features) const override;
    bool trained() const noexcept override;

private:
    ModelConfig config_;
    std::vector<DecisionTreeRegressor> trees_;
};

class GradientBoostingRegressor final : public Regressor {
public:
    explicit GradientBoostingRegressor(ModelConfig config = {});
    std::string name() const override;
    void fit(const std::vector<TrainingExample>& samples) override;
    double predict(const Vector& features) const override;
    bool trained() const noexcept override;

private:
    ModelConfig config_;
    double baseline_{0.0};
    std::vector<DecisionTreeRegressor> trees_;
};

using XGBoostFit = std::function<void(const std::vector<TrainingExample>&)>;
using XGBoostPredict = std::function<double(const Vector&)>;

class XGBoostAdapter final : public Regressor {
public:
    XGBoostAdapter(XGBoostFit fit_backend, XGBoostPredict predict_backend);
    std::string name() const override;
    void fit(const std::vector<TrainingExample>& samples) override;
    double predict(const Vector& features) const override;
    bool trained() const noexcept override;

private:
    XGBoostFit fit_backend_;
    XGBoostPredict predict_backend_;
    bool trained_{false};
};

class MLPRegressor final : public Regressor {
public:
    explicit MLPRegressor(ModelConfig config = {});
    std::string name() const override;
    void fit(const std::vector<TrainingExample>& samples) override;
    double predict(const Vector& features) const override;
    bool trained() const noexcept override;

private:
    ModelConfig config_;
    std::size_t input_size_{0};
    std::vector<Vector> weights_;
    Vector biases_;
    bool trained_{false};
};

}  // namespace neural_brain
