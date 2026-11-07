#include "neural_brain/feature_encoder.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace neural_brain {

void FeatureEncoder::fit(const std::vector<Vector>& samples) {
    if (samples.empty() || samples.front().empty()) {
        throw std::invalid_argument("feature encoder requires non-empty samples");
    }
    const std::size_t dimensions = samples.front().size();
    minima_.assign(dimensions, std::numeric_limits<double>::infinity());
    maxima_.assign(dimensions, -std::numeric_limits<double>::infinity());
    for (const auto& sample : samples) {
        if (sample.size() != dimensions || !is_finite(sample)) {
            throw std::invalid_argument("feature samples have inconsistent dimensions");
        }
        for (std::size_t i = 0; i < dimensions; ++i) {
            minima_[i] = std::min(minima_[i], sample[i]);
            maxima_[i] = std::max(maxima_[i], sample[i]);
        }
    }
    for (std::size_t i = 0; i < dimensions; ++i) {
        if (!std::isfinite(maxima_[i] - minima_[i])) {
            throw std::invalid_argument("feature range exceeds numeric limits");
        }
    }
}

Vector FeatureEncoder::encode(const Vector& features) const {
    if (!fitted() || features.size() != minima_.size() || !is_finite(features)) {
        throw std::invalid_argument("feature vector does not match encoder state");
    }
    Vector encoded(features.size());
    for (std::size_t i = 0; i < features.size(); ++i) {
        const double range = maxima_[i] - minima_[i];
        encoded[i] = range == 0.0 ? 0.0 :
            clamp((features[i] - minima_[i]) / range, 0.0, 1.0);
    }
    return encoded;
}

Vector FeatureEncoder::combine(const Vector& features,
                               const Vector& graph_embedding) const {
    if (!is_finite(graph_embedding)) {
        throw std::invalid_argument("graph embedding must be finite");
    }
    Vector combined = encode(features);
    combined.insert(combined.end(), graph_embedding.begin(), graph_embedding.end());
    return combined;
}

bool FeatureEncoder::fitted() const noexcept {
    return !minima_.empty() && minima_.size() == maxima_.size();
}

std::size_t FeatureEncoder::feature_count() const noexcept {
    return minima_.size();
}

}  // namespace neural_brain
