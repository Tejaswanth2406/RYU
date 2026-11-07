#include "neural_brain/types.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace neural_brain {

double clamp(double value, double lower, double upper) {
    if (!std::isfinite(value) || lower > upper) {
        throw std::invalid_argument("invalid value or clamp interval");
    }
    return value < lower ? lower : (value > upper ? upper : value);
}

double cosine_similarity(const Vector& lhs, const Vector& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("cosine similarity requires equal vector sizes");
    }
    double dot = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!std::isfinite(lhs[i]) || !std::isfinite(rhs[i])) {
            throw std::invalid_argument("cosine similarity requires finite values");
        }
        dot += lhs[i] * rhs[i];
        lhs_norm += lhs[i] * lhs[i];
        rhs_norm += rhs[i] * rhs[i];
    }
    if (lhs_norm == 0.0 || rhs_norm == 0.0) {
        return 0.0;
    }
    return clamp(dot / std::sqrt(lhs_norm * rhs_norm), -1.0, 1.0);
}

bool is_finite(const Vector& values) {
    for (const double value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

}  // namespace neural_brain
