#pragma once

#include "types.hpp"

#include <cstddef>
#include <vector>

namespace neural_brain {

class FeatureEncoder {
public:
    void fit(const std::vector<Vector>& samples);
    Vector encode(const Vector& features) const;
    Vector combine(const Vector& features, const Vector& graph_embedding) const;
    bool fitted() const noexcept;
    std::size_t feature_count() const noexcept;

private:
    Vector minima_;
    Vector maxima_;
};

}  // namespace neural_brain
