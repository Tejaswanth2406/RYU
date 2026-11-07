#include "neural_brain/memory_layer.hpp"

#include "neural_brain/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace neural_brain {
namespace {

double current_time_seconds() {
    using Clock = std::chrono::system_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

}  // namespace

MemoryLayer::MemoryLayer(MemoryConfig config) : config_(config) {
    if (!std::isfinite(config_.decay_rate) || config_.decay_rate < 0.0 ||
        !std::isfinite(config_.consolidation_threshold) ||
        config_.consolidation_threshold < 0.0 ||
        config_.consolidation_threshold > 1.0 ||
        !std::isfinite(config_.recency_scale) || config_.recency_scale <= 0.0) {
        throw std::invalid_argument("invalid memory configuration");
    }
}

void MemoryLayer::insert(Store& store, std::size_t capacity, Experience experience) {
    if (experience.features.empty() || !is_finite(experience.features) ||
        !is_finite(experience.outcome) || !std::isfinite(experience.importance) ||
        !std::isfinite(experience.timestamp) || experience.timestamp < 0.0) {
        throw std::invalid_argument("memory experience must contain finite values");
    }
    experience.importance = clamp(experience.importance, 0.0, 1.0);
    if (experience.timestamp <= 0.0) {
        experience.timestamp = current_time_seconds();
    }
    if (capacity == 0) {
        return;
    }
    store.push_back(std::move(experience));
    if (store.size() > capacity) {
        const auto least_valuable = std::min_element(
            store.begin(), store.end(), [this](const Experience& lhs, const Experience& rhs) {
                return retention(lhs) < retention(rhs);
            });
        store.erase(least_valuable);
    }
}

void MemoryLayer::store_working(Experience experience) {
    std::lock_guard lock(mutex_);
    insert(working_, config_.working_capacity, std::move(experience));
}

void MemoryLayer::store_episode(Experience experience) {
    std::lock_guard lock(mutex_);
    insert(episodic_, config_.episodic_capacity, std::move(experience));
}

void MemoryLayer::store_semantic(Experience experience) {
    std::lock_guard lock(mutex_);
    insert(semantic_, config_.semantic_capacity, std::move(experience));
}

std::vector<RetrievedMemory> MemoryLayer::retrieve(const Vector& query,
                                                   std::size_t limit) {
    if (query.empty() || !is_finite(query)) {
        throw std::invalid_argument("memory query must be non-empty and finite");
    }
    std::lock_guard lock(mutex_);
    std::vector<RetrievedMemory> results;
    auto collect = [&](Store& store, MemoryKind kind) {
        for (auto& experience : store) {
            if (experience.features.size() != query.size()) {
                continue;
            }
            const double similarity = cosine_similarity(query, experience.features);
            const double relevance = std::max(0.0, similarity);
            if (relevance == 0.0) {
                continue;
            }
            ++experience.retrievals;
            experience.importance = clamp(
                experience.importance + 0.001 * relevance, 0.0, 1.0
            );
            results.push_back({
                experience, kind, similarity, retention(experience)
            });
        }
    };
    collect(working_, MemoryKind::working);
    collect(episodic_, MemoryKind::episodic);
    collect(semantic_, MemoryKind::semantic);
    std::sort(results.begin(), results.end(), [](const auto& lhs, const auto& rhs) {
        const double left_score = lhs.similarity * lhs.retention_score;
        const double right_score = rhs.similarity * rhs.retention_score;
        return left_score > right_score;
    });
    if (results.size() > limit) {
        results.resize(limit);
    }
    return results;
}

void MemoryLayer::reinforce(MemoryKind kind, std::size_t index, double amount) {
    if (!std::isfinite(amount) || amount < 0.0) {
        throw std::invalid_argument("memory reinforcement must be non-negative");
    }
    std::lock_guard lock(mutex_);
    Store* store = kind == MemoryKind::working ? &working_ :
                   kind == MemoryKind::episodic ? &episodic_ : &semantic_;
    if (index >= store->size()) {
        throw std::out_of_range("memory index is out of range");
    }
    (*store)[index].importance = clamp(
        (*store)[index].importance + amount, 0.0, 1.0
    );
}

void MemoryLayer::consolidate() {
    std::lock_guard lock(mutex_);
    for (auto& experience : working_) {
        if (experience.importance >= config_.consolidation_threshold) {
            insert(episodic_, config_.episodic_capacity, experience);
        }
    }
    working_.clear();
    const double semantic_threshold =
        std::min(1.0, config_.consolidation_threshold * 1.5);
    for (auto& experience : episodic_) {
        if (experience.importance >= semantic_threshold) {
            const bool already_consolidated = std::any_of(
                semantic_.begin(), semantic_.end(), [&experience](const Experience& item) {
                    return item.context == experience.context &&
                           item.features == experience.features &&
                           item.outcome == experience.outcome;
                }
            );
            if (!already_consolidated) {
                insert(semantic_, config_.semantic_capacity, experience);
            }
        }
    }
}

void MemoryLayer::decay(double elapsed_time) {
    if (!std::isfinite(elapsed_time) || elapsed_time < 0.0) {
        throw std::invalid_argument("memory decay time must be non-negative");
    }
    std::lock_guard lock(mutex_);
    const double factor = std::exp(-config_.decay_rate * elapsed_time);
    auto apply = [factor](Store& store) {
        for (auto& experience : store) {
            experience.importance *= factor;
        }
    };
    apply(working_);
    apply(episodic_);
    apply(semantic_);
    prune_unlocked();
}

std::size_t MemoryLayer::prune() {
    std::lock_guard lock(mutex_);
    return prune_unlocked();
}

std::size_t MemoryLayer::prune_unlocked() {
    const std::size_t before = working_.size() + episodic_.size() + semantic_.size();
    const double minimum_retention = 0.01;
    auto remove_low_value = [this, minimum_retention](Store& store) {
        std::erase_if(store, [this, minimum_retention](const Experience& experience) {
            return retention(experience) < minimum_retention;
        });
    };
    remove_low_value(working_);
    remove_low_value(episodic_);
    remove_low_value(semantic_);
    return before - working_.size() - episodic_.size() - semantic_.size();
}

std::size_t MemoryLayer::working_size() const {
    std::lock_guard lock(mutex_);
    return working_.size();
}

std::size_t MemoryLayer::episodic_size() const {
    std::lock_guard lock(mutex_);
    return episodic_.size();
}

std::size_t MemoryLayer::semantic_size() const {
    std::lock_guard lock(mutex_);
    return semantic_.size();
}

double MemoryLayer::retention(const Experience& experience) const {
    const double age = std::max(0.0, current_time_seconds() - experience.timestamp);
    const double recency = std::exp(-age / config_.recency_scale);
    const double reinforcement = std::min(1.0, std::log1p(
        static_cast<double>(experience.retrievals)
    ) / 5.0);
    return clamp(
        0.5 * experience.importance + 0.3 * recency + 0.2 * reinforcement,
        0.0, 1.0
    );
}

}  // namespace neural_brain
