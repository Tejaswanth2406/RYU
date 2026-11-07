#pragma once

#include "config.hpp"

#include <mutex>
#include <vector>

namespace neural_brain {

class MemoryLayer {
public:
    explicit MemoryLayer(MemoryConfig config = {});

    void store_working(Experience experience);
    void store_episode(Experience experience);
    void store_semantic(Experience experience);
    std::vector<RetrievedMemory> retrieve(const Vector& query,
                                          std::size_t limit = 5);
    void reinforce(MemoryKind kind, std::size_t index, double amount);
    void consolidate();
    void decay(double elapsed_time = 1.0);
    std::size_t prune();

    std::size_t working_size() const;
    std::size_t episodic_size() const;
    std::size_t semantic_size() const;

private:
    using Store = std::vector<Experience>;
    void insert(Store& store, std::size_t capacity, Experience experience);
    double retention(const Experience& experience) const;
    std::size_t prune_unlocked();

    MemoryConfig config_;
    mutable std::mutex mutex_;
    Store working_;
    Store episodic_;
    Store semantic_;
};

}  // namespace neural_brain
