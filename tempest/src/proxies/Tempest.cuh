#ifndef TEMPEST_H
#define TEMPEST_H

#include <vector>
#include <thread>
#include "../core/tempest.cuh"
#include "../data/walk_set/walk_set.cuh"
#include "../data/enums.cuh"
#include "../common/const.cuh"

#ifdef HAS_CUDA

__global__ void get_edge_count_kernel(size_t* result, const TempestStore* tempest);

#endif

class Tempest {
    bool use_gpu;
    TempestStore* tempest;

public:
    explicit Tempest(
        bool is_directed,
        bool use_gpu,
        int64_t max_time_capacity=-1,
        bool enable_weight_computation=false,
        double timescale_bound=DEFAULT_TIMESCALE_BOUND,
        int walk_padding_value=EMPTY_NODE_VALUE);

    ~Tempest();

    void add_multiple_edges(
        const int* sources,
        const int* targets,
        const int64_t* timestamps,
        size_t edges_size) const;

    void add_multiple_edges(
        const std::vector<std::tuple<int, int, int64_t>>& edges) const;

    WalkSet get_random_walks_and_times_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) const;

    WalkSet get_random_walks_and_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) const;

    [[nodiscard]] size_t get_node_count() const;

    [[nodiscard]] size_t get_edge_count() const;

    [[nodiscard]] std::vector<int> get_node_ids() const;

    [[nodiscard]] std::vector<std::tuple<int, int, int64_t>> get_edges() const;

    [[nodiscard]] bool get_is_directed() const;

    void clear() const;

    [[nodiscard]] size_t get_memory_used() const;
};

#endif // TEMPEST_H
