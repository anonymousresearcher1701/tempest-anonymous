#include "tempest.cuh"

#include "tempest_cpu.cuh"
#include "tempest_kernels_full_walk.cuh"
#include "tempest_kernels_step_based.cuh"

/**
 * Common functions
*/

HOST void tempest::add_multiple_edges(
    const TempestStore *tempest,
    const int *sources,
    const int *targets,
    const int64_t *timestamps,
    const size_t num_edges) {
    #ifdef HAS_CUDA
    if (tempest->use_gpu) {
        temporal_graph::add_multiple_edges_cuda(
            tempest->temporal_graph,
            sources,
            targets,
            timestamps,
            num_edges);
    } else
    #endif
    {
        temporal_graph::add_multiple_edges_std(
            tempest->temporal_graph,
            sources,
            targets,
            timestamps,
            num_edges);
    }
}

HOST size_t tempest::get_node_count(const TempestStore *tempest) {
    return temporal_graph::get_node_count(tempest->temporal_graph);
}

HOST DEVICE size_t tempest::get_edge_count(const TempestStore *tempest) {
    return temporal_graph::get_total_edges(tempest->temporal_graph);
}

HOST DataBlock<int> tempest::get_node_ids(const TempestStore *tempest) {
    return temporal_graph::get_node_ids(tempest->temporal_graph);
}

HOST DataBlock<Edge> tempest::get_edges(const TempestStore *tempest) {
    return temporal_graph::get_edges(tempest->temporal_graph);
}

HOST bool tempest::get_is_directed(const TempestStore *tempest) {
    return tempest->is_directed;
}

HOST void tempest::clear(TempestStore *tempest) {
    tempest->temporal_graph = new TemporalGraphStore(
        tempest->is_directed,
        tempest->use_gpu,
        tempest->max_time_capacity,
        tempest->enable_weight_computation,
        tempest->timescale_bound);
}

/**
 * Std implementations
 */

HOST WalkSet tempest::get_random_walks_and_times_for_all_nodes_std(
    const TempestStore *tempest,
    const int max_walk_len,
    const RandomPickerType *walk_bias,
    const int num_walks_per_node,
    const RandomPickerType *initial_edge_bias,
    const WalkDirection walk_direction) {
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    const auto repeated_node_ids = repeat_elements(
        temporal_graph::get_node_ids(tempest->temporal_graph),
        num_walks_per_node,
        tempest->use_gpu);
    shuffle_vector_host<int>(repeated_node_ids.data, repeated_node_ids.size);

    WalkSet walk_set(repeated_node_ids.size, max_walk_len, tempest->walk_padding_value, tempest->use_gpu);

    // max_walk_len requires walk_len - 1 steps
    double *rand_nums = generate_n_random_numbers(repeated_node_ids.size + repeated_node_ids.size * max_walk_len * 2, false);

    launch_random_walk_cpu(
        tempest->temporal_graph,
        tempest->is_directed,
        &walk_set,
        max_walk_len,
        repeated_node_ids.data,
        repeated_node_ids.size,
        *walk_bias,
        *initial_edge_bias,
        walk_direction,
        rand_nums);

    // Clean up
    clear_memory(&rand_nums, false);

    return walk_set;
}

HOST WalkSet tempest::get_random_walks_and_times_std(
    const TempestStore *tempest,
    const int max_walk_len,
    const RandomPickerType *walk_bias,
    const int num_walks_total,
    const RandomPickerType *initial_edge_bias,
    const WalkDirection walk_direction) {
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    WalkSet walk_set(num_walks_total, max_walk_len, tempest->walk_padding_value, tempest->use_gpu);
    // max_walk_len requires walk_len - 1 steps
    double *rand_nums = generate_n_random_numbers(num_walks_total + num_walks_total * max_walk_len * 2, false);

    const std::vector<int> start_node_ids(num_walks_total, -1);

    launch_random_walk_cpu(
        tempest->temporal_graph,
        tempest->is_directed,
        &walk_set,
        max_walk_len,
        start_node_ids.data(),
        num_walks_total,
        *walk_bias,
        *initial_edge_bias,
        walk_direction,
        rand_nums);

    // Clean up
    clear_memory(&rand_nums, false);

    return walk_set;
}

/**
 * CUDA implementations
 */

#ifdef HAS_CUDA

HOST WalkSet tempest::get_random_walks_and_times_for_all_nodes_cuda(
    const TempestStore *tempest,
    const int max_walk_len,
    const RandomPickerType *walk_bias,
    const int num_walks_per_node,
    const RandomPickerType *initial_edge_bias,
    const WalkDirection walk_direction) {
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    // Get all node IDs and repeat them for multiple walks per node
    const auto node_ids = temporal_graph::get_node_ids(tempest->temporal_graph);
    const auto repeated_node_ids = repeat_elements(
        node_ids,
        num_walks_per_node,
        tempest->use_gpu);

    // Calculate optimal kernel launch parameters
    auto [grid_dim, block_dim] = get_optimal_launch_params(
        repeated_node_ids.size,
        tempest->cuda_device_prop,
        BLOCK_DIM_GENERATING_RANDOM_WALKS);

    // Initialize random numbers between [0.0, 1.0)
    double *rand_nums = generate_n_random_numbers(repeated_node_ids.size + repeated_node_ids.size * max_walk_len * 2, true);

    // Shuffle node IDs for randomization
    shuffle_vector_device<int>(repeated_node_ids.data, repeated_node_ids.size);
    CUDA_KERNEL_CHECK("After shuffle_vector_device in get_random_walks_and_times_for_all_nodes_cuda");

    // Create and initialize the walk set on device
    const WalkSet walk_set(repeated_node_ids.size, max_walk_len, tempest->walk_padding_value, true);
    WalkSet *d_walk_set;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_walk_set, sizeof(WalkSet)));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_walk_set, &walk_set, sizeof(WalkSet), cudaMemcpyHostToDevice));

    // Create device pointer for the temporal graph
    TemporalGraphStore *d_temporal_graph = temporal_graph::to_device_ptr(tempest->temporal_graph);

    launch_random_walk_kernel_full_walk(
        d_temporal_graph,
        tempest->is_directed,
        d_walk_set,
        max_walk_len,
        repeated_node_ids.data,
        repeated_node_ids.size,
        *walk_bias,
        *initial_edge_bias,
        walk_direction,
        rand_nums,
        grid_dim,
        block_dim);

    CUDA_KERNEL_CHECK("After generate_random_walks_kernel in get_random_walks_and_times_for_all_nodes_cuda");

    // Copy walk set from device to host
    WalkSet host_walk_set(repeated_node_ids.size, max_walk_len, tempest->walk_padding_value, false);
    host_walk_set.copy_from_device(d_walk_set);

    // Free device memory
    clear_memory(&rand_nums, true);
    temporal_graph::free_device_pointers(d_temporal_graph);
    CUDA_CHECK_AND_CLEAR(cudaFree(d_walk_set));

    return host_walk_set;
}

HOST WalkSet tempest::get_random_walks_and_times_cuda(
    const TempestStore *tempest,
    const int max_walk_len,
    const RandomPickerType *walk_bias,
    const int num_walks_total,
    const RandomPickerType *initial_edge_bias,
    const WalkDirection walk_direction) {
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    // Calculate optimal kernel launch parameters
    auto [grid_dim, block_dim] = get_optimal_launch_params(
        num_walks_total,
        tempest->cuda_device_prop,
        BLOCK_DIM_GENERATING_RANDOM_WALKS);

    // Initialize all start node IDs to -1 (indicating random start)
    int *start_node_ids;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&start_node_ids, num_walks_total * sizeof(int)));
    fill_memory(start_node_ids, num_walks_total, -1, tempest->use_gpu);

    // Initialize random numbers between [0.0, 1.0)
    double *rand_nums = generate_n_random_numbers(num_walks_total + num_walks_total * max_walk_len * 2, true);

    // Create and initialize the walk set on device
    const WalkSet walk_set(num_walks_total, max_walk_len, tempest->walk_padding_value, true);
    WalkSet *d_walk_set;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_walk_set, sizeof(WalkSet)));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_walk_set, &walk_set, sizeof(WalkSet), cudaMemcpyHostToDevice));

    // Create device pointer for the temporal graph
    TemporalGraphStore *d_temporal_graph = temporal_graph::to_device_ptr(tempest->temporal_graph);

    // Launch kernel
    launch_random_walk_kernel_full_walk(
        d_temporal_graph,
        tempest->is_directed,
        d_walk_set,
        max_walk_len,
        start_node_ids,
        num_walks_total,
        *walk_bias,
        *initial_edge_bias,
        walk_direction,
        rand_nums,
        grid_dim,
        block_dim);

    CUDA_KERNEL_CHECK("After generate_random_walks_kernel in get_random_walks_and_times_cuda");

    // Copy walk set from device to host
    WalkSet host_walk_set(num_walks_total, max_walk_len, tempest->walk_padding_value, false);
    host_walk_set.copy_from_device(d_walk_set);

    // Free device memory
    clear_memory(&rand_nums, true);
    temporal_graph::free_device_pointers(d_temporal_graph);
    CUDA_CHECK_AND_CLEAR(cudaFree(start_node_ids));
    CUDA_CHECK_AND_CLEAR(cudaFree(d_walk_set));

    return host_walk_set;
}

HOST TempestStore* tempest::to_device_ptr(const TempestStore *tempest) {
    // Create a new TemporalRandomWalk object on the device
    TempestStore *device_tempest;
    CUDA_CHECK_AND_CLEAR(cudaMalloc(&device_tempest, sizeof(TempestStore)));

    // Create a temporary copy to modify for device pointers
    TempestStore temp_tempest = *tempest;

    // Copy TemporalGraph to device
    if (tempest->temporal_graph) {
        temp_tempest.temporal_graph = temporal_graph::to_device_ptr(
            tempest->temporal_graph);
    }

    // cudaDeviceProp aren't needed on device, set to nullptr
    temp_tempest.cuda_device_prop = nullptr;

    // Make sure use_gpu is set to true
    temp_tempest.use_gpu = true;

    // Copy the updated struct to device
    CUDA_CHECK_AND_CLEAR(
        cudaMemcpy(device_tempest, &temp_tempest, sizeof(TempestStore),
            cudaMemcpyHostToDevice));

    temp_tempest.owns_data = false;

    return device_tempest;
}

HOST void tempest::free_device_pointers(TempestStore *d_tempest) {
    if (!d_tempest) return;

    // Copy the struct from device to host to access pointers
    TempestStore h_tempest;
    CUDA_CHECK_AND_CLEAR(
        cudaMemcpy(&h_tempest, d_tempest, sizeof(TempestStore),
            cudaMemcpyDeviceToHost));
    h_tempest.owns_data = false;

    // Free only the nested device pointers (not their underlying data)
    if (h_tempest.temporal_graph) temporal_graph::free_device_pointers(
        h_tempest.temporal_graph);
    if (h_tempest.cuda_device_prop) clear_memory(&h_tempest.cuda_device_prop, true);

    clear_memory(&d_tempest, true);
}

#endif

HOST size_t tempest::get_memory_used(TempestStore* tempest) {
    return temporal_graph::get_memory_used(tempest->temporal_graph);
}
