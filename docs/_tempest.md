Module _tempest
===============

Classes
-------

`ExponentialIndexRandomPicker(...)`
:   __init__(self: _tempest.ExponentialIndexRandomPicker, use_gpu: bool = False) -> None
    
    
    Initialize index based exponential time decay random picker.
    
    Args:
        Args:
        use_gpu (bool): Should use GPU or not.

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Methods

    `pick_random(...)`
    :   pick_random(self: _tempest.ExponentialIndexRandomPicker, start: typing.SupportsInt, end: typing.SupportsInt, prioritize_end: bool = True) -> int
        
        
        Pick random index with index based exponential time decay probability.
        
        Args:
            start (int): Start index inclusive
            end (int): End index exclusive
            prioritize_end (bool, optional): Prioritize recent timestamps. Default: True
        
        Returns:
            int: Selected index

`LinearRandomPicker(...)`
:   __init__(self: _tempest.LinearRandomPicker, use_gpu: bool = False) -> None
    
    
    Initialize linear time decay random picker.
    
    Args:
        use_gpu (bool): Should use GPU or not.

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Methods

    `pick_random(...)`
    :   pick_random(self: _tempest.LinearRandomPicker, start: typing.SupportsInt, end: typing.SupportsInt, prioritize_end: bool = True) -> int
        
        
        Pick random index with linear time decay probability.
        
        Args:
            start (int): Start index inclusive
            end (int): End index exclusive
            prioritize_end (bool, optional): Prioritize recent timestamps. Default: True
        
        Returns:
            int: Selected index

`Tempest(...)`
:   __init__(self: _tempest.Tempest, is_directed: bool, use_gpu: bool = False, max_time_capacity: typing.SupportsInt | None = None, enable_weight_computation: bool | None = None, timescale_bound: typing.SupportsFloat | None = None, walk_padding_value: typing.SupportsInt | None = None) -> None
    
    
    Initialize the Tempest instance.
    
    Args:
    is_directed (bool): Whether to create a directed graph.
    use_gpu (bool): Whether to use GPU or not.
    max_time_capacity (int, optional): Maximum time window for edges. Edges older than (latest_time - max_time_capacity) are removed. Use -1 for no limit. Defaults to -1.
    enable_weight_computation (bool, optional): Enable CTDNE weight computation. Required for ExponentialWeight picker. Defaults to False.
    timescale_bound (float, optional): Scale factor for temporal differences. Used to prevent numerical issues with large time differences. Defaults to -1.0.
    walk_padding_value (int, optional): Padding node value for prematurely broken walks. Default is -1.

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Methods

    `add_edges_from_networkx(self: _tempest.Tempest, arg0: object)`
    :   Add edges from a NetworkX graph.
        
        Args:
            nx_graph (networkx.Graph): NetworkX graph with timestamp edge attributes.

    `add_multiple_edges(...)`
    :   add_multiple_edges(self: _tempest.Tempest, sources: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], targets: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], timestamps: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> None
        
        
        Add multiple directed edges to the temporal graph.
        
        This function efficiently handles both Python lists and NumPy arrays without
        unnecessary data copying. The implementation automatically converts the input
        data to the appropriate C++ types.
        
        Args:
            sources: List or NumPy array of source node IDs (or first node in undirected graphs).
            targets: List or NumPy array of target node IDs (or second node in undirected graph).
            timestamps: List or NumPy array of timestamps representing when interactions occurred.
        
        Raises:
            RuntimeError: If arrays are not 1-dimensional or have different lengths.
        
        Note:
            For large datasets, NumPy arrays will provide better performance than Python lists.

    `clear(self: _tempest.Tempest)`
    :   Clears and reinitiates the underlying graph.

    `get_edge_count(self: _tempest.Tempest)`
    :   Returns the total number of directed edges in the temporal graph.
        
        Returns:
           int: The total number of directed edges.

    `get_memory_used(self: _tempest.Tempest)`
    :   Returns the memory used by the application in bytes.
        
        Returns:
            int: The total number of bytes allocated.

    `get_node_count(self: _tempest.Tempest)`
    :   Get total number of nodes in the graph.
        
        Returns:
            int: Number of active nodes.

    `get_node_ids(self: _tempest.Tempest)`
    :   get_node_ids(self: _tempest.Tempest) -> numpy.typing.NDArray[numpy.int32]
        
        
         Returns a NumPy array containing the IDs of all nodes in the temporal graph.
        
        Returns:
            np.ndarray: A NumPy array with all node IDs.

    `get_random_walks_and_times(...)`
    :   get_random_walks_and_times(self: _tempest.Tempest, max_walk_len: typing.SupportsInt, walk_bias: str, num_walks_total: typing.SupportsInt, initial_edge_bias: str | None = None, walk_direction: str = 'Forward_In_Time') -> tuple[numpy.typing.NDArray[numpy.int32], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.uint64]]
        
        
        Generate temporal random walks with timestamps.
        
        Args:
            max_walk_len (int): Maximum length of each random walk.
            walk_bias (str): Type of bias for selecting next node.
                Choices:
                    - "Uniform": Equal probability for all edges
                    - "Linear": Linear time decay
                    - "ExponentialIndex": Exponential decay with indices
                    - "ExponentialWeight": Exponential decay with weights
            num_walks_total (int): Total Number of walks to generate.
            initial_edge_bias (str, optional): Bias type for first edge selection.
                Uses walk_bias if not specified.
            walk_direction (str, optional): Direction of temporal random walks.
                Either "Forward_In_Time" (default) or "Backward_In_Time".
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - 2D array of node ids (shape: [num_walks, max_walk_len])
                - 2D array of timestamps (shape: [num_walks, max_walk_len])
                - 1D array of actual walk lengths (shape: [num_walks])

    `get_random_walks_and_times_for_all_nodes(...)`
    :   get_random_walks_and_times_for_all_nodes(self: _tempest.Tempest, max_walk_len: typing.SupportsInt, walk_bias: str, num_walks_per_node: typing.SupportsInt, initial_edge_bias: str | None = None, walk_direction: str = 'Forward_In_Time') -> tuple[numpy.typing.NDArray[numpy.int32], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.uint64]]
        
        
        Generate temporal random walks with timestamps starting from all nodes.
        
        Args:
            max_walk_len (int): Maximum length of each random walk.
            walk_bias (str): Type of bias for selecting next node.
                Choices:
                    - "Uniform": Equal probability
                    - "Linear": Linear time decay
                    - "ExponentialIndex": Exponential decay with indices
                    - "ExponentialWeight": Exponential decay with weights
            num_walks_per_node (int): Number of walks per starting node.
            initial_edge_bias (str, optional): Bias type for first edge selection.
                Uses walk_bias if not specified.
            walk_direction (str, optional): Direction of temporal random walks.
                Either "Forward_In_Time" (default) or "Backward_In_Time".
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - 2D array of node ids (shape: [num_walks, max_walk_len])
                - 2D array of timestamps (shape: [num_walks, max_walk_len])
                - 1D array of actual walk lengths (shape: [num_walks])

    `to_networkx(self: _tempest.Tempest)`
    :   Export graph to NetworkX format.
        
        Returns:
            networkx.Graph: NetworkX graph with timestamp edge attributes.

`UniformRandomPicker(...)`
:   __init__(self: _tempest.UniformRandomPicker, use_gpu: bool = False) -> None
    
    
    Initialize uniform random picker.
    
    Args:
        use_gpu (bool): Should use GPU or not.

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Methods

    `pick_random(...)`
    :   pick_random(self: _tempest.UniformRandomPicker, start: typing.SupportsInt, end: typing.SupportsInt, prioritize_end: bool = True) -> int
        
        
        Pick random index with uniform probability.
        
        Args:
            start (int): Start index inclusive
            end (int): End index exclusive
            prioritize_end (bool, optional): Prioritize recent timestamps. Default: True
        
        Returns:
            int: Selected index

`WeightBasedRandomPicker(...)`
:   __init__(self: _tempest.WeightBasedRandomPicker, use_gpu: bool = False) -> None
    
    
    Initialize weight-based exponential time decay random picker.
    
    Args:
        use_gpu (bool): Should use GPU or not.

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Methods

    `pick_random(...)`
    :   pick_random(self: _tempest.WeightBasedRandomPicker, cumulative_weights: collections.abc.Sequence[typing.SupportsFloat], group_start: typing.SupportsInt, group_end: typing.SupportsInt) -> int
        
        
        Pick random index with exponential weight-based probability using cumulative weights.
        
        Args:
            cumulative_weights (List[float]): List of cumulative weights
            group_start (int): Start index inclusive
            group_end (int): End index exclusive
        
        Returns:
            int: Selected index