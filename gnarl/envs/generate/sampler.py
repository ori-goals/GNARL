import abc
from typing import Any, Optional
import numpy as np
import networkx as nx
from clrs._src import specs
from .specs import SPECS
from gnarl.util.classes import get_clean_kwargs
from gnarl.util.graph_data import GraphProblemData, map_data_to_inputs


class Sampler(abc.ABC):

    def __init__(
        self,
        spec: specs.Spec,
        seed: int,
        num_nodes: int,
        graph_generator: Optional[str] = None,
        graph_generator_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """Initializes a `Sampler`.

        Args:
          spec: The algorithm spec.
          seed: RNG seed.
        """

        # Use `RandomState` to ensure deterministic sampling across Numpy versions.
        self._rng = np.random.RandomState(seed)
        self._num_nodes = num_nodes
        self._graph_generator = graph_generator
        self._graph_generator_kwargs = graph_generator_kwargs or {}
        self._spec = spec
        self._kwargs = kwargs

    def next(self) -> GraphProblemData:
        data = self._sample_data(**self._kwargs)
        return map_data_to_inputs(data, self._spec)

    def _create_graph(
        self, n, weighted, directed, low=0.0, high=1.0, **kwargs
    ) -> dict[str, np.ndarray]:
        """Create graph."""
        graph = {}
        if self._graph_generator is None or self._graph_generator == "er":
            graph["adj"] = self._random_er_graph(n=n, **kwargs)
        elif self._graph_generator == "ws":
            assert not directed, "Directed graphs not supported yet."
            graph["adj"] = self._watt_strogatz_graph(n=n, **kwargs)
        elif self._graph_generator == "complete":
            graph["adj"] = self._complete_graph(n=n, **kwargs)
        elif self._graph_generator == "ba":
            assert not directed, "Directed graphs not supported yet."
            graph["adj"] = self._barabasi_albert_graph(n=n, **kwargs)
        elif self._graph_generator == "coordinate":
            assert not directed, "Directed graphs not supported yet."
            mat, coordinates = self._coordinate_graph(n=n, **kwargs)
            graph["adj"] = mat
            graph["xc"] = coordinates[:, 0]
            graph["yc"] = coordinates[:, 1]
            if weighted:
                weights = np.array(
                    [
                        [
                            np.linalg.norm(coordinates[i] - coordinates[j])
                            for j in range(mat.shape[0])
                        ]
                        for i in range(mat.shape[1])
                    ]
                )
                weights = weights / np.max(weights)  # Normalize weights to [0, 1]
                graph["A"] = mat.astype(float) * weights
        else:
            raise ValueError(f"Unknown graph generator {self._graph_generator}.")
        n = graph["adj"].shape[0]
        if weighted and "A" not in graph:
            weights = -self._rng.uniform(low=-high, high=-low, size=(n, n))
            if not directed:
                weights *= np.transpose(weights)
                weights = np.sqrt(weights + 1e-3)  # Add epsilon to protect underflow
            weights = graph["adj"].astype(float) * weights
            graph["A"] = weights
        elif not weighted and "A" not in graph and "A" in self._spec:
            graph["A"] = graph["adj"].astype(float)
        if kwargs.get("node_weights", False):
            graph["nw"] = -self._rng.uniform(low=-high, high=-low, size=n)
        return graph

    @abc.abstractmethod
    def _sample_data(self, *args, **kwargs) -> dict[str, np.ndarray]:
        pass

    def _select_parameter(self, parameter, parameter_range=None, integer=False):
        if parameter_range is not None:
            assert len(parameter_range) == 2
            if integer:
                return self._rng.randint(*parameter_range)
            else:
                return self._rng.uniform(*parameter_range)
        if isinstance(parameter, list) or isinstance(parameter, tuple):
            return self._rng.choice(parameter)
        else:
            return parameter

    def _random_er_graph(
        self,
        n,
        p=None,
        p_range=None,
        directed=False,
        connected=True,
        *args,
        **kwargs,
    ):
        """Random Erdos-Renyi graph."""
        p = self._select_parameter(p, p_range)

        while True:
            g = nx.erdos_renyi_graph(n, p, directed=directed)
            if connected:
                # ensure that the graph is connected
                if not nx.is_connected(g):
                    continue
            return nx.to_numpy_array(g)

    def _watt_strogatz_graph(self, n, k, *args, p=None, p_range=None, **kwargs):
        """Watts-Strogatz graph."""
        k = self._select_parameter(k)
        p = self._select_parameter(p, p_range)
        g = nx.connected_watts_strogatz_graph(n, k, p)
        mat = nx.to_numpy_array(g)
        return mat

    def _complete_graph(self, n, *args, **kwargs):
        """Complete graph."""
        mat = np.ones((n, n))
        return mat

    def _coordinate_graph(self, n, *args, **kwargs):
        """Coordinate graph."""
        coordinates = self._rng.randint(0, 1001, size=(n, 2)) / 1000.0
        mat = np.ones((n, n))
        return mat, coordinates

    def _barabasi_albert_graph(self, n, M=None, M_range=None, *args, **kwargs):
        """Barabasi-Albert graph."""
        M = self._select_parameter(M, M_range, integer=True)
        g = nx.barabasi_albert_graph(n, M)
        mat = nx.to_numpy_array(g)
        return mat


class DfsSampler(Sampler):
    """DFS sampler."""

    def _sample_data(self):
        graph_data = self._create_graph(
            self._num_nodes,
            directed=True,
            acyclic=False,
            weighted=False,
            **self._graph_generator_kwargs,
        )
        return graph_data


class BfsSampler(Sampler):
    """BFS sampler."""

    def _sample_data(self):
        graph_data = self._create_graph(
            self._num_nodes,
            directed=False,
            acyclic=False,
            weighted=False,
            **self._graph_generator_kwargs,
        )
        graph_data["s"] = self._rng.choice(graph_data["adj"].shape[0])
        return graph_data


class BellmanFordSampler(Sampler):
    """Bellman-Ford sampler."""

    def _sample_data(self, low=0.0, high=1.0):
        graph_data = self._create_graph(
            self._num_nodes,
            directed=False,
            acyclic=False,
            weighted=True,
            low=low,
            high=high,
            **self._graph_generator_kwargs,
        )
        graph_data["s"] = self._rng.choice(graph_data["adj"].shape[0])
        return graph_data


class MSTKruskalSampler(Sampler):
    """Kruskal's algorithm sampler."""

    def _sample_data(self, low=0.0, high=1.0):
        graph_data = self._create_graph(
            self._num_nodes,
            directed=False,
            acyclic=False,
            weighted=True,
            low=low,
            high=high,
            **self._graph_generator_kwargs,
        )
        return graph_data


class TspSampler(Sampler):
    """TSP sampler for travelling salesperson problem."""

    def _sample_data(self):
        if self._graph_generator != "coordinate":
            raise ValueError(
                "TSP sampler requires coordinate graph generator. "
                f"Got {self._graph_generator} instead."
            )
        graph_data = self._create_graph(
            self._num_nodes,
            directed=False,
            acyclic=False,
            weighted=True,
            **self._graph_generator_kwargs,
        )
        graph_data["s"] = self._rng.choice(graph_data["adj"].shape[0])
        return graph_data


class MVCSampler(Sampler):
    """MVC sampler for minimum vertex cover."""

    def _sample_data(self):
        graph_data = self._create_graph(
            self._num_nodes,
            directed=False,
            acyclic=False,
            weighted=False,
            node_weights=True,
            **self._graph_generator_kwargs,
        )
        return graph_data


class RobustConstructionSampler(Sampler):
    """Robust construction sampler for robust graph construction."""

    def _sample_data(self):
        graph_data = self._create_graph(
            self._num_nodes,
            directed=False,
            acyclic=False,
            weighted=False,
            **self._graph_generator_kwargs,
        )
        graph_data["initial_edges"] = graph_data["adj"]
        graph_data["tau"] = np.array([self._graph_generator_kwargs.get("tau", 0.05)])
        return graph_data


def build_sampler(
    name: str,
    seed: int,
    num_nodes: int,
    graph_generator: str,
    graph_generator_kwargs: Optional[dict[str, Any]] = None,
    **kwargs,
) -> tuple[Sampler, specs.Spec]:
    if name not in SPECS or name not in SAMPLERS:
        raise NotImplementedError(
            f"No implementation of algorithm {name}. {name in SPECS} in SPECS, {name in SAMPLERS} in SAMPLERS."
        )
    spec = SPECS[name]
    sampler_class = SAMPLERS[name]
    clean_kwargs = get_clean_kwargs(
        sampler_class._sample_data, warn=True, kwargs=kwargs
    )

    sampler = sampler_class(
        spec,
        seed=seed,
        num_nodes=num_nodes,
        graph_generator=graph_generator,
        graph_generator_kwargs=graph_generator_kwargs,
        **clean_kwargs,
    )
    return sampler, spec


SAMPLERS = {
    "dfs": DfsSampler,
    "bfs": BfsSampler,
    "bellman_ford": BellmanFordSampler,
    "mst_prim": BellmanFordSampler,
    "mst_kruskal": MSTKruskalSampler,
    "tsp": TspSampler,
    "mvc": MVCSampler,
    "robust_construction": RobustConstructionSampler,
}
