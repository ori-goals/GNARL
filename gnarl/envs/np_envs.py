import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gnarl.envs.alg_env import PhasedNodeSelectEnv
from gnarl.envs.generate.graph_generator import GraphGenerator
import torch as th
from gnarl.util.mvc import min_weighted_vertex_cover, min_weighted_vertex_cover_approx
from third_party.concorde.concorde_wrapper import solve_tsp_with_concorde
from gnarl.util.graph_format import unpad_array
from torch_geometric.utils import to_dense_adj
from third_party.relnet.objective_functions.objective_functions import (
    CriticalFractionRandom,
    CriticalFractionTargeted,
)
from gnarl.util.algorithms import pad_to_max_nodes
from gnarl.util.graph_data import GraphProblemData


class TSPEnv(PhasedNodeSelectEnv):
    """
    A custom Gymnasium environment for simulating the TSP.
    """

    def __init__(
        self,
        max_nodes: int,
        graph_generator: GraphGenerator,
        **kwargs,
    ):
        super(TSPEnv, self).__init__(
            max_nodes=max_nodes,
            num_phases=1,
            graph_generator=graph_generator,
        )

    def _init_observation_space(
        self,
    ) -> tuple[gym.spaces.Dict, dict[str, tuple[str, str, str]]]:
        obs_space = spaces.Dict(
            {
                # outputs
                "next_node": spaces.Box(
                    low=0,
                    high=self.max_nodes - 1,
                    shape=(self.max_nodes,),
                    dtype=np.int32,
                ),
                # state
                "in_tour": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_nodes,),
                    dtype=np.int32,
                ),
            }
        )
        spec = {
            "next_node": ("state", "node", "pointer"),
            "in_tour": ("state", "node", "mask"),
        }
        return obs_space, spec

    def _get_observation(self) -> dict:
        return {
            "next_node": pad_to_max_nodes(self.next_nodes, self.max_nodes),
            "in_tour": pad_to_max_nodes(self.in_tour, self.max_nodes),
        }

    def _reset_state(self, seed=None, options=None):
        # Initialise state variables
        self.next_nodes = np.arange(self.graph_data.num_nodes, dtype=np.int32)
        self.in_tour = np.zeros(self.graph_data.num_nodes, dtype=np.int32)

        self.weights = th.sparse_coo_tensor(
            self.graph_data.edge_index,
            self.graph_data.A,
            (self.graph_data.num_nodes, self.graph_data.num_nodes),
        )
        return {}

    def get_max_episode_steps(n: int, e: int) -> int:
        return n

    def is_terminal(self) -> bool:
        done = self.in_tour.all()
        if done and set(self.next_nodes) != set(range(self.graph_data.num_nodes)):
            raise ValueError(
                f"Environment is not in a valid state, all nodes should be in the tour {self.next_nodes}."
            )
        return done

    def _step_env(self, action: int) -> dict:
        new_node = action
        self.in_tour[new_node] = 1

        prev_node = self.last_selected[0]

        if prev_node is not None:
            self.next_nodes[new_node] = self.next_nodes[prev_node]
            self.next_nodes[prev_node] = new_node

        return {}

    @staticmethod
    def _objective_function(**kwargs) -> float:
        path_sum = 0.0
        for i, next_node in enumerate(kwargs["next_node"]):
            if not kwargs["in_tour"][i] or next_node == i:
                continue
            path_sum += kwargs["A"][i, next_node].item()
        return -path_sum

    def objective_function(self, **kwargs) -> float:
        return self._objective_function(**kwargs)

    def action_masks(self):
        if self.current_phase == 0:
            if self.last_selected[0] is None:
                return pad_to_max_nodes(self.graph_data.s.numpy(), self.max_nodes)
            return pad_to_max_nodes(self.in_tour == 0, self.max_nodes)
        else:
            raise ValueError("Invalid phase for action masks.")

    def render(self, mode="human"):
        print("Next nodes:")
        print(self.next_nodes)
        print("In tour:")
        print(self.in_tour)

    @staticmethod
    def _expert_policy_strong(o: dict[str, np.ndarray], i) -> np.ndarray:
        if np.all(o["last_selected_0"][i] == 0):
            return np.array([np.argmax(o["s"][i])])

        max_nodes = unpad_array(o["adj"][i]).shape[0]
        solution = solve_tsp_with_concorde(
            np.stack(
                [
                    1000 * o["xc"][i][:max_nodes],
                    1000 * o["yc"][i][:max_nodes],
                ],
                axis=-1,
            )
        )
        next_node = solution[np.argmax(o["last_selected_0"][i])]
        return np.array([next_node])

    @staticmethod
    def _expert_policy_greedy(o: dict[str, np.ndarray], i) -> np.ndarray:
        if np.all(o["last_selected_0"][i] == 0):
            return np.array([np.argmax(o["s"][i])])

        max_nodes = unpad_array(o["adj"][i]).shape[0]
        distances_from_last_node = o["A"][i][np.argmax(o["last_selected_0"][i])][
            :max_nodes
        ]
        candidate_nodes = (
            distances_from_last_node + np.inf * o["in_tour"][i][:max_nodes]
        )
        next_node = np.argmin(candidate_nodes)
        return np.array([next_node])

    @staticmethod
    def expert_policy(obs: dict[str, np.ndarray], *args, **kwargs) -> np.ndarray:
        """Expert policy for the TSP environment.

        Returns action demonstrations rather than probabilities.
        """

        to_unnest = False
        if len(obs["last_selected_0"].shape) < 2:
            to_unnest = True
        for key, value in obs.items():
            value = np.expand_dims(value, axis=0)

        def single_policy(o, i):
            if kwargs.get("method", "concorde") == "concorde":
                return TSPEnv._expert_policy_strong(o, i)
            elif kwargs.get("method") == "greedy":
                return TSPEnv._expert_policy_greedy(o, i)

        actions = np.array(
            [single_policy(obs, i) for i in range(len(obs["last_selected_0"]))]
        )
        if to_unnest:
            return actions.reshape(-1, len(actions[0]))

        return actions

    @staticmethod
    def pre_transform(data: GraphProblemData) -> GraphProblemData:
        """Add the expert policy value to the data."""
        solution = solve_tsp_with_concorde(
            np.stack([1000 * data.xc, 1000 * data.yc], axis=-1)
        )
        obj = TSPEnv._objective_function(
            next_node=solution,
            in_tour=np.ones(data.num_nodes, dtype=np.int32),
            A=to_dense_adj(data.edge_index, edge_attr=data.A)[0].numpy(),
        )
        data.expert_objective = th.tensor(obj, dtype=th.float32)
        return data


class MVCEnv(PhasedNodeSelectEnv):
    """
    A custom Gymnasium environment for simulating the Minimum Vertex Cover problem.
    """

    def __init__(
        self,
        max_nodes: int,
        graph_generator: GraphGenerator,
        **kwargs,
    ):
        super().__init__(
            max_nodes=max_nodes,
            num_phases=1,
            graph_generator=graph_generator,
            observe_final_selection=False,
        )

    def _init_observation_space(
        self,
    ) -> tuple[gym.spaces.Dict, dict[str, tuple[str, str, str]]]:
        obs_space = spaces.Dict(
            {
                # state
                "cover_weight": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "covered_proportion": spaces.Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "covered_edges": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_nodes, self.max_nodes),
                    dtype=np.int32,
                ),
                # outputs
                "in_cover": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_nodes,),
                    dtype=np.int32,
                ),
            }
        )
        spec = {
            "in_cover": ("state", "node", "mask"),
            "cover_weight": ("state", "graph", "scalar"),
            "covered_proportion": ("state", "graph", "scalar"),
            "covered_edges": ("state", "edge", "mask"),
        }
        return obs_space, spec

    def _get_covered_edges(self) -> np.ndarray:
        chosen = np.where(self.in_cover == 1)[0]
        edge_mask = np.isin(self.graph_data.edge_index[0].numpy(), chosen) | np.isin(
            self.graph_data.edge_index[1].numpy(), chosen
        )
        return edge_mask

    def _get_observation(self) -> dict:
        covered_edges = th.Tensor(self._get_covered_edges())
        return {
            "in_cover": pad_to_max_nodes(self.in_cover, self.max_nodes),
            "cover_weight": np.array(
                [np.sum(self.in_cover * self.graph_data.nw.numpy())], dtype=np.float32
            ),
            "covered_proportion": np.array(
                [covered_edges.mean()],
                dtype=np.float32,
            ),
            "covered_edges": to_dense_adj(
                self.graph_data.edge_index,
                max_num_nodes=self.graph_data.num_nodes,
                edge_attr=covered_edges,
            )
            .numpy()[0]
            .astype(np.int32),
        }

    def _reset_state(self, seed=None, options=None):
        # Initialise state variables
        self.in_cover = np.zeros(self.graph_data.num_nodes, dtype=np.int32)
        return {}

    def get_max_episode_steps(n: int, e: int) -> int:
        return n

    def is_terminal(self) -> bool:
        return bool(self._get_covered_edges().all())

    def _step_env(self, action: int) -> dict:
        new_node = action
        self.in_cover[new_node] = 1
        return {}

    def objective_function(self, **kwargs) -> float:
        return -sum(kwargs["in_cover"] * kwargs["nw"])

    def action_masks(self):
        return pad_to_max_nodes(self.in_cover == 0, self.max_nodes)

    def render(self, mode="human"):
        print("In cover:")
        print(self.in_cover)

    @staticmethod
    def _expert_policy_probabilities(
        o: dict[str, np.ndarray], i: int, **kwargs
    ) -> np.ndarray:
        """Get the expert policy probabilities for a single observation."""
        edge_index = np.where(o["adj"][i] == 1)
        max_nodes = np.max(edge_index) + 1
        if kwargs.get("solver", "exact") == "exact":
            obj, solution = min_weighted_vertex_cover(
                edge_index, o["nw"][i][:max_nodes]
            )
        elif kwargs.get("solver") == "approx":
            obj, solution = min_weighted_vertex_cover_approx(
                edge_index, o["nw"][i][:max_nodes]
            )

        solution_mask = np.zeros((max_nodes,), dtype=np.int32)
        solution_mask[solution] = 1

        to_select = solution_mask - o["in_cover"][i][:max_nodes]
        if to_select.sum() == 0:
            raise ValueError("No valid nodes to select for the vertex cover.")

        weighting = kwargs.get("weighting", "uniform")
        if weighting == "uniform":
            node_probs = np.zeros_like(o["nw"][i], dtype=np.float32)
            node_probs[: to_select.shape[0]] = to_select
        elif weighting == "node_weights":
            node_weights = o["nw"][i][:max_nodes] * to_select
            node_probs = 1 - node_weights
            node_probs[to_select == 0] = 0
        elif weighting == "degree":
            degrees = o["adj"][i][:max_nodes].sum(axis=1) * to_select
            node_probs = degrees.astype(np.float32)
        elif weighting == "weighted_degree":
            degrees = o["adj"][i][:max_nodes].sum(axis=1) * to_select
            node_weights = o["nw"][i][:max_nodes]
            inverse_weights = (1 - node_weights) * to_select
            node_probs = degrees * inverse_weights
        else:
            raise ValueError(f"Unknown weighting method: {weighting}")

        node_probs /= node_probs.sum()
        return node_probs

    @staticmethod
    def _expert_policy_demonstrations(
        obs: dict[str, np.ndarray], i: int, **kwargs
    ) -> np.ndarray:
        probs = MVCEnv._expert_policy_probabilities(obs, i, **kwargs)
        if kwargs.get("sampling", "stochastic") == "greedy":
            return np.array([np.argmax(probs)])
        return np.array([np.random.choice(len(probs), p=probs)])

    @staticmethod
    def expert_policy(obs: dict[str, np.ndarray], *args, **kwargs) -> np.ndarray:
        """Expert policy for the MVC environment."""

        def single_policy(o, i):
            method = kwargs.get("method", "probabilities")
            if method == "probabilities":
                return MVCEnv._expert_policy_probabilities(o, i, **kwargs)
            elif method == "demonstrations":
                return MVCEnv._expert_policy_demonstrations(o, i, **kwargs)

        actions = np.array([single_policy(obs, i) for i in range(len(obs["nw"]))])

        return actions

    @staticmethod
    def pre_transform(data: GraphProblemData) -> GraphProblemData:
        """Add the APPROXIMATE expert policy value to the data."""
        edge_index = data.edge_index.numpy()
        obj, _ = min_weighted_vertex_cover_approx(edge_index, data.nw)
        data.expert_objective = th.tensor(-obj, dtype=th.float64)
        return data


class RobustConstructionEnv(PhasedNodeSelectEnv):
    def __init__(
        self,
        max_nodes: int,
        graph_generator: GraphGenerator,
        objective_method: str = "random",
        **kwargs,
    ):
        super().__init__(
            max_nodes=max_nodes,
            num_phases=2,
            graph_generator=graph_generator,
            observe_final_selection=False,
        )
        self.objective_method = objective_method
        if self.objective_method not in ["random", "targeted"]:
            raise ValueError(
                f"Unknown objective method: {self.objective_method}. "
                "Supported methods are 'random' and 'targeted'."
            )

    def _init_observation_space(
        self,
    ) -> tuple[gym.spaces.Dict, dict[str, tuple[str, str, str]]]:
        obs_space = spaces.Dict(
            {
                # outputs
                "adj": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_nodes, self.max_nodes),
                    dtype=np.int32,
                ),
                "remaining_budget": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )
        spec = {
            "adj": ("state", "edge", "mask"),
            "remaining_budget": ("state", "graph", "scalar"),
        }
        return obs_space, spec

    def _calculate_reminaining_edges(self) -> np.ndarray:
        available_edges = 1 - self.adj
        np.fill_diagonal(available_edges, 0)
        return available_edges

    def _get_observation(self) -> dict:
        max_edges = (self.num_nodes * (self.num_nodes - 1)) / 2
        remaining_budget_proportion = self.remaining_budget / max_edges
        return {
            "adj": self.adj,
            "remaining_budget": np.array(
                [remaining_budget_proportion], dtype=np.float32
            ),
        }

    def _reset_state(self, seed=None, options=None):
        # Initialise state variables
        self.adj = to_dense_adj(
            self.graph_data.edge_index,
            edge_attr=self.graph_data.initial_edges,
            max_num_nodes=self.max_nodes,
        ).numpy()[0]
        max_edges = (self.num_nodes * (self.num_nodes - 1)) / 2
        self.remaining_budget = int(max(1, np.ceil(self.graph_data.tau * max_edges)))
        self.start_budget = self.remaining_budget

        self.previous_obj = self.objective_function(adj=self.adj)
        return {}

    def get_max_episode_steps(n: int, e: int) -> int:
        return 2 * ((n * (n - 1)) // 2)  # Worst case

    def is_terminal(self) -> bool:
        done = self.remaining_budget <= 0
        if done:
            assert (
                self.adj.sum() / 2 - self.graph_data.initial_edges.sum() / 2
                == self.start_budget
            )
        return done

    def _step_env(self, action: int) -> dict:
        if self.current_phase != 1:
            return {}

        u = self.last_selected[0]
        v = action
        self.adj[u, v] = 1
        self.adj[v, u] = 1

        self.remaining_budget -= 1

        return {}

    @staticmethod
    def _objective_function(**kwargs) -> float:
        current_graph_state = kwargs["adj"]
        uni_edges = np.triu(current_graph_state, k=1)
        edge_index = np.where(uni_edges == 1)
        edge_pairs = np.stack(edge_index, axis=-1)
        num_nodes = int(np.max(edge_pairs) + 1)
        num_edges = len(edge_pairs)

        if kwargs.get("objective") == "random":
            return CriticalFractionRandom().compute(
                num_nodes,
                num_edges,
                edge_pairs.flatten().astype(np.int32),
                num_mc_sims=num_nodes * 2,
                **kwargs,
            )
        elif kwargs.get("objective") == "targeted":
            return CriticalFractionTargeted().compute(
                num_nodes,
                num_edges,
                edge_pairs.flatten().astype(np.int32),
                num_mc_sims=num_nodes * 2,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown objective method: {kwargs.get('objective')}")

    def objective_function(self, **kwargs) -> float:
        return RobustConstructionEnv._objective_function(
            objective=self.objective_method, **kwargs
        )

    def action_masks(self):
        available_edges = self._calculate_reminaining_edges()
        if self.current_phase == 0:
            node_mask = (available_edges.sum(axis=1) > 0).astype(np.int32)
            return pad_to_max_nodes(node_mask, self.max_nodes)
        elif self.current_phase == 1:
            return pad_to_max_nodes(
                available_edges[self.last_selected[0]], self.max_nodes
            )
        raise ValueError("Invalid phase for action masks.")

    def render(self, mode="human"):
        print("In graph:")
        print(self.adj)
        print(f"Remaining budget: {self.remaining_budget}")

    @staticmethod
    def _greedy_expert_policy(obs: dict[str, np.ndarray], **kwargs) -> np.ndarray:
        """Greedy expert policy for the Robust Construction environment."""
        to_unnest = False
        if len(obs["last_selected_0"].shape) < 2:
            to_unnest = True
        for key, value in obs.items():
            value = np.expand_dims(value, axis=0)

        def phase_1_policy(o, i):
            edge_improvement = np.zeros_like(o["adj"][i], dtype=np.float32)
            for j in range(o["adj"][i].shape[0]):
                for k in range(o["adj"][i].shape[1]):
                    if j == k:
                        continue
                    if o["adj"][i][j, k] == 0:
                        new_adj = o["adj"][i].copy()
                        new_adj[j, k] = 1
                        new_adj[k, j] = 1
                        edge_improvement[j, k] = (
                            RobustConstructionEnv._objective_function(
                                adj=new_adj,
                                objective=kwargs.get("objective", "random"),
                                random_seed=41,  # different
                            )
                        )
            edge_id = np.argmax(edge_improvement)
            next_node = np.unravel_index(edge_id, edge_improvement.shape)[0]
            return [next_node]

        def phase_2_policy(o, i):
            j = np.argmax(o["last_selected_0"][i])
            edge_improvement = (
                np.ones_like(o["last_selected_0"][i], dtype=np.float32) * -np.inf
            )
            for k in range(o["adj"][i].shape[1]):
                if k == j:
                    continue
                if o["adj"][i][j, k] == 0:
                    new_adj = o["adj"][i].copy()
                    new_adj[j, k] = 1
                    new_adj[k, j] = 1
                    edge_improvement[k] = RobustConstructionEnv._objective_function(
                        adj=new_adj,
                        objective=kwargs.get("objective", "random"),
                        random_seed=41,  # different
                    )
            next_node = np.argmax(edge_improvement)
            return [next_node]

        def single_policy(o, i):
            if o["phase"][i] == 0:  # phase 1
                return phase_1_policy(o, i)

            elif o["phase"][i] == 1:  # phase 2
                return phase_2_policy(o, i)

            else:
                raise ValueError("Invalid phase for action selection.")

        actions = np.array(
            [single_policy(obs, i) for i in range(len(obs["last_selected_0"]))]
        )
        if to_unnest:
            return actions.reshape(-1, len(actions[0]))

        return actions

    @staticmethod
    def expert_policy(obs: dict[str, np.ndarray], *args, **kwargs) -> np.ndarray:
        if kwargs.get("method") == "greedy":
            return RobustConstructionEnv._greedy_expert_policy(obs, **kwargs)

    @staticmethod
    def pre_transform(data: GraphProblemData) -> GraphProblemData:
        """Add the expert policy value to the data."""
        current_graph_state = data.initial_edges
        uni_edges = np.triu(current_graph_state, k=1)
        edge_index = np.where(uni_edges == 1)
        edge_pairs = np.stack(edge_index, axis=-1)
        num_nodes = int(np.max(edge_pairs) + 1)
        num_edges = len(edge_pairs)
        random = CriticalFractionRandom().compute(
            num_nodes,
            num_edges,
            edge_pairs.flatten().astype(np.int32),
            num_mc_sims=num_nodes * 2,
        )
        targeted = CriticalFractionTargeted().compute(
            num_nodes,
            num_edges,
            edge_pairs.flatten().astype(np.int32),
            num_mc_sims=num_nodes * 2,
        )
        data.init_random_removal = th.tensor(random, dtype=th.float32)
        data.init_targeted_removal = th.tensor(targeted, dtype=th.float32)
        return data
