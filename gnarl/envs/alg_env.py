from abc import ABC, abstractmethod
import gymnasium as gym
import torch as th
import numpy as np
from gnarl.envs.generate.graph_generator import GraphGenerator
from gnarl.util.graph_data import GraphProblemData, GraphProblemData_to_dense


class PhasedNodeSelectEnv(gym.Env, ABC):
    """
    Abstract base class for node selection environments.
    """

    @abstractmethod
    def _init_observation_space(
        self,
    ) -> tuple[gym.spaces.Dict, dict[str, tuple[str, str, str]]]:
        """
        Initialize the observation space and the graph features spec for the environment.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _reset_state(self, seed=None, options=None) -> dict:
        """
        Reset the state of the environment and return the info.
        This method should be implemented by subclasses.
        Observations are not returned here.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _get_observation(self) -> dict:
        """
        Get the current observation of the environment.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _step_env(self, action: int) -> dict:
        """
        Perform a step in the environment.
        This method should be implemented by subclasses.
        It should return info.
        The observation and reward are not returned here.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Check if the environment is in a terminal state.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    @abstractmethod
    def get_max_episode_steps(n: int, e: int) -> int:
        """
        Get the maximum number of steps in an episode.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def action_masks(self):
        """
        Get the action masks for the current state.
        This method should be implemented by subclasses.
        It should return a list of boolean masks indicating valid actions.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def objective_function(self, **kwargs) -> float:
        """
        Objective function to maximise. Will be called with the **observation.
        This method must be implemented to enable learning with PPO.
        """
        return 0.0

    @staticmethod
    @abstractmethod
    def expert_policy(obs: dict[str, np.ndarray], *args, **kwargs) -> np.ndarray:
        """
        Expert policy for the environment.
        This method should be implemented by subclasses.
        It should return the action distribution based on the observation.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def is_success(self) -> bool | None:
        """
        Override this method to define the success condition for the environment.
        """
        return None

    def __init__(
        self,
        max_nodes: int,
        num_phases: int,
        graph_generator: GraphGenerator,
        observe_pos: bool = False,
        observe_final_selection: bool = True,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.num_phases = num_phases
        self.observe_pos = observe_pos
        self.observe_final_selection = observe_final_selection
        self.sampler = graph_generator
        self.graph_spec: dict = graph_generator.graph_spec
        self.generator = graph_generator.generate()

        self.current_phase: int | None = None
        self.last_selected: list[int | None] = [None] * num_phases
        self.graph_data: GraphProblemData = None

        self.action_space = gym.spaces.Discrete(max_nodes)

        obs_space, add_spec = self._init_observation_space()
        self.graph_spec = self._add_state_to_spec()
        self.graph_spec.update(add_spec)
        input_spec = self._get_input_spec()
        self.observation_space = gym.spaces.Dict(
            {
                **input_spec,
                **obs_space,
            }
        )

    def _add_state_to_spec(self):
        state_spec = self.graph_spec.copy()
        max_observable_phases = (
            self.num_phases if self.observe_final_selection else self.num_phases - 1
        )
        state_spec.update(
            {
                f"last_selected_{i}": (
                    "state",
                    "node",
                    "categorical",
                    2,
                )  # selected or not
                for i in range(max_observable_phases)
            }
        )
        state_spec.update({"phase": ("state", "graph", "categorical", self.num_phases)})
        return state_spec

    def _get_input_spec(self) -> gym.spaces.Dict:
        obs_space = gym.spaces.Dict()
        for key, info in self.graph_spec.items():
            stage, location, typ = info[:3]
            if stage != "input" and stage != "state":
                continue
            if key == "pos" and not self.observe_pos:
                continue
            if location == "node":
                if typ == "mask" or typ == "pointer" or typ == "mask_one":
                    obs_space[key] = gym.spaces.Box(
                        low=0, high=1, shape=(self.max_nodes,), dtype=np.int32
                    )
                elif typ == "categorical":
                    obs_space[key] = gym.spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.max_nodes,),
                        dtype=np.int64,
                    )
                elif typ == "scalar":
                    obs_space[key] = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.max_nodes,),
                        dtype=np.float64,
                    )
                else:
                    raise ValueError(
                        f"Unsupported input feature type: {typ} for key: {key}"
                    )
            elif location == "graph":
                if typ == "mask" or typ == "mask_one":
                    obs_space[key] = gym.spaces.Box(
                        low=0, high=1, shape=(1,), dtype=np.int32
                    )
                elif typ == "scalar":
                    obs_space[key] = gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64
                    )
                elif typ == "categorical":
                    obs_space[key] = gym.spaces.Box(
                        low=0, high=1, shape=(1,), dtype=np.int64
                    )
                else:
                    raise ValueError(
                        f"Unsupported input feature location: {typ} for key: {key}"
                    )
            elif location == "edge":
                if typ == "mask" or typ == "mask_one":
                    obs_space[key] = gym.spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.max_nodes, self.max_nodes),
                        dtype=np.int32,
                    )
                elif typ == "scalar":
                    obs_space[key] = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.max_nodes, self.max_nodes),
                        dtype=np.float64,
                    )
                elif typ == "categorical":
                    obs_space[key] = gym.spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.max_nodes, self.max_nodes),
                        dtype=np.int64,
                    )
                else:
                    raise ValueError(
                        f"Unsupported input feature type: {typ} for key: {key}"
                    )
            else:
                raise ValueError(
                    f"Unsupported input feature location: {location} for key: {key}"
                )
        return obs_space

    def _get_observation_with_info(self):
        """
        Get the current observation of the environment.
        """
        _cls_obs = self._get_observation()
        input_features = GraphProblemData_to_dense(
            self.graph_data,
            self.graph_spec,
            stage="input",
            obs_space=self.observation_space,
            max_nodes=self.max_nodes,
        )

        # Build last_selected_{i} for each phase
        last_selected_dict = {}
        for i, selected_node in enumerate(self.last_selected):
            if not self.observe_final_selection and i == self.num_phases - 1:
                break
            m = np.zeros((self.max_nodes), dtype=np.int32)
            if selected_node is not None and 0 <= selected_node < self.max_nodes:
                m[selected_node] = 1
            last_selected_dict[f"last_selected_{i}"] = m
        return {
            "phase": np.array([self.current_phase]),
            **last_selected_dict,
            **_cls_obs,
            **input_features,
        }

    @property
    def max_episode_steps(self) -> int:
        """
        Get the maximum number of steps in this episode.
        """
        return self.__class__.get_max_episode_steps(
            self.graph_data.num_nodes, self.graph_data.num_edges
        )

    def get_reward(self, **kwargs) -> float:
        current_obj = self.objective_function(**kwargs)
        obj_diff = current_obj - self.previous_obj
        self.previous_obj = current_obj
        return obj_diff

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """
        Perform a step in the environment.
        """
        if self.current_phase is None:
            raise ValueError("Environment is not in a valid phase.")

        if action < 0 or action >= self.max_nodes:
            raise ValueError(
                f"Invalid action: {action}. Must be between 0 and {self.max_nodes - 1}."
            )

        info = self._step_env(action)

        if self.is_success() is not None:
            info["is_success"] = self.is_success()

        if self.current_phase is not None:
            self.last_selected[self.current_phase] = action
        self.current_phase = (
            (self.current_phase + 1) % self.num_phases
            if self.current_phase is not None
            else None
        )

        # Note that reward/done are determined after the phase is updated.
        obs = self._get_observation_with_info()
        reward = self.get_reward(**obs) if self.current_phase == 0 else 0.0
        done = self.is_terminal() if self.current_phase == 0 else False

        return obs, reward, done, False, info

    @property
    def num_nodes(self) -> int:
        return self.graph_data.num_nodes if self.graph_data else 0

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.sampler.seed(seed)
            self.generator = self.sampler.generate()

        self.graph_data = next(self.generator)
        if self.graph_spec.get("adj", (None, None, None))[0] != "state":
            self.adj = th.sparse_coo_tensor(
                self.graph_data.edge_index,
                th.ones(self.graph_data.edge_index.size(1), dtype=th.bool),
                (self.graph_data.num_nodes, self.graph_data.num_nodes),
            )

        self.current_phase = 0
        self.last_selected = [None] * self.num_phases
        self.previous_obj = 0.0

        info = self._reset_state(seed=seed, options=options)

        return self._get_observation_with_info(), info
