from gymnasium.envs.registration import register

register(
    id="BellmanFord-v2",
    entry_point="gnarl.envs.clrs_envs:BellmanFordEnvV2",
    nondeterministic=False,
)

register(
    id="BFS-v1",
    entry_point="gnarl.envs.clrs_envs:BFSEnv",
    nondeterministic=False,
)

register(
    id="DFS-v1",
    entry_point="gnarl.envs.clrs_envs:DFSEnv",
    nondeterministic=False,
)

register(
    id="MSTPrim-v1",
    entry_point="gnarl.envs.clrs_envs:MSTPrimEnv",
    nondeterministic=False,
)

register(
    id="MSTKruskal-v1",
    entry_point="gnarl.envs.clrs_envs:MSTKruskalEnv",
    nondeterministic=False,
)

register(
    id="TSP-v1",
    entry_point="gnarl.envs.np_envs:TSPEnv",
    nondeterministic=False,
)

register(
    id="MVC-v1",
    entry_point="gnarl.envs.np_envs:MVCEnv",
    nondeterministic=False,
)

register(
    id="RobustConstruction-v1",
    entry_point="gnarl.envs.np_envs:RobustConstructionEnv",
    nondeterministic=True,
)

ENV_MAPPING = {
    "bfs": "BFS-v1",
    "bellman_ford": "BellmanFord-v2",
    "dfs": "DFS-v1",
    "mst_prim": "MSTPrim-v1",
    "mst_kruskal": "MSTKruskal-v1",
    "tsp": "TSP-v1",
    "mvc": "MVC-v1",
    "robust_construction": "RobustConstruction-v1",
}
