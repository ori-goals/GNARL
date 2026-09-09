from gymnasium import spaces

import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from typing import Callable, Any, Optional

from gnarl.network.gnns import get_network_class
from gnarl.util.graph_format import (
    matrix_features_to_batch,
)


class GraphFeatureTransformer(BaseFeaturesExtractor):
    """
    Processes features in the observation space according to their type.
    Outputs a flattened tensor of features.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        graph_spec: dict,
        node_max: int,
        embed_dim: int = 64,
        weights_init: Optional[Callable] = None,
    ) -> None:
        self.node_max = node_max
        self.embed_dim = embed_dim
        self.graph_spec = graph_spec

        features_dim = 1  # unused
        super().__init__(observation_space, features_dim=features_dim)

        self.encoders = self._construct_encoders(embed_dim)

        # weights initialisation
        if weights_init is not None:
            for key, encoder in self.encoders.items():
                weights_init(encoder, key=key, graph_spec=self.graph_spec)

    def forward(self, observations) -> Batch:
        node_features, edge_features, graph_features, adj_matrix = (
            self._encode_features(observations)
        )
        batch = matrix_features_to_batch(
            node_features, edge_features, graph_features, adj_matrix
        )
        return batch

    def _construct_encoders(self, embed_dim: int) -> nn.ModuleDict:
        """
        Construct the encoders for the node and edge features.
        The encoders are simple linear layers that map the input features to the embedding dimension.
        """
        encoders = {}
        for key, s in self.graph_spec.items():
            typ = s[2]
            if typ == "categorical":
                num_categories = s[3]
                encoders[key] = nn.Linear(num_categories, embed_dim)
            else:
                encoders[key] = nn.Linear(1, embed_dim)

        return nn.ModuleDict(encoders)

    def _encode_features(
        self, observations: dict[str, th.Tensor]
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Convert the observations to node features, edge features, and graph features. Also output the adjacency matrix."""

        batch_size = next(iter(observations.values())).shape[0]

        # Separate the features into node and edge features
        edge_features = {}
        node_features = {}
        graph_features = {}
        adj_matrix = observations["adj"]
        for key, feature in observations.items():
            s = self.graph_spec.get(key, (None, None, None))
            stage, location, typ = s[:3]
            if stage is None or location is None or typ is None:
                raise ValueError(
                    f"Key {key} not found in graph_spec. Please check the graph_spec."
                )
            if key in ["adj", "xc", "yc", "initial_edges", "tau"]:
                continue  # don't encode these features
            elif location == "node":
                if typ == "pointer":
                    # change to one-hot encoding on edges
                    feature = F.one_hot(feature.long(), num_classes=self.node_max)
                    edge_features[key] = self.encoders[key](
                        feature.unsqueeze(-1).to(th.float32)
                    )
                elif typ == "categorical":
                    # encode as one-hot vector
                    feature = F.one_hot(feature.long(), num_classes=s[3])
                    node_features[key] = self.encoders[key](feature.to(th.float32))
                else:
                    node_features[key] = self.encoders[key](
                        feature.unsqueeze(-1).to(th.float32)
                    )

            elif location == "edge":
                if typ == "pointer":
                    raise ValueError("Unsupported edge feature type 'pointer'.")
                elif typ == "categorical":
                    # encode as one-hot vector
                    feature = F.one_hot(feature.long(), num_classes=s[3])
                    edge_features[key] = self.encoders[key](feature.to(th.float32))
                else:
                    edge_features[key] = self.encoders[key](
                        feature.unsqueeze(-1).to(th.float32)
                    )

            elif location == "graph":
                if typ == "pointer":
                    raise ValueError("Unsupported graph feature type 'pointer'.")
                elif typ == "categorical":
                    # encode as one-hot vector
                    feature = F.one_hot(feature.squeeze(-1).long(), num_classes=s[3])
                    graph_features[key] = self.encoders[key](feature.to(th.float32))
                else:
                    graph_features[key] = self.encoders[key](feature.to(th.float32))

            else:
                raise ValueError(
                    f"Unknown location {location} for feature {key}. "
                    "Supported locations are 'node', 'edge', and 'graph'."
                )

        # Accumulate the encoded node features
        if node_features == {}:
            node_features = {
                "node": th.zeros(batch_size, self.node_max, self.embed_dim)
            }
        encoded_node_features = th.mean(
            th.stack(list(node_features.values()), dim=0), dim=0
        )
        # Accumulate the encoded edge features
        if edge_features == {}:
            edge_features = {
                "edge": th.zeros(
                    batch_size,
                    self.node_max,
                    self.node_max,
                    self.embed_dim,
                )
            }
        encoded_edge_features = th.mean(
            th.stack(list(edge_features.values()), dim=0), dim=0
        )
        # Accumulate the encoded graph features
        if graph_features == {}:
            graph_features = {"graph": th.zeros(batch_size, self.embed_dim)}
        encoded_graph_features = th.mean(
            th.stack(list(graph_features.values()), dim=0), dim=0
        )

        return (
            encoded_node_features,
            encoded_edge_features,
            encoded_graph_features,
            adj_matrix,
        )


class GraphFeatureEncoderProcessor(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor
    :param last_layer_dim_pi: dimension of the last layer of the policy network (must be the same as the action space)
    :param last_layer_dim_vf: dimension of the last layer of the value network
    """

    def __init__(
        self,
        embed_dim: int = 64,
        pooling_type: str = "max",
        weights_init: Optional[Callable] = None,
        output_features: list[str] = ["node"],  # ["node", "edge", "graph", "adj"]
        cat_raw_features: bool = False,
        network_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__()
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_vf = embed_dim
        self.latent_dim_pi = 0  # unused
        self.embed_dim = embed_dim
        self.pooling_type = pooling_type

        self.output_features = output_features
        self.cat_raw_features = cat_raw_features

        if network_kwargs is None:
            network_kwargs = {}

        processor_class = get_network_class(network_kwargs["network"])

        self.processor = processor_class(
            in_channels=embed_dim,
            out_channels=embed_dim,
            edge_dim=embed_dim,
            weights_init=weights_init,
            **network_kwargs,
        )

    def _process_graph(
        self,
        batch: Batch,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Process the graph with the graph network"""

        # # Process the graph with the graph network
        node_embedding, edge_embedding = self.processor(
            node_fts=batch.x,
            edge_attr=batch.edge_attr,
            graph_fts=batch.graph_attr,
            edge_index=batch.edge_index,
            batch=batch.batch,
        )

        if self.pooling_type == "max":
            graph_embedding = global_max_pool(node_embedding, batch.batch)
        elif self.pooling_type == "mean":
            graph_embedding = global_mean_pool(node_embedding, batch.batch)
        elif self.pooling_type == "sum":
            graph_embedding = global_add_pool(node_embedding, batch.batch)

        return node_embedding, edge_embedding, graph_embedding

    def forward(self, batch: Batch) -> tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        # Process the graph
        node_embedding, edge_embedding, graph_embedding = self._process_graph(batch)

        processed_batch = Batch(
            x=node_embedding,
            edge_index=batch.edge_index,
            edge_attr=edge_embedding,
            graph_attr=graph_embedding,
            batch=batch.batch,
        )

        return processed_batch, graph_embedding

    def forward_critic(self, x: Batch) -> th.Tensor:
        return self.forward(x)[1]

    def forward_actor(self, x: Batch) -> Batch:
        return self.forward(x)[0]


class NodeSimilarityMatchAgg(nn.Module):
    """
    Action network that uses distance-based matching to select a node.

    :param embed_dim: dimension of the node embeddings
    :param pooling_type: pooling type to use for graph embedding computation
        (options: "max", "mean", "sum")
    :param distance_metric: distance metric to use for similarity computation
    :param temp: temperature parameter for softmax
    """

    def __init__(
        self,
        embed_dim: int,
        max_nodes: int,
        distance_metric: str = "euclidean",
        temp: float = 1.0,
        weights_init: Optional[Callable] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_nodes = max_nodes
        self.distance_metric = distance_metric
        self.linear_out = nn.Linear(self.embed_dim, self.embed_dim)
        self.softmax_temp = nn.Parameter(th.tensor(temp), requires_grad=True)

        if weights_init is not None:
            weights_init(self.linear_out)

        if self.distance_metric not in ["euclidean", "cosine"]:
            raise ValueError(f"Unknown distance metric {self.distance_metric}")

    def compute_embedding_similarities(self, embedded_acts, pn_output):
        if self.distance_metric == "euclidean":
            similarities = -th.cdist(embedded_acts, pn_output, p=2).squeeze(-1)

        elif self.distance_metric == "cosine":
            similarities = F.cosine_similarity(embedded_acts, pn_output, dim=-1)
        else:
            raise ValueError(f"unknown distance metric {self.distance_metric}")

        return similarities

    def forward(self, batch: Batch) -> th.Tensor:
        """Forward pass of the action network.
        This method takes the concatenated features from the feature extractor and computes the similarities
        between the graph embedding and the node embeddings.
        The input must be a flattened tensor of node embeddings and the graph embedding.

        Args:
            x (th.Tensor): (b, n*f + f) matrix of node features and the graph embedding
              where n is the number of nodes and f is the embedding dimension.

        Returns:
            th.Tensor: (b, n) matrix of similarities between the graph embedding and the node embeddings
        """

        node_embeddings = batch.x
        graph_embedding = batch.graph_attr

        pn_output = self.linear_out(graph_embedding)

        # Compute similarities between the graph embedding and the node embeddings per batch
        similarities = th.zeros_like(batch.batch, dtype=th.float32)
        unique_batches = batch.batch.unique()
        for batch_id in unique_batches:
            batch_mask = batch.batch == batch_id
            batch_embeddings = node_embeddings[batch_mask]
            batch_target = pn_output[batch_id].unsqueeze(0)  # Target for this batch
            batch_similarities = self.compute_embedding_similarities(
                batch_embeddings, batch_target
            )
            similarities[batch_mask] = batch_similarities

        similarities = similarities / self.softmax_temp

        # Reshape similarities along the batch dimension
        similarities = to_dense_batch(
            similarities.unsqueeze(-1),
            batch.batch,
            fill_value=-1e9,
            max_num_nodes=self.max_nodes,
        )[0]

        return similarities


class MaskableNodeActorCriticPolicy(MaskableActorCriticPolicy):
    """
    Custom Actor-Critic Policy with a custom feature extractor and network architecture.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Callable[[float], float],
        graph_spec: dict,
        embed_dim: int = 64,
        pooling_type: str = "max",
        distance_metric: str = "euclidean",
        temp: float = 1.0,
        weights_init: Optional[Callable] = None,
        network_kwargs: dict = None,
        *args,
        **kwargs,
    ):
        self.embed_dim = embed_dim
        self.pooling_type = pooling_type
        self.distance_metric = distance_metric
        self.temp = temp
        self.weights_init = weights_init
        self.network_kwargs = network_kwargs
        self.graph_spec = {
            k: v for k, v in graph_spec.items() if v[0] in ["input", "state"]
        }

        self.extractor_args = {
            "output_features": ["node", "graph"],
            "cat_raw_features": False,
        }

        kwargs.setdefault("features_extractor_class", GraphFeatureTransformer)
        features_extractor_kwargs = kwargs.setdefault("features_extractor_kwargs", {})
        features_extractor_kwargs.update(
            {
                "embed_dim": embed_dim,
                "weights_init": self.weights_init,
                "graph_spec": self.graph_spec,
                "node_max": action_space.n,
            }
        )

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        # override action net to use distance-based matching
        self.action_net = NodeSimilarityMatchAgg(
            embed_dim=self.embed_dim,
            max_nodes=self.action_space.n,
            distance_metric=self.distance_metric,
            temp=self.temp,
            weights_init=self.weights_init,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GraphFeatureEncoderProcessor(
            embed_dim=self.embed_dim,
            pooling_type=self.pooling_type,
            weights_init=self.weights_init,
            network_kwargs=self.network_kwargs,
            **self.extractor_args,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                graph_spec=self.graph_spec,
                embed_dim=self.embed_dim,
                pooling_type=self.pooling_type,
                distance_metric=self.distance_metric,
                temp=self.temp,
                weights_init=self.weights_init,
                network_kwargs=self.network_kwargs,
            )
        )
        return data
