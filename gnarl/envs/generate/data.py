from torch_geometric.data import Dataset
import os.path as osp
from tqdm import tqdm
import torch as th
from .sampler import build_sampler

from gnarl.util.classes import dict2string


class GraphProblemDataset(Dataset):

    def __init__(
        self,
        root: str,
        split: str,  # train, val, or test
        algorithm: str,
        num_nodes: int,
        num_samples: int,
        seed: int,
        graph_generator: str,
        graph_generator_kwargs: dict | None = None,
        **kwargs,
    ):
        self.split = split
        self.algorithm = algorithm
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.seed = seed
        self.graph_generator = graph_generator
        self.graph_generator_kwargs = graph_generator_kwargs or {}

        name = f"{graph_generator}_{dict2string(graph_generator_kwargs)}_seed={seed}"
        root = osp.join(root, algorithm, name)

        self.sampler, self.specs = build_sampler(
            self.algorithm,
            self.seed,
            self.num_nodes,
            self.graph_generator,
            self.graph_generator_kwargs,
        )

        super().__init__(root, **kwargs)

    @property
    def processed_file_names(self):
        return [f"data_{i}.pt" for i in range(self.num_samples)]

    @property
    def processed_dir(self):
        return osp.join(
            self.root, f"num_nodes_{self.num_nodes}", "processed", self.split
        )

    def process(self):
        """Process the raw graph data into the final format."""

        pbar = tqdm(range(self.num_samples))
        i = 0
        while i < self.num_samples:
            data = self.sampler.next()

            if self.pre_filter and not self.pre_filter(data):  # filter out
                continue

            processed = self.pre_transform(data) if self.pre_transform else data
            th.save(processed, osp.join(self.processed_dir, f"data_{i}.pt"))
            pbar.update(1)
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        d = th.load(osp.join(self.processed_dir, f"data_{idx}.pt"), weights_only=False)
        if self.transform is not None:
            return self.transform(d)
        return d
