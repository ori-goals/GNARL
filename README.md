# GNARL
This is the official code repository for [Tackling GNARLy Problems: Graph Neural Algorithmic Reasoning Reimagined through Reinforcement Learning](https://openreview.net/forum?id=3aeMfeKlKh) by [Alex Schutz](https://alex-schutz.github.io/), [Victor-Alexandru Darvariu](https://victor.darvariu.me/), [Efimia Panagiotaki](https://efimiap.github.io/), [Bruno Lacerda](https://bfalacerda.github.io/), and [Nick Hawes](https://www.robots.ox.ac.uk/~nickh/), published in TMLR.
If you use this code, please consider citing our paper:
```
@article{schutz2026tackling,
  author    = {Schutz, Alex and Darvariu, Victor-Alexandru and Panagiotaki, Efimia and Lacerda, Bruno and Hawes, Nick},
  title     = {Tackling GNARLy Problems: Graph Neural Algorithmic Reasoning Reimagined through Reinforcement Learning},
  journal   = {Transactions on Machine Learning Research},
  year      = {2026},
  url       = {https://openreview.net/forum?id=3aeMfeKlKh}
}
```

## Installation

**Prerequisites:**
- Python 3.10 or 3.11
- Boost libraries (for the C++ objective functions, required for robustness calculations)
	- If these are not available, the main GNARL functionality will still work, but the Robust Construction environment will not work


```bash
pip install . 
cd third_party/concorde && ./build_concorde.sh && cd ../..
```

## License

This project is licensed under the MIT License.

It also includes code from [DEAR](https://github.com/HekpoMaH/DEAR), which is licensed under the Apache License 2.0, and [SALSA-CLRS](https://github.com/jkminder/SALSA-CLRS), which is licensed under the Apache License 2.0.
See LICENSE and NOTICE for details.

## Third-Party Solvers

This repo uses the [Concorde TSP Solver](http://www.math.uwaterloo.ca/tsp/concorde.html) as a third-party executable.
See `third_party/concorde/README.md` for build and usage instructions.

## Running Experiments

To run all experiments from the paper, use the `run_experiments.py` script. This will train and evaluate models for all configurations in the `configs` directory.

```bash
python scripts/run_experiments.py
```

For more fine-grained control, see below.

### Generating data
To generate data for a specific problem, run the `generate_data.py` script with the appropriate configuration file. For example, to generate data for the BFS problem:
	
```bash
python scripts/generate_data.py configs/bfs.yaml
```

The results in the paper rely on external datasets for the TSP, MVC, and Robust Graph Construction problems.
- To use the TSP data, generate the `tsp_large` dataset [here](https://github.com/danilonumeroso/conar), then place it in the `data/tsp_large` directory.
Run `python scripts/generate_data.py configs/tsp.yaml` to convert the data into the required format.
- To use the MVC data, generate the "vertex_cover" dataset from [here](https://github.com/dransyhe/pdnar), and place it in the `data/mvc` directory.
Run `python scripts/generate_data.py configs/mvc.yaml` to convert the data into the required format.
- To use the RGC data, generate the dataset from [here](https://github.com/VictorDarvariu/graph-construction-rl-lite/tree/98aff28856a75b8f4016a315af47a5c84c7c94c4), then place it in the `data/graph-construction-datasets` directory.
Run `python scripts/generate_data.py configs/rgc_er_r.yaml` to convert the data into the required format.


### Training a model

To train a model, run the `train.py` script with the appropriate configuration file. For example, to train a model for the BFS problem using BC:
	
```bash
python scripts/train.py --config configs/bfs.yaml
```

Use the optional `-w` flag to enable Weights & Biases logging.

### Evaluating a model

After training a model, you can evaluate it using the `eval.py` script. For example, to evaluate a trained model for the BFS problem:
```bash
python scripts/eval.py --config configs/bfs.yaml --path path/to/trained/model.pth
```
You will find some pre-trained models in the `checkpoints` directory.


## Adding a New Problem Type

To solve a new problem type using GNARL, implement the following:

1. Add a new environment in `gnarl/envs/` which inherits from the `PhasedNodeSelectEnv`. 
You must implement the following methods:
    - `_init_observation_space` — define the state variables
	- `_reset_state` — initialise the state variables
	- `_get_observation` — implement the observation function for the state variables
	- `_step_env` — implement the transition function
	- `get_max_episode_steps` — define the maximum number of steps for the episode
	- `action_masks` — define the valid actions for the state
	- `is_terminal` — define the terminal states

    **BC**:
	- `expert_policy` — implement the expert for imitation learning
	- `is_success` (optional) — determine if the algorithm was completed successfully

    **PPO**
    - `objective_function` — define the objective function to be maximised
	- `pre_transform` (optional) — add the expert objective function value to the dataset for comparison at test time

1. Register the new environment in `gnarl/envs/__init__.py` and add it to the `ENV_MAPPING` dict.

1. Add an input spec for the problem type in `gnarl/envs/generate/specs.py`.

1. Add a sampler for the problem type in `gnarl/envs/generate/sampler.py` and add it to `SAMPLERS`.

1. Create a configuration file for the problem in `configs`, and then you should be ready to go!


