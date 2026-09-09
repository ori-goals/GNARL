#!/usr/bin/env python3

import gnarl
import numpy as np
import torch as th
import os
import yaml
import argparse

from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

import wandb
from wandb.integration.sb3 import WandbCallback

from gnarl.agent.policy import MaskableNodeActorCriticPolicy
from gnarl.agent.imitation.imitation import (
    behavioural_cloning,
    split_dataset,
    make_data_loader,
)
from gnarl.util.evaluation import (
    ExpertPolicy,
    calculate_env_split,
)
from gnarl.util.envs import make_train_env, make_eval_env
from gnarl.util.classes import get_clean_kwargs
from gnarl.util.bc import get_bc_experience, complete_config
from gnarl.util.callbacks import MaskableEvalCallback
import random

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run GNARL environment training.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-w",
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed in the config.",
    )
    return parser.parse_args()


def seed_worker(worker_id):
    worker_seed = (th.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def default_collate(batch: list[dict]):
    """Simple, picklable collate function that stacks tensors and handles dict observations.

    Designed to be safe for DataLoader workers (top-level function).
    """
    out = {}
    keys = set().union(*[b.keys() for b in batch])
    for k in keys:
        vals = [b.get(k) for b in batch]
        if vals[0] is None:
            out[k] = vals
            continue
        if isinstance(vals[0], dict):
            subkeys = set().union(*[v.keys() for v in vals if isinstance(v, dict)])
            out[k] = {}
            for sk in subkeys:
                subvals = [v.get(sk) if isinstance(v, dict) else None for v in vals]
                # try stacking tensors
                try:
                    stacked = th.stack(
                        [
                            sv if isinstance(sv, th.Tensor) else th.tensor(sv)
                            for sv in subvals
                        ]
                    )
                    out[k][sk] = stacked
                except Exception:
                    out[k][sk] = subvals
        else:
            if all(isinstance(v, th.Tensor) for v in vals):
                out[k] = th.stack(vals)
            else:
                try:
                    out[k] = th.stack([th.tensor(v) for v in vals])
                except Exception:
                    out[k] = vals
    return out


def log_model_artifact(run_id: str, method: str):
    """Log the best model as a W&B artifact"""
    model_path = os.path.join(f"models/{run_id}", f"{method}_best_model.pt")
    artifact = wandb.Artifact(f"{run_id}_{method}_best_model", type="model")
    artifact.add_file(model_path)
    meta_path = os.path.join(f"models/{run_id}", f"{method}_best_model_metadata.yaml")
    with open(meta_path, "r") as f:
        metadata = yaml.safe_load(f)

    artifact.metadata = metadata
    wandb.log_artifact(artifact)


def train_ppo(
    train_env: VecEnv, val_envs: list[VecEnv], run, config: dict, policy_path=None
):
    ppo_kwargs = get_clean_kwargs(
        MaskablePPO.__init__,
        warn=False,
        kwargs=config["PPO"],
    )

    # Create the PPO model
    model = MaskablePPO(
        MaskableNodeActorCriticPolicy,
        train_env,
        **ppo_kwargs,
        policy_kwargs=config["policy_kwargs"],
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
    )

    # Load the model if a path is provided
    if policy_path is not None and os.path.exists(policy_path):
        print(f"Loading pre-trained model from {policy_path}.")
        if "best_model.pt" in policy_path:
            model.policy.load_state_dict(
                th.load(policy_path, weights_only=False)["state_dict"]
            )
        else:
            model.policy.load_state_dict(th.load(policy_path, weights_only=True))
        print("Pre-trained model loaded successfully.")
    else:
        print("No pre-trained model found, starting training from scratch.")

    model.policy = model.policy.to(device)

    per_env_samples = calculate_env_split(
        node_samples=config["val_data"]["node_samples"],
        max_envs=config["val_data"]["num_envs"],
    )
    eval_episode_list = [c * e for n, c, e in per_env_samples]
    eval_callback = MaskableEvalCallback(
        eval_env=val_envs,
        best_model_save_path=f"models/{run.id}",
        log_path=f"eval_logs/{run.id}",
        eval_freq=max(config["PPO"]["eval_freq"] // train_env.num_envs, 1),
        n_eval_episodes=eval_episode_list,
        deterministic=config["val_data"]["deterministic"],
        render=False,
        log_prefix="ppo",
    )

    # Create list of callbacks
    callbacks = [
        WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
        eval_callback,
    ]

    # Train the model
    model.learn(
        total_timesteps=config["PPO"]["timesteps"],
        progress_bar=True,
        callback=callbacks,
    )

    log_model_artifact(run.id, "ppo")


def train_bc(
    train_env: VecEnv,
    val_envs: list[VecEnv],
    config: dict,
    run,
    experience_path=None,
):

    dataset = get_bc_experience(experience_path, train_env, config)

    print("Creating data loader for Behavioural Cloning.")
    torch_generator = th.Generator().manual_seed(config["BC"]["seed"])
    torch_generator_train = th.Generator().manual_seed(config["BC"]["seed"])
    train_data, val_data = split_dataset(
        dataset,
        split=0.8,
        generator=torch_generator,
    )
    num_workers = config["train_data"].get("num_envs", 0)
    collate_fn = (
        dataset.collate_fn if hasattr(dataset, "collate_fn") else default_collate
    )
    data_loader = make_data_loader(
        train_data,
        shuffle=True,
        batch_size=config["BC"].get("batch_size", 64),
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=torch_generator_train,
    )
    val_loader = make_data_loader(
        val_data,
        shuffle=False,
        batch_size=config["BC"].get("batch_size", 64),
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=torch_generator,
    )

    print("Training a policy using Behavioural Cloning.")
    policy = MaskableNodeActorCriticPolicy(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        lr_schedule=get_linear_fn(
            config["BC"]["learning_rate"], config["BC"]["learning_rate"], 1.0
        ),  # start, end, duration
        **config["policy_kwargs"],
    )
    wandb.watch(policy, log="all", log_freq=100)

    per_env_samples = calculate_env_split(
        node_samples=config["val_data"]["node_samples"],
        max_envs=config["val_data"]["num_envs"],
    )
    eval_episode_list = [c * e for n, c, e in per_env_samples]

    policy = behavioural_cloning(
        policy=policy,
        data_loader=data_loader,
        val_loader=val_loader,
        val_envs=val_envs,
        **config["BC"],
        progress_bar=True,
        best_model_save_path=f"models/{run.id}",
        deterministic_eval=config["val_data"]["deterministic"],
        n_eval_episodes=eval_episode_list,
    )

    log_model_artifact(run.id, "bc")


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    config = complete_config(config)

    if args.seed is not None:
        config["seed"] = args.seed
        if "PPO" in config:
            config["PPO"]["seed"] = args.seed

    os.environ["PYTHONHASHSEED"] = str(config["seed"])
    random.seed(config["seed"])
    th.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    run = wandb.init(
        project="GNARL",
        config=config,
        mode="disabled" if not args.wandb else "online",
        name=f"{config['env']}_{list(config['train_data']['node_samples'].keys())}train",
        tags=[
            config["algorithm"],
            f"train-{list(config['train_data']['node_samples'].keys())}",
            f"network-{config['policy_kwargs']['network_kwargs']['network']}",
            f"aggr-{config['policy_kwargs']['network_kwargs']['aggr']}",
            f"pooling-{config['policy_kwargs']['pooling_type']}",
            f"embed-{config['policy_kwargs']['embed_dim']}",
        ],
        notes="",
    )

    print(f"============== Beginning run: {run.id} ==============")

    print("Constructing train env")
    train_env = make_train_env(config, run.id)
    print("Constructing val env")
    val_envs = make_eval_env(config, run.id, "val")

    # Get the spec attribute from the first environment in vec_env
    env_spec = train_env.get_attr("graph_spec")[0]
    config["policy_kwargs"]["graph_spec"] = env_spec

    if "BC" in config:
        print("Starting Behavioural Cloning training...")
        # Train a policy using Behavioural Cloning
        train_bc(
            train_env,
            val_envs,
            config,
            run,
            experience_path=config["BC"]["data_path"],
        )
    if "PPO" in config:
        print("Starting PPO training...")
        # Train the policy using PPO
        train_ppo(
            train_env,
            val_envs,
            run,
            config,
            policy_path=(
                f"models/{run.id}/bc_best_model.pt" if "BC" in config else None
            ),
        )

    print(f"============== Ending run: {run.id} ==============")


if __name__ == "__main__":
    main()
