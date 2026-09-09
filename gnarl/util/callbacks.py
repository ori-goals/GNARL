from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
from typing import Any
import numpy as np
from gnarl.util.evaluation import (
    process_evaluation_output,
    save_best_model,
    evaluate_policy,
)


class ActorFreezeCallback(BaseCallback):
    """
    Callback that freezes everything except the critic network for a specified number of steps,
    allowing only the critic (value function) to train during that period.
    """

    def __init__(self, actor_freeze_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.actor_freeze_steps = actor_freeze_steps
        self.networks_frozen = False
        self.freeze_end_step = actor_freeze_steps

    def _freeze_networks(self):
        """Freeze all parameters except the critic network."""
        if not self.networks_frozen:
            # Freeze action_net (NodeSimilarityMatchAgg) - the actor head
            for param in self.model.policy.action_net.parameters():
                param.requires_grad = False

            # Freeze features_extractor (GraphFeatureTransformer) - shared feature extraction
            for param in self.model.policy.features_extractor.parameters():
                param.requires_grad = False

            # Freeze mlp_extractor (GraphFeatureEncoderProcessor) - shared graph processing
            for param in self.model.policy.mlp_extractor.parameters():
                param.requires_grad = False

            # Keep value_net unfrozen - this is the critic we want to train
            # for param in self.model.policy.value_net.parameters():
            #     param.requires_grad = True  # This should already be True

            self.networks_frozen = True
            if self.verbose > 0:
                print(f"All networks except critic frozen at step {self.num_timesteps}")

    def _unfreeze_networks(self):
        """Unfreeze all parameters for normal training."""
        if self.networks_frozen:
            # Unfreeze action_net
            for param in self.model.policy.action_net.parameters():
                param.requires_grad = True

            # Unfreeze features_extractor
            for param in self.model.policy.features_extractor.parameters():
                param.requires_grad = True

            # Unfreeze mlp_extractor
            for param in self.model.policy.mlp_extractor.parameters():
                param.requires_grad = True

            # value_net should already be unfrozen

            self.networks_frozen = False
            if self.verbose > 0:
                print(f"All networks unfrozen at step {self.num_timesteps}")

    def _on_training_start(self) -> None:
        """Called before the first rollout starts."""
        if self.actor_freeze_steps > 0:
            self._freeze_networks()

    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Check if we should unfreeze the networks
        if self.networks_frozen and self.num_timesteps >= self.freeze_end_step:
            self._unfreeze_networks()
        return True


class MaskableEvalCallback(EvalCallback):
    """
    Callback for evaluating an agent. Supports invalid action masking.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    :param use_masking: Whether to use invalid action masks during evaluation
    """

    def __init__(self, *args, use_masking: bool = True, log_prefix="", **kwargs):
        self.eval_envs = kwargs.pop("eval_env")
        kwargs["eval_env"] = self.eval_envs[0]
        super().__init__(*args, **kwargs)
        self.use_masking = use_masking
        self.pfx = log_prefix + "/val" if log_prefix else "val"
        self.best = {
            "mean_success": 0.0,
            "mean_reward": -np.inf,
            "mean_length": np.inf,
        }

    def _on_step(self) -> bool:
        continue_training = True

        def flat(lst: list[list[Any]]) -> list[Any]:
            """Flatten a list of lists."""
            return [item for sublist in lst for item in sublist]

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            (
                episode_rewards,
                episode_lengths,
                episode_success,
                num_nodes,
                objective_values,
                expert_objectives,
                _,
            ) = evaluate_policy(
                self.model.policy,
                self.eval_envs,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                warn=self.warn,
                callback=self._log_success_callback,
                use_masking=self.use_masking,
            )

            # Flatten the lists
            flt_episode_rewards = flat(episode_rewards)
            flt_episode_lengths = flat(episode_lengths)

            if self.log_path is not None:
                assert isinstance(flt_episode_rewards, list)
                assert isinstance(flt_episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(flt_episode_rewards)
                self.evaluations_length.append(flt_episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_reward, std_reward = np.mean(flt_episode_rewards), np.std(
                flt_episode_rewards
            )
            mean_ep_length, std_ep_length = np.mean(flt_episode_lengths), np.std(
                flt_episode_lengths
            )
            self.last_mean_reward = float(mean_reward)

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            log_info = process_evaluation_output(
                self.pfx,
                episode_rewards,
                episode_lengths,
                episode_success,
                num_nodes,
                objective_values,
                expert_objectives,
            )
            for key, value in log_info.items():
                self.logger.record(key, value)

            if len(self._is_success_buffer) > 0:
                mean_success = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * mean_success:.2f}%")
            else:
                mean_success = 0

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if save_best_model(
                self.best,
                {
                    "mean_success": mean_success,
                    "mean_reward": mean_reward,
                    "mean_length": mean_ep_length,
                },
                self.model,
                self.best_model_save_path,
                "ppo",
                self.verbose,
            ):
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
