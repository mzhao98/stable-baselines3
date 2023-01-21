import copy
import pdb
import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
import gym
from gym import spaces
from torch.nn import functional as F
import time

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, \
    MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from torch.autograd import Variable
import sklearn
import sklearn.metrics
import warnings

warnings.filterwarnings("ignore")
from scipy.special import rel_entr


class INFLUENCE_PPO_HARVEST_VECTOR_V1_TRUE_PARTNER_PRETRAINED_DESIRED_DREAM_INFLUENCE(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "cuda:1",
            _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                    batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                    buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.episode_instance = 0
        self.all_game_data_by_ep = {}
        self.true_partner_model = None
        self.desired_partner_policy = None

        if _init_setup_model:
            self._setup_model()

    def set_reward_params(self, action_to_one_hot, marginal_probability, desired_strategy):
        # action_to_one_hot = {0: [1, 0, 0,0], 1: [0, 1, 0,0], 2: [0, 0, 1,0], 3:[0,0,0,1]}
        # action_to_one_hot = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
        # action_to_one_hot = {0: [1, 0], 1: [0, 1]}
        self.action_to_one_hot = action_to_one_hot
        # marginal_probability = np.array([0, 0])  # np.array([0,0,0])
        # marginal_probability = np.array([0, 0, 0])
        # marginal_probability = np.array([0, 0, 0, 0])
        self.default_marginal_probability = marginal_probability

        # desired_probability = [0.9, 0.1]  # [0,1,0]
        # desired_probability = [0.01, 0.98, 0.01]
        # desired_probability = [0.48, 0.01, 0.01, 0.48]
        self.desired_strategy = desired_strategy

    def set_true_partner(self, true_partner_model):
        self.true_partner_model = true_partner_model

    def set_desired_partner_policy(self, desired_partner_policy):
        self.desired_partner_policy = desired_partner_policy

    def set_partner_model(self, partner_model, transform_influence_reward, transform_influence_reward_w_weights, device):
        self.partner_model = partner_model
        self.device = device
        self.partner_model.to(self.device)
        self.transform_influence_reward = transform_influence_reward
        self.transform_influence_reward_w_weights = transform_influence_reward_w_weights

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        # Define candidate weight combinations and separate buffers
        self.candidate_weights = dict(enumerate([(1, 0, 0), (1, 10, 10), (0, 10, 10), (1, 5, 10), (1, 10, 20)]))
        self.candidate_buffers = {}
        for i in self.candidate_weights:
            self.candidate_buffers[i] = RolloutBuffer(
                self.n_steps,
                self.observation_space,
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
            )

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


    def hypothetical_train(self, temp_policy, buffer) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """

        # Switch to train mode (this affects batch norm / dropout)
        temp_policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(temp_policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = temp_policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                temp_policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(temp_policy.parameters(), self.max_grad_norm)
                temp_policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(buffer.values.flatten(), buffer.returns.flatten())

        # Logs
        # self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        # self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        # self.logger.record("train/value_loss", np.mean(value_losses))
        # self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        # self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        # self.logger.record("train/loss", loss.item())
        # self.logger.record("train/explained_variance", explained_var)
        # if hasattr(self.policy, "log_std"):
        #     self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        #
        # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/clip_range", clip_range)
        # if self.clip_range_vf is not None:
        #     self.logger.record("train/clip_range_vf", clip_range_vf)
        return temp_policy

    def set_episode_instance(self, episode_instance):
        self.episode_instance = episode_instance
        self.all_game_data_by_ep[self.episode_instance] = []

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()





        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        game_outcomes = []
        self._previous_ego_action = [0] * self.action_space.n
        mi_divergence = 0
        mi_guiding_divergence = 1

        self.prev_guidance_divergence = None

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                # pdb.set_trace()
                actions, values, log_probs, action_proba_distr = self.policy(obs_tensor, deterministic=False)

            # print(f"action_proba_distr: {action_proba_distr}")
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            ## Compute new rewards
            # print("actions", actions)
            # print("infos", infos)
            # pdb.set_trace()

            # new_rewards = copy.deepcopy(rewards)
            mi_divergence = 0
            if self.true_partner_model is not None:
                # Conditional Probability distribution
                # see human model prediction
                # Predicting using partner action
                # state_t = self._last_obs[0].squeeze(axis=1)
                # state_t1 = new_obs[0].squeeze(axis=1)
                # ego_action_t = self.action_to_one_hot[actions[0]]
                # human_pred_input = np.concatenate([ego_action_t, state_t[:-1], state_t1[:-1]], axis=0)  # length = (14,)
                # human_pred_input = np.expand_dims(np.array([human_pred_input]), axis=2)
                # if self.true_partner_model is not None:
                #     pdb.set_trace()

                # obs_np = obs_tensor.cpu().detach().numpy()
                obs_np = new_obs
                obs_np = np.squeeze(obs_np, axis=2)
                obs_np = np.squeeze(obs_np, axis=0)

                partner_last_action = obs_np[0:6]
                ego_last_action = obs_np[6:12]

                # pdb.set_trace()
                # if np.where(ego_last_action == 1) != clipped_actions[0]:
                # print(f"obs = {obs_np}: {np.where(ego_last_action == 1)} != {clipped_actions[0]}")
                if 1 in ego_last_action:
                    assert np.where(ego_last_action == 1) == clipped_actions[0]

                ego_pos = obs_np[12:14]
                partner_pos = obs_np[14:16]
                num_apples_left = obs_np[16:17]
                five_closest = obs_np[17:]
                apple_locs = [(obs_np[17], obs_np[18]),
                              (obs_np[19], obs_np[20]),
                              (obs_np[21], obs_np[22]),
                              (obs_np[23], obs_np[24]),
                              (obs_np[25], obs_np[26])]

                obs_from_partner_perspective = []
                obs_from_partner_perspective.extend(ego_last_action)
                obs_from_partner_perspective.extend(partner_last_action)
                obs_from_partner_perspective.extend(partner_pos)
                obs_from_partner_perspective.extend(ego_pos)
                obs_from_partner_perspective.extend(num_apples_left)

                closest_apple_locs = []
                check_apple_positions = []
                for i in range(len(apple_locs)):
                    apple_loc = apple_locs[i]
                    distance_to_player = np.sqrt(
                        (partner_pos[0] - apple_loc[0]) ** 2 + (partner_pos[1] - apple_loc[1]) ** 2)
                    check_apple_positions.append((i, distance_to_player))

                check_apple_positions = sorted(check_apple_positions, key=lambda x: x[1])
                for i in range(5):
                    if i >= len(check_apple_positions):
                        closest_apple_locs.extend([0, 0])
                    else:
                        apple_idx = check_apple_positions[i][0]
                        closest_apple_locs.extend([apple_locs[apple_idx][0], apple_locs[apple_idx][1]])

                obs_from_partner_perspective.extend(closest_apple_locs)

                new_partner_obs = np.expand_dims(np.array([obs_from_partner_perspective]), axis=2)
                tensor_partner_obs = Variable(th.Tensor(new_partner_obs)).to(device=self.device)
                _, _, _, action_proba_distr = self.true_partner_model.policy(tensor_partner_obs, deterministic=False)

                _, _, _, desired_partner_action_proba_distr = self.desired_partner_policy(tensor_partner_obs, deterministic=False)

                action_proba_distr = action_proba_distr.cpu().detach().numpy()
                predicted_partner_action = np.squeeze(action_proba_distr, axis=0)

                desired_partner_action_proba_distr = desired_partner_action_proba_distr.cpu().detach().numpy()
                desired_partner_action_proba_distr = np.squeeze(desired_partner_action_proba_distr, axis=0)

                conditional_probability = predicted_partner_action / sum(predicted_partner_action)
                desired_conditional_probability = desired_partner_action_proba_distr / sum(desired_partner_action_proba_distr)

                # Marginal Probability distribution
                marginal_probability = self.default_marginal_probability
                for candidate_action in self.action_to_one_hot:
                    # print("ego_action_t init", ego_action_t)
                    # print("action_to_one_hot", self.action_to_one_hot)
                    ego_action_t = copy.deepcopy(self.action_to_one_hot[candidate_action])
                    obs_from_partner_perspective[0:6] = ego_action_t

                    new_partner_obs = np.expand_dims(np.array([obs_from_partner_perspective]), axis=2)
                    tensor_partner_obs = Variable(th.Tensor(new_partner_obs)).to(device=self.device)
                    _, _, _, action_proba_distr = self.true_partner_model.policy(tensor_partner_obs,
                                                                                 deterministic=False)
                    action_proba_distr = action_proba_distr.cpu().detach().numpy()
                    intervention_predicted_partner_action = np.squeeze(action_proba_distr, axis=0)

                    marginal_probability = marginal_probability + intervention_predicted_partner_action

                marginal_probability = marginal_probability / sum(marginal_probability)

                # pdb.set_trace()
                # mi_divergence = sklearn.metrics.mutual_info_score(conditional_probability,marginal_probability)
                # mi_divergence = sum(rel_entr(conditional_probability, marginal_probability))

                mi_divergence_1 = sum(
                    rel_entr(marginal_probability, conditional_probability))
                mi_divergence_2 = sum(
                    rel_entr(conditional_probability, marginal_probability))
                mi_divergence = max(mi_divergence_1, mi_divergence_2)

            # print(f"conditional_probability = {conditional_probability}, marginal_probability = {marginal_probability}")

            # mi_guiding_divergence = sklearn.metrics.mutual_info_score(conditional_probability, desired_probability)

            ## Compute divergence from desired strategy
            # current_position = str((state_t1[0], state_t1[1]))
            mi_guiding_divergence = 0

            if self.desired_partner_policy is not None and self.true_partner_model is not None:
                mi_guiding_divergence_1 = sum(
                    rel_entr(desired_conditional_probability, conditional_probability))
                mi_guiding_divergence_2 = sum(
                    rel_entr(conditional_probability, desired_conditional_probability))
                mi_guiding_divergence = max(mi_guiding_divergence_1, mi_guiding_divergence_2)
            # if self.desired_strategy is not None:
            #     desired_action_prob = self.desired_strategy
            #
            #     mi_guiding_divergence = sum(rel_entr(conditional_probability, desired_action_prob))

            ##### DONE COMPUTING DIVERGENCES FOR NEW REWARDS

            # print("mi_divergence", mi_divergence)
            # print("mi_guiding_divergence", mi_guiding_divergence)
            # pdb.set_trace()

            alpha = 0.5
            beta = 10
            gamma = 20
            # modified_reward = alpha * rewards[0] + beta * mi_divergence - gamma * mi_guiding_divergence
            # modified_reward = mi_divergence
            # if self._previous_ego_action is None:
            #     modified_reward = 0
            # else:
            #     modified_reward = 1 / (mi_guiding_divergence)
            #
            # if self.prev_guidance_divergence is not None and mi_guiding_divergence < 0.2:
            #     mi_guiding_divergence_reward = 10
            # else:
            #     mi_guiding_divergence_reward = -1

            # mi_guiding_divergence_reward

            # rewards[0] = 1 * mi_divergence + (-100 * np.tan(mi_guiding_divergence - 0.5))
            # rewards[0] = 1 * mi_divergence + (-np.exp(5*mi_guiding_divergence)) + mi_guiding_divergence_reward
            env_reward = rewards[0]
            # rewards[0] =  rewards[0] + 10 * mi_divergence + (-100 * np.tan((mi_guiding_divergence - 0.5) % np.pi))
            # rewards[0] = rewards[0] + 10 * mi_divergence + (-100 * np.tan( np.clip(mi_guiding_divergence, 0, np.pi/2) - 0.5) )
            # rewards[0] = rewards[0] + (10 * np.tan(np.clip(mi_divergence, 0, np.pi / 2) - 0.5)) + (-100 * np.tan(np.clip(mi_guiding_divergence, 0, np.pi / 2) - 0.5))
            # rewards[0] = rewards[0] + (10 * np.tan(np.clip(mi_divergence, 0, np.pi / 2) - 0.5)) + (-100 * np.tan(np.clip(mi_guiding_divergence, 0, np.pi / 2) - 0.5))
            # mi_guiding_divergence = (100 * np.tan(np.clip(mi_guiding_divergence, 0, np.pi / 2) - 0.5))
            # rewards[0] = alpha * rewards[0] + beta * mi_divergence - gamma * mi_guiding_divergence
            rewards[0] = self.transform_influence_reward(env_reward, mi_divergence, mi_guiding_divergence, self.episode_instance)

            new_reward = rewards[0]
            old_reward = env_reward

            # rewards[0] = alpha * rewards[0] + beta * mi_divergence

            # print(f"env rew: {alpha * env_reward} + influence div: {beta * mi_divergence} + (guiding:{-gamma * mi_guiding_divergence})  ---> rewards[0] :{rewards[0]}")

            # if self.prev_guidance_divergence is None:
            #     self.prev_guidance_divergence = mi_guiding_divergence
            #
            # if mi_guiding_divergence < self.prev_guidance_divergence:
            #     self.prev_guidance_divergence = mi_guiding_divergence

            # DONE MODIFYING REWARDS

            # Remember previous ego actions
            self._previous_ego_action = actions
            ## Done computing new rewards

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            # actions = actions.cpu().detach().numpy()
            # values = values.cpu().detach()
            # log_probs = log_probs.cpu().detach()
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)



            # Recompute for each candidate reward function
            for c_idx in self.candidate_weights:
                (c_alpha, c_beta, c_gamma) = self.candidate_weights[c_idx]
                candidate_rewards = copy.deepcopy(rewards)
                candidate_rewards[0] = self.transform_influence_reward_w_weights(env_reward, mi_divergence, mi_guiding_divergence,
                                                             self.episode_instance, c_alpha, c_beta, c_gamma)


                self.candidate_buffers[c_idx].add(self._last_obs, actions, candidate_rewards, self._last_episode_starts, values, log_probs)

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        self.all_game_data_by_ep[self.episode_instance].extend(game_outcomes)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        for c_idx in self.candidate_weights:
            self.candidate_buffers[c_idx].compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True


    def rollout_environment_reward(
            self,
            temp_policy,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        temp_policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()





        # Sample new weights for the state dependent exploration
        if self.use_sde:
            temp_policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        total_env_reward = 0
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                temp_policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                # pdb.set_trace()
                actions, values, log_probs, action_proba_distr = temp_policy(obs_tensor, deterministic=False)

            # print(f"action_proba_distr: {action_proba_distr}")
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            total_env_reward += rewards[0]

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            n_steps += 1



        callback.on_rollout_end()

        return total_env_reward

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "INFLUENCE_PPO",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "INFLUENCE_PPO":

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            best_env_reward = -1000000
            for c_idx in self.candidate_buffers:
                temp_policy = copy.deepcopy(self.policy)
                temp_policy_updated = self.hypothetical_train(temp_policy, self.candidate_buffers[c_idx])
                candidate_rollout_buffer = copy.deepcopy(self.candidate_buffers[c_idx])
                total_env_reward = self.rollout_environment_reward(temp_policy, self.env, callback, self.candidate_buffers[c_idx],
                                                      n_rollout_steps=self.n_steps)
                if total_env_reward > best_env_reward:
                    best_env_reward = total_env_reward
                    self.rollout_buffer = candidate_rollout_buffer


            self.train()

        callback.on_training_end()

        return self