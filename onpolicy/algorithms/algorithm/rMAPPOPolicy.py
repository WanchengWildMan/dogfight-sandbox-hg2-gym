"""
# @Time    : 2021/7/1 6:53 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : rMAPPOPolicy.py
"""

import torch
from torch.autograd import Variable

from onpolicy.algorithms.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule
import numpy as  np


class RMAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.action_bound_method = args.action_bound_method
        self.action_scaling = args.action_scaling

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.merged_critic = args.merged_critic

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        if not self.merged_critic:
            self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        if not self.merged_critic:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                     lr=self.critic_lr,
                                                     eps=self.opti_eps,
                                                     weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        if not self.merged_critic:
            update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        所有变量都在cpu
        输入numpy
        输出tensor
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        :param merged_critic: (bool) if True ,the critics of all agents are in one class

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        obs_torch = Variable(torch.tensor(obs), requires_grad=False)
        # obs = Variable(torch.Tensor(obs), requires_grad=False)
        rnn_states_actor_torch = Variable(torch.tensor(rnn_states_actor), requires_grad=False)
        masks_torch = Variable(torch.tensor(masks), requires_grad=False)

        actions, action_log_probs, rnn_states_actor = self.actor(obs_torch, rnn_states_actor_torch, masks_torch, available_actions, deterministic)

        cent_obs_torch = Variable(torch.tensor(cent_obs), requires_grad=False)
        rnn_states_critic_torch = Variable(torch.tensor(rnn_states_critic), requires_grad=False)
        masks_torch = Variable(torch.tensor(masks), requires_grad=False)
        if not self.merged_critic:
            values, rnn_states_critic = self.critic(cent_obs_torch, rnn_states_critic_torch, masks_torch)
        else:
            values = None
            rnn_states_critic = torch.from_numpy(rnn_states_critic)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions of a single agent.
        """
        cent_obs_torch = Variable(torch.tensor(cent_obs), requires_grad=False)
        rnn_states_critic_torch = Variable(torch.tensor(rnn_states_critic), requires_grad=False)
        masks_torch = Variable(torch.tensor(masks), requires_grad=False)
        if not self.merged_critic:
            values, _ = self.critic(cent_obs_torch, rnn_states_critic_torch, masks_torch)
        else:
            values = None
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)
        if not self.merged_critic:
            values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        else:
            values = None
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

    def map_action(self, act):
        """Map raw network output to action range in gym's env.action_space.

        This function is called in :meth:`~tianshou.data.Collector.collect` and only
        affects action sending to env. Remapped action will not be stored in buffer
        and thus can be viewed as a part of env (a black box action transformation).

        Action mapping includes 2 standard procedures: bounding and scaling. Bounding
        procedure expects original action range is (-inf, inf) and maps it to [-1, 1],
        while scaling procedure expects original action range is (-1, 1) and maps it
        to [action_space.low, action_space.high]. Bounding procedure is applied first.

        :param act: a data batch or numpy.ndarray which is the action taken by
            policy.forward.

        :return: action in the same form of input "act" but remap to the target action
            space.
        """
        if self.act_space.__class__.__name__ == "Box" and \
                isinstance(act, np.ndarray):
            # currently this action mapping only supports np.ndarray action
            if self.action_bound_method == "clip":
                act = np.clip(act, -1.0, 1.0)
            elif self.action_bound_method == "tanh":
                act = np.tanh(act)
            if self.action_scaling:
                assert np.min(act) >= -1.0 and np.max(act) <= 1.0, \
                    "action scaling only accepts raw action range = [-1, 1]"
                low, high = self.act_space.low, self.act_space.high
                act = low + (high - low) * (act + 1.0) / 2.0  # type: ignore
        return act
