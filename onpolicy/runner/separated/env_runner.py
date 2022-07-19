"""
# @Time    : 2021/7/1 7:14 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_runner.py
"""

import time
import math
import numpy as np
from itertools import chain
import torch.nn as nn
import torch
import wandb
from torch.autograd import Variable
from torch.nn import MSELoss

from onpolicy.runner.separated.base_runner import Runner
import imageio

from onpolicy.utils.util import check
import warnings

warnings.filterwarnings("ignore", category=Warning)


def _t2n(x):
    return x.detach().cpu().numpy()


def _n2tv(x):
    return torch.autograd.Variable(torch.from_numpy(x))


def _o2a(arr):
    return np.array(list(map(lambda x: x.tolist(), list(arr.tolist()))))


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        all_args = config["all_args"]

    def train(self):
        train_infos = []
        samples_per_agent = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            if self.merged_critic:
                self.central_critic.to(self.device)

            # actor update
            train_info, samples_per_epoch, advantages = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)
            samples_per_agent.append(samples_per_epoch)
            self.buffer[agent_id].after_update()
        if self.merged_critic:
            dim_each_agent = np.zeros(self.num_agents, dtype=int)
            for agent_id in range(self.num_agents):
                dim_each_agent[agent_id] = self.envs.observation_space[agent_id].shape[0]
            cum_dim_each_agent = np.concatenate([[0], np.cumsum(dim_each_agent)])
            # 集中计算value loss才能backward？
            for agent_id in range(self.num_agents):
                value_batches = []
                value_preds_batches = []
                return_batches = []
                active_masks_batches = []
                # value_loss = []  # torch.autograd.Variable(torch.zeros(1)).to(self.device)
                # value_loss = torch.autograd.Variable(torch.zeros(1)).to(self.device)
                for epoch in range(self.trainer[0].ppo_epoch):
                    # num_sample_need = self.episode_length // self.num_agents
                    num_sample_need = self.episode_length
                    for si, sample in enumerate(samples_per_agent[agent_id][epoch]):
                        if si >= num_sample_need: break
                        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                        adv_targ, available_actions_batch, indices = sample
                        # old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
                        # adv_targ = check(adv_targ).to(**self.tpdv)
                        all_return_batch = check(self.buffer[agent_id].all_returns[indices % self.episode_length, indices // self.episode_length]).to(**self.tpdv)  # 所有智能体的return
                        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
                        all_value_preds_batch = check(np.array([self.buffer[i].value_preds[indices % self.episode_length, indices // self.episode_length] for i in range(self.num_agents)])).to(**self.tpdv)

                        return_batch = check(return_batch).to(**self.tpdv)
                        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

                        inps = []
                        for a_id in range(self.num_agents):
                            a = Variable(torch.tensor(np.zeros((len(share_obs_batch), self.envs.action_space[a_id].shape[0])))).to(self.device)
                            inps.append((Variable(torch.tensor(share_obs_batch[:, cum_dim_each_agent[a_id]:cum_dim_each_agent[a_id + 1]])).to(self.device), a))
                        values = self.central_critic(inps)  # 列表 num_agents个eps_len,1
                        all_values = self.central_critic(inps)  # 列表 num_agents个eps_len,1
                        all_values = torch.stack(all_values)
                        value_batches.append(values[agent_id].to(self.device))
                        value_preds_batches.append(value_preds_batch)
                        return_batches.append(return_batch)
                        active_masks_batches.append(active_masks_batch)
                        # value_loss += self.trainer[agent_id].cal_value_loss(values[agent_id].to(self.device), torch.tensor(value_preds_batch), torch.tensor(return_batch), active_masks_batch)
                        value_loss = self.trainer[agent_id].cal_value_loss(all_values.to(self.device), all_value_preds_batch, all_return_batch, active_masks_batch)
                        (value_loss.squeeze() * self.all_args.value_loss_coef).backward()
                # value_loss = self.trainer[0].cal_value_loss(torch.stack(value_batches), torch.stack(value_preds_batches), torch.stack(return_batches), torch.stack(active_masks_batches))
                # (value_loss.squeeze() * self.all_args.value_loss_coef).backward()

                # self.central_critic.scale_shared_grads()

                def get_gard_norm(it):
                    sum_grad = 0
                    for x in it:
                        if x.grad is None:
                            continue
                        sum_grad += x.grad.norm() ** 2
                    return math.sqrt(sum_grad)

                if self.all_args.use_max_grad_norm:
                    critic_grad_norm = nn.utils.clip_grad_norm_(self.central_critic.parameters(), self.all_args.max_grad_norm)
                else:
                    critic_grad_norm = get_gard_norm(self.central_critic.parameters())
                self.central_critic_optimizer.step()
                self.central_critic_optimizer.zero_grad()
                train_infos[agent_id]['value_loss'] += value_loss.item()
                train_infos[agent_id]['critic_grad_norm'] += critic_grad_norm

            # for _ in range(self.trainer[0].ppo_epoch):
            #     pass
            #     # value_loss,critic_grad_norm=
            # for agent_id in range(self.num_agents):
            #
            #     train_infos[agent_id]['value_loss'] += value_loss.item()
            #     train_infos[agent_id]['critic_grad_norm'] += critic_grad_norm
        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            if not self.merged_critic:
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
        if self.merged_critic:
            policy_critic = self.central_critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + "all" + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            if not self.merged_critic:
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
        if self.merged_critic:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + "all" + '.pt')
            self.central_critic.load_state_dict(policy_critic_state_dict)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        episode_st = (wandb.run.step if wandb.run else 1) // self.episode_length // self.n_rollout_threads
        eps_infos = []
        for episode in range(episode_st, episodes):

            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                st = time.time()
                # Sample actions actions_env is actions_remap
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                et = time.time()
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)

                while infos[0][0].get("next") is not None and infos[0][0]["next"] == False:
                    values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    obs, rewards, dones, infos = self.envs.step(actions_env)
                    self.pop()
                    data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                    # insert data into buffer
                    self.insert(data)
                eps_infos.append(infos[0])

                et = time.time()
                if self.use_render:
                    self.envs.render()

                if step % self.log_interval * 200 == 0 and False:
                    info_step = [infos[0]]
                    if self.env_name == "MPE" or True:
                        preserved_keys = ['individual_reward', 'next', 'average_episode_rewards']
                        for agent_id in range(self.num_agents):
                            # idv_rews = []
                            info_i_key = {}  # 一个episode内，第i个agent键为key 的 info
                            keys_i = info_step[0][agent_id].keys()
                            for key in keys_i:
                                if key in preserved_keys:
                                    continue
                                info_i_key[key] = []
                            for info in info_step:
                                for key in keys_i:
                                    if key in preserved_keys:
                                        continue
                                    info_i_key[key].append(info[agent_id][key])

                            for key in info_i_key.keys():
                                info_step[0][agent_id].update({key: sum(info_i_key[key])})
                                info_step[0][agent_id].update({key: sum(info_i_key[key])})

                    if step > 0:
                        for agent_id in range(self.num_agents):
                            info_step[0][agent_id].update(
                                {"average_step_rewards": (self.buffer[agent_id].rewards[step - self.log_interval * 200:step].mean())})
                    total_num_steps = (episode * self.episode_length + step) * self.n_rollout_threads

                    self.log_train(info_step[0], total_num_steps)
                    eps_infos = []

            # compute return and update network
            self.compute()
            st = time.time()

            train_infos = self.train()
            et = time.time()
            print(et - st)
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if total_num_steps > 10000:
                print(10000)

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                try:
                    self.save()
                except:
                    pass

            # log information
            end = time.time()
            print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                  .format(self.all_args.scenario_name,
                          self.algorithm_name,
                          self.experiment_name,
                          episode,
                          episodes,
                          total_num_steps,
                          self.num_env_steps,
                          int(total_num_steps / (end - start))), )

            if episode % self.log_interval == 0 and True:
                if self.env_name == "MPE" or True:
                    preserved_keys = ['individual_reward', 'next', 'average_episode_rewards']
                    for agent_id in range(self.num_agents):
                        # idv_rews = []
                        info_i_key = {}  # 一个episode内，第i个agent键为key 的 info
                        keys_i = eps_infos[0][agent_id].keys()
                        for key in keys_i:
                            if key in preserved_keys:
                                continue
                            info_i_key[key] = []
                        for info in eps_infos:
                            for key in keys_i:
                                if key in preserved_keys:
                                    continue
                                info_i_key[key].append(info[agent_id][key])

                            # if 'individual_reward' in info[agent_id].keys():
                            #     idv_rews.append(info[agent_id]['individual_reward'])

                        # train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                        # train_infos[agent_id].update(
                        #     {"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                        for key in info_i_key.keys():
                            if key.endswith("sum"):
                                train_infos[agent_id].update({key: np.sum(info_i_key[key])})
                            elif key.endswith("mean"):
                                train_infos[agent_id].update({key: np.mean(info_i_key[key])})
                for agent_id in range(self.num_agents):
                    train_infos[agent_id].update({'individual_rewards': np.sum(self.buffer[agent_id].rewards)})
                    train_infos[agent_id].update(
                        {"average_episode_rewards": np.mean(self.buffer[agent_id].rewards)})

                self.log_train(train_infos, total_num_steps)
                eps_infos = []

            print(train_infos)
            print(self.all_args)
            if wandb.run is not None:
                print(wandb.run.url)
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    @torch.no_grad()
    def compute(self):
        '''
        get next_value (V_{T+1}) and compute GAE
        '''
        if not self.merged_critic:
            for agent_id in range(self.num_agents):  # 智能体一个一个从critic中获取状态值
                self.trainer[agent_id].prep_rollout()
                next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1],
                                                                      self.buffer[agent_id].rnn_states_critic[-1],
                                                                      self.buffer[agent_id].masks[-1])
                next_value = _t2n(next_value)
                self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)  # 计算折扣回报
        else:
            inps = []
            for agent_id in range(self.num_agents):
                inps.append((_n2tv(self.buffer[agent_id].obs[-1]), _n2tv(self.buffer[agent_id].actions[-1])))

            next_values = self.central_critic(inps)  # 一次性从公共critics中获取状态值
            next_values = list(map(_t2n, next_values))
            for agent_id in range(self.num_agents):
                self.buffer[agent_id].compute_returns(next_values[agent_id], self.trainer[agent_id].value_normalizer)
            # all_returns = np.zeros((self.buffer[0].returns.shape[0], self.n_rollout_threads, self.num_agents))
            all_returns = np.concatenate([self.buffer[i].returns for i in range(self.num_agents)], axis=2)
            for agent_id in range(self.num_agents):
                self.buffer[agent_id].all_returns = all_returns  # 共享了

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []
        # 先收集所有action 再收集所有value
        for agent_id in np.arange(self.num_agents):
            self.trainer[agent_id].prep_rollout()  # to cpu!!!!!
            if self.merged_critic:
                self.central_critic.to("cpu")

            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            # if merged_critic ==True, value=none
            # [agents, envs, dim]
            values.append(_t2n(value) if value is not None else None)
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                action_env = self.policy[agent_id].map_action(action)

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))
        # 如果使用合并后的critic,集中计算values _pred
        if self.merged_critic:
            inps = []
            for agent_id in range(self.num_agents):
                inps.append((torch.autograd.Variable(torch.from_numpy(self.buffer[agent_id].obs[step])), torch.autograd.Variable(torch.from_numpy(actions[agent_id]))))
            values = self.central_critic(inps)
            values = list(map(_t2n, values))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        try:
            actions = np.array([[actions[a][b] for a in range(self.num_agents)] for b in range(len(actions[0]))])
        except:
            actions = np.array(actions).transpose(1, 0, 2)

        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def pop(self):
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].pop()

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                    dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            actions_tmp = actions[:, agent_id]

            if actions[:, agent_id].dtype == 'O':
                actions_tmp = _o2a(actions[:, agent_id])

            self.buffer[agent_id].insert(share_obs,
                                         np.array(list(obs[:, agent_id])),
                                         rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id],
                                         actions_tmp,
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id],
                                         rewards[:, agent_id].reshape([self.n_rollout_threads, 1]),
                                         masks[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length / (self.all_args.PERIODS_PER_EPISODE if hasattr(self.all_args, "PERIODS_PER_EPISODE") else 1)):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
                            eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()

                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                          rnn_states[:, agent_id],
                                                                          masks[:, agent_id],
                                                                          deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                     dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
