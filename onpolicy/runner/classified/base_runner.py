import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter

from onpolicy.utils.shared_buffer import SharedReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.num_agent_types = self.all_args.num_agent_types
        self.num_agents_each_type = self.all_args.num_agents_each_type
        self.last_agent_each_type = np.cumsum(self.all_args.num_agents_each_type) - 1
        self.type_each_agent = np.zeros(self.num_agents, dtype=int)
        for i, last_agent in enumerate(self.last_agent_each_type):
            self.type_each_agent[(0 if i == 0 else self.last_agent_each_type[i - 1] + 1):last_agent + 1] = i
        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.merged_critic = self.all_args.merged_critic

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        # else:
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        from onpolicy.algorithms.algorithm.r_mappo import RMAPPOTrainAlgo as TrainAlgo
        from onpolicy.algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy

        self.policy = []
        for agent_type in range(self.num_agent_types):
            agent_id = self.last_agent_each_type[agent_type]
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            # policy network
            po = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device=self.device)
            self.policy.append(po)

        if self.merged_critic:
            from onpolicy.algorithms.algorithm.mhat_critics import MultiHeadAttentionCritic
            self.central_critic = MultiHeadAttentionCritic([(obs.shape[0], acs.shape[0]) for obs, acs in zip(self.envs.observation_space, self.envs.action_space)], hidden_dim=self.all_args.hidden_size)
            # from onpolicy.algorithms.algorithm.at_critics import AttentionCritic
            # self.central_critic = AttentionCritic([(obs.shape[0], acs.shape[0]) for obs, acs in zip(self.envs.observation_space, self.envs.action_space)], hidden_dim=self.all_args.hidden_size)
            # self.central_critic = MultiHeadAttentionCritic([(obs.shape[0], acs.shape[0]) for obs, acs in zip(self.envs.observation_space, self.envs.action_space)], hidden_dim=self.all_args.hidden_size)
            self.central_critic_optimizer = torch.optim.Adam(self.central_critic.parameters(),
                                                             lr=self.all_args.lr,
                                                             eps=self.all_args.opti_eps,
                                                             weight_decay=self.all_args.weight_decay)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        # buffer_type = []

        for agent_type in range(self.all_args.num_agent_types):
            # self.buffer.append(SharedReplayBuffer(self.all_args, self.envs.observation_space[last_agent_each_type[agent_type]], share_observation_space))
            # for agent_id in range(self.num_agent_types):
            agent_id = self.last_agent_each_type[agent_type]
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_type], device=self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_type] if self.use_centralized_V else self.envs.observation_space[agent_id]

            # if self.use_centralized_V:
            #     # 单例
            #     self.buffer.append(buffer_type[type_each_agent[agent_id]])
            # else:
            bu = SharedReplayBuffer(self.all_args,
                                    self.num_agents_each_type[agent_type],
                                    self.envs.observation_space[agent_id],
                                    share_observation_space,
                                    self.envs.action_space[agent_id],
                                    long_decision_agent=self.all_args.is_long_decision_agent[agent_id],

                                    )
            self.buffer.append(bu)
            self.trainer.append(tr)

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_type in range(self.num_agent_types):
            self.trainer[agent_type].prep_rollout()
            next_values = self.trainer[agent_type].policy.get_values(self.buffer[agent_type].share_obs[-1],
                                                                     self.buffer[agent_type].rnn_states_critic[-1],
                                                                     self.buffer[agent_type].masks[-1])
            next_value = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
            self.buffer[agent_type].compute_returns(next_value, self.trainer[agent_type].value_normalizer)

    def train(self):
        train_infos = []
        for agent_type in range(self.num_agent_types):
            self.trainer[agent_type].prep_training()
            train_info = self.trainer[agent_type].train(self.buffer[agent_type])
            train_infos.append(train_info)
            self.buffer[agent_type].after_update()

        return train_infos

    def save(self):
        for agent_type in range(self.num_agent_types):
            policy_actor = self.trainer[agent_type].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_type) + ".pt")
            if not self.merged_critic:
                policy_critic = self.trainer[agent_type].policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_type) + ".pt")

    def restore(self):
        for agent_type in range(self.num_agent_types):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_type) + '.pt')
            self.policy[agent_type].actor.load_state_dict(policy_actor_state_dict)
            if not self.merged_critic:
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_type) + '.pt')
                self.policy[agent_type].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        # for agent_id in range(self.num_agents):
        #     for k, v in train_infos[agent_id].items():
        #         agent_k = "agent%i/" % agent_id + k
        #         if self.use_wandb:
        #             wandb.log({agent_k: v}, step=total_num_steps)
        #         else:
        #             self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
        for agent_type in range(self.num_agent_types):
            for k, v in train_infos[agent_type].items():
                agent_k = "agent_type%i/" % agent_type + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
