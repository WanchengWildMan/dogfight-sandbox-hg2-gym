"""
# @Time    : 2021/6/30 10:07 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : train.py
"""

# !/usr/bin/env python
import sys
import os
import warnings

import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import yaml

from env import HarFangDogFightEnv
from onpolicy.config import get_config

from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

"""Train script for MPEs."""


# def make_train_env(all_args):
#     return SubprocVecEnv(all_args)

def make_train_env(all_args):
    def get_env_fn(i):  # 返回能获取环境的函数
        def init_env(i):
            env = HarFangDogFightEnv()
            np.random.seed(all_args.seed)
            return env

        return init_env

    all_args.fn = [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
    return SubprocVecEnv(all_args)


def make_eval_env(all_args):
    return DummyVecEnv(all_args)


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='Dogfight', help="Which scenario to run on")
    parser.add_argument("--env_name", type=str, default='Dogfight', help="specify the name of environment")
    parser.add_argument("--experiment_name", type=str, default=f"1v3",
                        help="an identifier to distinguish different experiment.")
    parser.add_argument("--merged_critic", type=bool, default=False, help="use merged critic")
    parser.add_argument("--USE_BETA", type=bool, default=True, help="use beta distribution")

    parser.add_argument("--gamma", type=float, default=0.6,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gamma_long", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')

    parser.add_argument("--lr", type=float, default=1e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=1e-4,
                        help='critic learning rate (default: 5e-4)')

    all_args = parser.parse_known_args(args)[0]
    # WERF
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.share_policy = True

    # specific configs
    all_args.cuda = True
    all_args.episode_length = 1000
    # \files
    all_args.model_dir =None#r"D:\dogfight-sandbox-hg2\source\onpolicy\results\Dogfight\Dogfight\mappo\1v3\wandb\run-20220719_010421-2ojn5i6r\files"# r"D:\dogfight-sandbox-hg2\source\onpolicy\results\Dogfight\Dogfight\mappo\1v3\wandb\run-20220719_001634-2wmk4u1t\files"  # r"E:\OneDrive\DRL\light_mappo\onpolicy\results\MainEnv3agent\MainEnv3agent\mappo\MainEnergy100,10,1,1000m,shared,UAV_ALLOC_b,W_server=0.001,episode_length=200,PERIODS_PER_EPISODE=1,POSITION_SAMPLE_STEPS=1,ppo_epoch=5,\wandb\run-20220712_175002-1h0760ec\files"
    if all_args.model_dir is not None:
        model_dir = all_args.model_dir
        yml = yaml.full_load(open(model_dir + "/config.yaml"))
        dic = dict(zip([k for k in yml.keys()], [yml[k]['value'] if type(yml[k]) == dict else yml[k] for k in yml.keys()]))
        parser.set_defaults(**dic)
        all_args_curr = all_args
        all_args = parser.parse_args([])
        if all_args_curr.restore_only_pararms:
            all_args.model_dir = None
        else:
            all_args.model_dir = model_dir
    if not all_args.use_wandb:
        warnings.warn(f"{'NOT ' if not all_args.use_wandb else ' '}using wandb")
    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         id=all_args.model_dir.split("\\")[-2][-8:] if all_args.model_dir is not None else None,
                         resume="must " if all_args.model_dir is not None else None,
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                              str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
        all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    all_args.num_agents = envs.num_agents
    num_agents = all_args.num_agents
    all_args.num_agent_types = envs.env_list[0].num_agent_types
    all_args.num_agents_each_type = envs.env_list[0].num_agents_each_type  # short,long
    all_args.is_long_decision_agent = np.ones(all_args.num_agents)
    all_args.ID = wandb.run.id if wandb.run is not None else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    from onpolicy.runner.classified.env_runner import EnvRunner as Runner

    print(all_args)
    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
