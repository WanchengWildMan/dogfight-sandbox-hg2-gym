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

from onpolicy.config import get_config
from onpolicy.envs.env import Env
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from onpolicy.envs.main_env import MainEnv

"""Train script for MPEs."""


# def make_train_env(all_args):
#     return SubprocVecEnv(all_args)

def make_train_env(all_args):
    def get_env_fn(i):  # 返回能获取环境的函数
        def init_env(i):
            env = MainEnv(i, all_args)
            np.random.seed(all_args.seed)
            return env

        return init_env

    all_args.fn = [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
    return SubprocVecEnv(all_args)


def make_eval_env(all_args):
    return DummyVecEnv(all_args)


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='MainEnv', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    num_users = 16
    num_UAVs = 4
    num_BSs = 1
    WIDTH = 500
    parser.add_argument('--num_users', type=int, default=num_users, help="number of users")
    parser.add_argument('--num_UAVs', type=int, default=num_UAVs, help="number of UAVs and BS!@!")
    parser.add_argument("--num_BSs", type=int, default=num_BSs, help="number of BSs in UAVs")
    parser.add_argument("--ground_length", type=float, default=WIDTH, help="ground length")
    parser.add_argument("--ground_width", type=float, default=WIDTH, help="ground width")
    parser.add_argument("--B", type=float, default=5e6, help="bandwidth")
    parser.add_argument("--p_max", type=float, default=0.5, help="transmit power")
    parser.add_argument("--p_min", type=float, default=0.2, help="transmit power")
    parser.add_argument("--sigma2", type=float, default=1e-9, help="noise power")
    parser.add_argument("--T", type=float, default=100, help="time period")  # useless!!!!!

    parser.add_argument("--delta_T", type=float, default=1, help="time slot")
    parser.add_argument("--UAV_H", type=float, default=100, help="UAV height")
    parser.add_argument("--BS_H", type=float, default=10, help="BS height")
    parser.add_argument("--relay_H", type=float, default=500, help="BS height")

    parser.add_argument("--deg_users_min", type=float, default=0)
    parser.add_argument("--deg_users_max", type=float, default=2 * np.pi)
    parser.add_argument("--v_users_min", type=float, default=1)
    parser.add_argument("--v_users_max", type=float, default=4)
    parser.add_argument("--v_servers_max", type=float, default=20, help="maximum velocity of UAVs")
    parser.add_argument("--a_servers_max", type=float, default=5, help="maximum accelerate of UAVs")
    parser.add_argument("--v_users_noise_sigma2", type=float, default=1, help="maximum velocity of UAVs")
    parser.add_argument("--deg_users_noise_sigma2", type=float, default=1, help="maximum velocity of UAVs")
    parser.add_argument("--mu_v", type=float, default=0.2, help="maximum velocity of UAVs")
    parser.add_argument("--mu_deg", type=float, default=0.2, help="maximum velocity of UAVs")

    parser.add_argument("--f_user", type=float, default=1e9)
    parser.add_argument("--f_BS", type=float, default=20e9)
    parser.add_argument("--f_UAV", type=float, default=10e9)
    parser.add_argument("--f_server", type=float, default=10e9)

    parser.add_argument("--p_hover", type=float, default=0.5, help="power")
    parser.add_argument("--RC", type=float, default=1e-27, help="power")
    parser.add_argument("--Dmin", type=int, default=0.5e6, help="minimum data size")
    parser.add_argument("--Dmax", type=int, default=1.5e6, help="maximum data size")
    parser.add_argument("--Cmin", type=int, default=0.5e3, help="maximum data size")
    parser.add_argument("--Cmax", type=int, default=1.5e3, help="maximum data size")

    parser.add_argument("--UAV_R", type=int, default=np.inf, help="maximum data coverage")
    parser.add_argument("--BS_R", type=int, default=np.inf, help="maximum data coverage")
    parser.add_argument("--user_R", type=int, default=np.inf, help="maximum user coverage")

    parser.add_argument("--describe_step", type=int, default=0)

    parser.add_argument("--FDMA", type=bool, default=True, help="number of BSs.")
    parser.add_argument("--FIXED_p", type=bool, default=True, help="fixed power")
    parser.add_argument("--FIXED_b", type=bool, default=False, help="fixed band")
    parser.add_argument("--UAV_ALLOC_b", type=bool, default=True, help="=UAV allocate band")
    parser.add_argument("--W_server", type=float, default=0.001, help="weight of server")
    parser.add_argument("--W_self", type=float, default=1.0, help="weight of self")
    parser.add_argument("--W_TLE", type=float, default=1, help="weight of TLE")
    parser.add_argument("--W_TLE_BS", type=float, default=1, help="weight of TLE inBS")

    parser.add_argument("--DVFS", type=bool, default=True, help="if True, DVFS is used and minimizing energy")
    parser.add_argument("--INTELLIGENT_BS", type=bool, default=True, help="num_UAVs=num_UAVs_real+num_BS, BS is regarded as UAV")
    parser.add_argument("--CROWD", type=int, default=4, help="number of crowds.")
    parser.add_argument("--CROWD_R", type=float, default=80, help="radius of crowds.")
    parser.add_argument("--UAV_START_AT_O", type=bool, default=False, help="number of BSs.")
    parser.add_argument("--ROLLING_MEAN", type=bool, default=False, help="number of BSs.")
    parser.add_argument("--CASE", type=int, default=1, help="action cases")
    parser.add_argument("--REW_CLIP", type=float, default=10, help="action cases")
    parser.add_argument("--OPTIM_rho", type=bool, default=False, help="action cases")
    parser.add_argument("--E_FLY_WEIGHT", type=float, default=0.1, help="action cases")
    parser.add_argument("--SINGLE_CORE", type=bool, default=True, help="sum(f)**3")
    parser.add_argument("--TLE_PENAL_TO_E", type=bool, default=False, help="f<inf, E")
    parser.add_argument("--TLE_CLIP_EPS", type=float, default=0.1, help="0.01 for calc if T_off exceed")
    parser.add_argument("--REW_ONLY_TLE", type=bool, default=False, help="TLE reward")
    parser.add_argument("--REW_ONLY_R", type=bool, default=False, help="rate reward")
    parser.add_argument("--REW_ONLY_R_SELF", type=bool, default=False, help="rate reward")
    parser.add_argument("--REW_ONLY_D", type=bool, default=False, help="dist reward")
    parser.add_argument("--T_OFF_MEAN", type=bool, default=True, help="mean T_off to T_server_comp_req")
    parser.add_argument("--USE_BETA", type=bool, default=False, help="use beta distribution")
    parser.add_argument("--UAV_FREEZE", type=bool, default=False, help="UAV fixed position")
    parser.add_argument("--user_FREEZE", type=bool, default=False, help="users fixed position")
    parser.add_argument("--USE_ACCELERATE", type=bool, default=True, help="alpha fixed in")
    parser.add_argument("--LOCAL_COMP", type=bool, default=False, help="UAV fixed position")
    parser.add_argument("--FIX_alpha_STEP", type=int, default=1, help="alpha fixed in")
    parser.add_argument("--EXP_REW", type=bool, default=False, help="alpha fixed in")
    POSITION_SAMPLE_STEPS = 400000
    PERIODS_PER_EPISODE = 1
    parser.add_argument("--POSITION_SAMPLE_STEPS", type=int, default=POSITION_SAMPLE_STEPS, help="alpha fixed in")
    parser.add_argument("--PERIODS_PER_EPISODE", type=int, default=PERIODS_PER_EPISODE, help="alpha fixed in")

    parser.add_argument("--DEMO", type=bool, default=False, help="DEMO")
    parser.add_argument("--env_name", type=str, default='MainEnv', help="specify the name of environment")
    parser.add_argument("--experiment_name", type=str, default=f"MainEnergy{num_users},{num_UAVs},{num_BSs},{WIDTH}m,UAV_ALLOC_b,W_server=0.001,episode_length=200,PERIODS_PER_EPISODE={PERIODS_PER_EPISODE},POSITION_SAMPLE_STEPS={POSITION_SAMPLE_STEPS},ppo_epoch=5,",
                        help="an identifier to distinguish different experiment.")
    parser.add_argument("--merged_critic", type=bool, default=False, help="use merged critic")
    parser.add_argument("--use_attention", type=bool, default=False, help="use ATTENTION")

    parser.add_argument("--gamma", type=float, default=0.95,
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
    all_args.num_agents = all_args.num_UAVs + all_args.num_users
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents
    all_args.num_agent_types = 2
    all_args.num_agents_each_type = np.array([all_args.num_users, all_args.num_UAVs])  # short,long
    all_args.is_long_decision_agent = np.concatenate([np.zeros(all_args.num_users), np.ones(all_args.num_UAVs)])
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
    if all_args.share_policy:
        from onpolicy.runner.shared.env_runner import EnvRunner as Runner
    else:
        from onpolicy.runner.separated.env_runner import EnvRunner as Runner

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
