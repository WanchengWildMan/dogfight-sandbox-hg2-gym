# Copyright (C) 2018-2021 Eric Kernin, NWNC HARFANG.

import harfang as hg
import numpy as np
from gym import spaces

import Machines
import Missions
from Missions import *
from master import Main
import data_converter as dc
import states
from Particles import *
import network_server as netws
import time
import sys
from os import path, getcwd
import json
from math import log, floor
from Machines import *
from scripts.state_config import *

main_name = sys.argv.pop(0)


class HarFangDogFightEnv():
    input_mapping_name = "AircraftUserInputsMapping"
    inputs_mapping_file = "scripts/aircraft_user_inputs_mapping.json"
    inputs_mapping = {}

    def __init__(self):

        # ----
        # speedup/slowdown
        # turn
        # LURD
        # missile
        # gun

        #
        # --------------- Inline arguments handler

        for i in range(len(sys.argv)):
            cmd = sys.argv[i]
            if cmd == "network_port":
                try:
                    nwport = int(sys.argv[i + 1])
                    netws.dogfight_network_port = nwport
                    print("Network port:" + str(nwport))
                except:
                    print("ERROR !!! Bad port format - network port must be a valid number !!!")
                i += 1
            elif cmd == "vr_mode":
                Main.flag_vr = True

        # ---------------- Read config file:

        file_name = "../config.json"

        file = open(file_name, "r")
        json_script = file.read()
        file.close()
        if json_script != "":
            script_parameters = json.loads(json_script)
            Main.flag_OpenGL = script_parameters["OpenGL"]
            Main.flag_fullscreen = script_parameters["FullScreen"]
            Main.resolution.x = script_parameters["Resolution"][0]
            Main.resolution.y = script_parameters["Resolution"][1]
            Main.antialiasing = script_parameters["AntiAliasing"]
            Main.flag_shadowmap = script_parameters["ShadowMap"]

        # --------------- Compile assets:
        print("Compiling assets...")
        # if sys.platform == "linux" or sys.platform == "linux2":
        #     assetc_cmd = [path.join(getcwd(), "../", "bin", "assetc", "assetc"), "assets", "-quiet", "-progress"]
        #     dc.run_command(assetc_cmd)
        # else:
        #     if Main.flag_OpenGL:
        #         dc.run_command("../bin/assetc/assetc assets -api GL -quiet -progress")
        #     else:
        #         dc.run_command("../bin/assetc/assetc assets -quiet -progress")

        # --------------- Init system

        hg.InputInit()
        hg.WindowSystemInit()

        hg.SetLogDetailed(False)

        res_x, res_y = int(Main.resolution.x), int(Main.resolution.y)

        hg.AddAssetsFolder(Main.assets_compiled)

        # ------------------- Setup output window

        def get_monitor_mode(width, height):
            monitors = hg.GetMonitors()
            for i in range(monitors.size()):
                monitor = monitors.at(i)
                f, monitorModes = hg.GetMonitorModes(monitor)
                if f:
                    for j in range(monitorModes.size()):
                        mode = monitorModes.at(j)
                        if mode.rect.ex == width and mode.rect.ey == height:
                            print("get_monitor_mode() : Width %d Height %d" % (mode.rect.ex, mode.rect.ey))
                            return monitor, j
            return None, 0

        Main.win = None
        if Main.flag_fullscreen:
            monitor, mode_id = get_monitor_mode(res_x, res_y)
            if monitor is not None:
                Main.win = hg.NewFullscreenWindow(monitor, mode_id)

        if Main.win is None:
            Main.win = hg.NewWindow(res_x, res_y)

        if Main.flag_OpenGL:
            hg.RenderInit(Main.win, hg.RT_OpenGL)
        else:
            hg.RenderInit(Main.win)

        alias_modes = [hg.RF_MSAA2X, hg.RF_MSAA4X, hg.RF_MSAA8X, hg.RF_MSAA16X]
        aa = alias_modes[min(3, floor(log(Main.antialiasing) / log(2)) - 1)]
        hg.RenderReset(res_x, res_y, hg.RF_VSync | aa | hg.RF_MaxAnisotropy)

        # -------------------- OpenVR initialization

        if Main.flag_vr:
            if not Main.setup_vr():
                sys.exit()

        # ------------------- Imgui for UI

        imgui_prg = hg.LoadProgramFromAssets('core/shader/imgui')
        imgui_img_prg = hg.LoadProgramFromAssets('core/shader/imgui_image')
        hg.ImGuiInit(10, imgui_prg, imgui_img_prg)

        # --------------------- Setup dogfight sim
        hg.AudioInit()
        Main.init_game()

        node = Main.scene.GetNode("platform.S400")
        nm = node.GetName()

        # rendering pipeline
        Main.pipeline = hg.CreateForwardPipeline()
        hg.ResetClock()

        # ------------------- Setup state:
        Main.current_state = states.init_menu_phase()
        [Main.update() for i in range(100)]

        self.agent_type_action_mapping = json.load(open("scripts/agent_type_action_mapping.json"))["actions_each_agent_type"]
        self.num_agent_types = len(self.agent_type_action_mapping)
        self.num_agents = 2
        self.num_agents_each_type = [self.num_agents // self.num_agent_types] * self.num_agent_types  # args
        self.num_opr_each_type = [len(self.agent_type_action_mapping[t]) for t in range(self.num_agent_types)]
        self.action_key_set = []
        for t in range(self.num_agent_types):
            self.action_key_set += self.agent_type_action_mapping[t]
        if self.inputs_mapping_file != "":
            self.load_inputs_mapping_file(self.inputs_mapping_file)

        self.num_keys = len(self.action_key_set)
        self.num_keys_each_type = [len(self.agent_type_action_mapping[t]) for t in range(self.num_agent_types)]
        self.input_status = dict(zip(self.action_key_set, [False] * self.num_keys))

        self.observation_spaces = []  # len=number of agents for enc_wrapper to compute share_obs_space
        self.action_spaces = []  # len=number of agents for enc_wrapper to compute share_obs_space
        self.obs_dim = []
        self.action_dim = []
        # self.action_dim=[]
        self.action_tag = self.get_high_action(get_dim_tag=True)
        self.action_tag_cumsum = [[self.action_tag[t][i].cumsum() for i in range(self.num_agents_each_type[t])] for t in range(self.num_agent_types)]

        for agent_type in range(self.num_agent_types):
            for agent_id in range(self.num_agents_each_type[agent_type]):
                self.action_spaces.append(spaces.Box(low=self.get_low_action(agent_id, agent_type), high=self.get_high_action(agent_id, agent_type)))
                self.observation_spaces.append(spaces.Box(low=self.get_low_state(agent_id, agent_type), high=self.get_high_state(agent_id, agent_type)))
                self.obs_dim.append(self.observation(agent_id, agent_type).shape[0])  # self.get_obs().shape[0]  # 设置智能体的观测纬度
                self.action_dim.append(self.action_spaces[-1].shape[0])

    @classmethod
    def load_inputs_mapping_file(cls, file_name):
        file = hg.OpenText(file_name)
        if not file:
            print("ERROR - Can't open json file : " + file_name)
        else:
            json_script = hg.ReadString(file)
            hg.Close(file)
            if json_script != "":
                cls.inputs_mapping_encoded = json.loads(json_script)
                im = cls.inputs_mapping_encoded["AircraftUserInputsMapping"]
                cmode_decode = {}
                for cmode, maps in im.items():
                    maps_decode = {}
                    for cmd, hg_enum in maps.items():
                        if hg_enum != "":
                            if not hg_enum.isdigit():
                                try:
                                    exec("maps_decode['%s'] = hg.%s" % (cmd, hg_enum))
                                except AttributeError:
                                    print("ERROR - Harfang Enum not implemented ! - " + "hg." + hg_enum)
                                    maps_decode[cmd] = ""
                            else:
                                maps_decode[cmd] = int(hg_enum)
                        else:
                            maps_decode[cmd] = ""
                    cmode_decode[cmode] = maps_decode
                cls.inputs_mapping = {cls.input_mapping_name: cmode_decode}
            else:
                print("ERROR - Inputs parameters empty : " + file_name)

    def reset(self):
        Main.fading_to_next_state = True
        return self.observation()

    def get_low_state(self, agent_id=None, agent_type=None, get_dim_tag=False):
        return self.observation(agent_id, agent_type)

    def get_high_state(self, agent_id=None, agent_type=None, get_dim_tag=False):
        return self.observation(agent_id, agent_type) + 1

    def get_low_action(self, agent_id=None, agent_type=None, get_dim_tag=False):
        if agent_type is None:
            return [[self.get_low_action(i, t, get_dim_tag) for i in range(self.num_agents)] for t in range(self.num_agent_types)]
        else:
            return np.array([2] * self.num_keys_each_type[agent_type]) if get_dim_tag else np.array([0, 0] * self.num_opr_each_type[agent_type])

    def get_high_action(self, agent_id=None, agent_type=None, get_dim_tag=False):
        if agent_type is None:
            return [[self.get_high_action(i, t, get_dim_tag) for i in range(self.num_agents_each_type[t])] for t in range(self.num_agent_types)]
        else:
            return np.array([2] * self.num_keys_each_type[agent_type]) if get_dim_tag else np.array([1, 1] * self.num_opr_each_type[agent_type])

    """
    :param actions: [[],[]]
    """

    def parse_actions(self, actions):
        for agent_type in range(self.num_agent_types):
            action_type = actions[agent_type]
            for agent_id in range(self.num_agents_each_type[agent_type]):
                a_i = action_type[agent_id]
                tags = self.action_tag_cumsum[agent_type][agent_id]
                actions_parsed_t_i = []
                action_sum = []
                # split to operations
                for ind in range(len(tags)):
                    st = 0 if ind == 0 else tags[ind - 1]
                    # actions_parsed_t_i.append(a_i[st:tags[ind]][1] > a_i[st:tags[ind]][0])
                    action_sum.append(a_i[st:tags[ind]][1] + a_i[st:tags[ind]][0])
                # self.input_status.update(dict(zip(self.agent_type_action_mapping[agent_type], actions_parsed_t_i)))
                actions_parsed_t_i = np.zeros(len(tags))
                actions_parsed_t_i[np.array(action_sum).argmax()] = 1  # !!!!!
                self.input_status.update(dict(zip(self.agent_type_action_mapping[agent_type], actions_parsed_t_i)))

    def observation(self, agent_id=None, agent_type=None):
        if agent_type is None:
            obs = []
            [[obs.append(self.observation(i, t)) for i in range(self.num_agents_each_type[t])] for t in range(self.num_agent_types)]
            return obs
        else:
            obs = []
            for machine in Main.players_allies:
                if isinstance(machine, Aircraft):
                    obs += [get_obs(machine, state_configs["AlliesAirCraft"])]
            for machine in Main.players_ennemies:
                if isinstance(machine, Aircraft):
                    obs += [get_obs(machine, state_configs["EnermyAirCraft"])]
            for en_missiles in Main.missiles_ennemies:
                for missile in en_missiles:
                    obs += [get_obs(missile, state_configs["Missile"])]
            if obs != []:
                obs = np.concatenate(obs)
            if len(obs) == 70:
                print("!!!!!!!!!!")
            return obs

    def get_rewards(self):
        first = 0
        aircrafts = []
        for machine in Destroyable_Machine.machines_list:
            if isinstance(machine, Aircraft):
                aircrafts.append(machine)
        rewards = [[1 - 20 * Missions.get_current_mission().failed] for i in range(self.num_agents)]
        # try:
        #     for agent in range(self.num_agents):
        #         my = aircrafts[0]
        #         pos_my = get_vec(my.get_position())
        #         pos_en = np.zeros((len(aircrafts) - 1, 3))
        #         for en, aircraft in enumerate(aircrafts[1:]):
        #             pos_en[en] = get_vec(aircraft.get_position())
        #         rewards[agent] = [-(min((((pos_en - pos_my) ** 2).sum(1)) ** 0.5))]
        # except:
        #     pass
        return rewards

    # ------------------- Main loop:
    def step(self, actions):
        # while not Main.flag_exit:
        # parse actions to inputs
        self.parse_actions(actions)

        Main.update_inputs(input_status=self.input_status)

        if (not Main.flag_client_update_mode) or ((not Main.flag_renderless) and Main.flag_client_ask_update_scene):
            Main.update()
        else:
            time.sleep(1 / 120)

        Main.update_window()
        obs_next = []
        try:
            obs_next = self.observation()
        except:
            pass
        done = False
        if Missions.get_current_mission().failed:
            done = True

        return obs_next, self.get_rewards(), [done] * self.num_agents, [{} for i in range(self.num_agents)]

        # ----------------- Exit:

    def exit(self):
        if Main.flag_network_mode:
            netws.stop_server()

        hg.StopAllSources()
        hg.AudioShutdown()

        hg.RenderShutdown()
        hg.DestroyWindow(Main.win)

# env = HarFangDogFightEnv()
# while True:
#     a = env.get_high_action()
#     env.step([[a[t][0] * 0 * np.random.random(a[t][0].shape)] for t in range(env.num_agent_types)])
