# Copyright (C) 2018-2021 Eric Kernin, NWNC HARFANG.

import harfang as hg
import numpy as np
from gym import spaces
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

main_name = sys.argv.pop(0)


class HarFangDogFightEnv():
    def __init__(self, all_args):

        self.agent_type_action_mapping = json.load(open("scripts/agent_type_action_mapping.json"))
        self.num_agent_types = len(self.agent_type_action_mapping["actions_each_agent_type"])
        self.num_agents = 2
        self.num_agents_each_type = [self.num_agents // self.num_agent_types] * self.num_agent_types  # args
        self.input_status = {"SWITCH_ACTIVATION": False,
                             "NEXT_PILOT": False,
                             "INCREASE_HEALTH_LEVEL": False,
                             "DECREASE_HEALTH_LEVEL": False,
                             "INCREASE_THRUST_LEVEL": False,
                             "DECREASE_THRUST_LEVEL": False,
                             "SET_THRUST_LEVEL": False,
                             "INCREASE_BRAKE_LEVEL": False,
                             "DECREASE_BRAKE_LEVEL": False,
                             "INCREASE_FLAPS_LEVEL": False,
                             "DECREASE_FLAPS_LEVEL": False,
                             "ROLL_LEFT": False,
                             "ROLL_RIGHT": False,
                             "SET_ROLL": False,
                             "PITCH_UP": False,
                             "PITCH_DOWN": False,
                             "SET_PITCH": False,
                             "YAW_LEFT": False,
                             "YAW_RIGHT": False,
                             "SET_YAW": False,
                             "SWITCH_POST_COMBUSTION": False,
                             "NEXT_TARGET": False,
                             "SWITCH_GEAR": False,
                             "ACTIVATE_AUTOPILOT": False,
                             "ACTIVATE_IA": False,
                             "SWITCH_EASY_STEERING": False,
                             "FIRE_MACHINE_GUN": False,
                             "FIRE_MISSILE": False
                             }

        self.observation_spaces = []
        self.action_spaces = []
        # self.action_dim=[]
        self.action_tag = self.get_high_action(get_dim_tag=True)
        self.action_tag_cumsum = [[self.action_tag[t][i].cumsum() for i in range(self.num_agents_each_type[t])] for t in range(self.num_agent_types)]

        for agent_type in range(self.num_agent_types):
            self.action_spaces.append([spaces.Box(low=0, high=1, shape=self.action_tag_cumsum[agent_type][i][0]) for i in range(self.num_agents_each_type[agent_type])])
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
        if sys.platform == "linux" or sys.platform == "linux2":
            assetc_cmd = [path.join(getcwd(), "../", "bin", "assetc", "assetc"), "assets", "-quiet", "-progress"]
            dc.run_command(assetc_cmd)
        else:
            if Main.flag_OpenGL:
                dc.run_command("../bin/assetc/assetc assets -api GL -quiet -progress")
            else:
                dc.run_command("../bin/assetc/assetc assets -quiet -progress")

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

    def get_low_action(self, agent_id=None, agent_type=None, get_dim_tag=False):
        if agent_type is None:
            return [[self.get_low_action(i, agent_type) for i in range(self.num_agents)] for t in range(self.num_agent_types)]
        else:
            action_i = np.concatenate([
                self.speedup * 0,
                self.slowdown * 0,
                self.turnleft * 0,
                self.turnright * 0,
                self.rollleft * 0,
                self.pullup * 0,
                self.pitchdown * 0,
                self.missile * 0,
                self.gun * 0,
            ])
            return np.array([2] * 9) if get_dim_tag else action_i

    #
    def get_high_action(self, agent_id=None, agent_type=None, get_dim_tag=False):
        if agent_type is None:
            return [[self.get_high_action(i, agent_type) for i in range(self.num_agents_each_type[t])] for t in range(self.num_agent_types)]
        else:
            if agent_type == 0:
                action_i = np.concatenate([
                    self.speedup[agent_id] * 0,
                    self.slowdown[agent_id] * 0,
                    self.turnleft[agent_id] * 0,
                    self.turnright[agent_id] * 0,
                    self.rollleft[agent_id] * 0,
                    self.pullup[agent_id] * 0,
                    self.pitchdown[agent_id] * 0,
                ])
            else:
                action_i = np.concatenate([
                    self.missile[agent_id] * 0,
                    self.gun[agent_id] * 0,
                ])
            return [2] * 9 if get_dim_tag else action_i

    def get_high_action(self, agent_id=None, agent_type=None, get_dim_tag=False):
        if agent_type is None:
            return [[self.get_high_action(i, agent_type) for i in range(self.num_agents_each_type[t])] for t in range(self.num_agent_types)]
        else:
            if agent_type == 0:
                action_i = np.concatenate([
                    self.speedup[agent_id] * 0 + 1,
                    self.slowdown[agent_id] * 0 + 1,
                    self.turnleft[agent_id] * 0 + 1,
                    self.turnright[agent_id] * 0 + 1,
                    self.rollleft[agent_id] * 0 + 1,
                    self.pullup[agent_id] * 0 + 1,
                    self.pitchdown[agent_id] * 0 + 1,
                ])
            else:
                action_i = np.concatenate([
                    self.missile[agent_id] * 0 + 1,
                    self.gun[agent_id] * 0 + 1,
                ])
            return [2] * 9 if get_dim_tag else action_i

    """
    :param actions: [[],[]]
    """

    def parse_actions(self, actions):
        actions_parsed = [[] for i in range(self.num_agent_types)]
        for agent_type in range(self.num_agent_types):
            action_type = actions[agent_type]
            for agent_id in range(agent_type):
                a_i = action_type[agent_id]
                ind = 0
                tags = self.action_tag_cumsum[action_type][agent_id]
                actions_parsed_t_i = []
                # split to operations
                for ind in range(len(tags)):
                    st = 0 if ind == 0 else tags[ind - 1]
                    actions_parsed_t_i.append(a_i[st:tags[ind]])
                if agent_type == 0:
                    self.speedup[agent_id], self.slowdown[agent_id], self.turnleft[agent_id], self.turnright[agent_id], self.rollleft[agent_id], self.rollright[agent_id], self.pullup[agent_id], self.pitchdown[agent_id] = tuple(actions_parsed_t_i)
                else:
                    self.missile[agent_id], self.gun[agent_id] = tuple(actions_parsed)
                actions_parsed[agent_type].append(tuple(actions_parsed_t_i))
        return actions_parsed

    # ------------------- Main loop:
    def step(self, actions):
        # while not Main.flag_exit:
        # parse actions to inputs
        actions_parsed = self.parse_actions(actions)

        Main.update_inputs(agent_actions=[actions_parsed[t][0] for t in range(self.num_agent_types)])

        if (not Main.flag_client_update_mode) or ((not Main.flag_renderless) and Main.flag_client_ask_update_scene):
            Main.update()
        else:
            time.sleep(1 / 120)

        Main.update_window()

        # ----------------- Exit:

    def exit(self):
        if Main.flag_network_mode:
            netws.stop_server()

        hg.StopAllSources()
        hg.AudioShutdown()

        hg.RenderShutdown()
        hg.DestroyWindow(Main.win)
