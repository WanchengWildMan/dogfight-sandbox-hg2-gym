import numpy as np
from Machines import Aircraft
import harfang as hg

state_configs = {"AlliesAirCraft": [{"name": "get_move_vector", "unit": 100},
                                    {"name": "get_linear_speed", "unit": 100},
                                    {"name": "get_linear_acceleration", "unit": 1},
                                    {"name": "get_heading", "unit": 10},
                                    {"name": "get_position", "unit": 1000},
                                    {"name": "get_altitude", "unit": 1000},
                                    {"name": "get_health_level", "unit": 1},
                                    {"name": "get_machinegun_count", "unit": 1},
                                    {"name": "get_num_bullets", "unit": 1000},
                                    {"name": "get_brake_level", "unit": 1},
                                    {"name": "get_flaps_level", "unit": 1}
                                    ],
                 "EnermyAirCraft": [
                     {"name": "get_move_vector", "unit": 100},
                     {"name": "get_linear_speed", "unit": 100},
                     {"name": "get_linear_acceleration", "unit": 1},
                     {"name": "get_heading", "unit": 10},
                     {"name": "get_position", "unit": 1000},
                     {"name": "get_altitude", "unit": 1000},
                     {"name": "get_health_level", "unit": 1},
                 ],
                 "Missile": [
                     {"name": "get_move_vector", "unit": 100},
                     {"name": "get_linear_speed", "unit": 100},
                     {"name": "get_linear_acceleration", "unit": 1},
                     {"name": "get_heading", "unit": 10},
                     {"name": "get_position", "unit": 1000},
                     {"name": "get_altitude", "unit": 1000},
                 ]
                 }
unit = []


def get_vec(vec):
    if isinstance(vec, list):
        return np.concatenate([[vec[i].x, vec[i].y, vec[i].z] for i in range(len(vec))])
    return np.array([vec.x, vec.y, vec.z])


def get_obs(machine, attrs):
    obs = []
    for attr in attrs:
        f = getattr(machine, attr["name"])
        s = f()
        if not isinstance(s, hg.Vec3):
            obs.append([s / attr["unit"]])
        else:
            obs.append(get_vec(s) / attr["unit"])
    return np.concatenate(obs)
