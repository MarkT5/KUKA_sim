from constants import *
import numpy as np
import math
from Markov import *

class Robot:
    def __init__(self, **kwargs):
        self.lidar_res = lidar_res
        self.increment = np.array([0.0, 0.0, 0])
        self.move_speed = np.array([0.0, 0.0, 0.0])
        self.bin_map = np.zeros(map_size)
        self.c_space = np.zeros((*map_size, ang_res, lidar_res)).astype(np.float64)

    def move_base(self, f=0.0, s=0.0, r=0.0):
        self.move_speed = np.array([f, -s, r])

    def update(self):
        ang = -self.increment[2]
        self.increment[:2] += (self.move_speed[:2]*0.005) @ np.array([[math.cos(ang), -math.sin(ang)],
                                                              [math.sin(ang), math.cos(ang)]])
        self.increment[2] -= self.move_speed[2]*0.01

    def var_for_raycast(self):
        return int(self.increment[0] * scale), int(self.increment[1] * scale), -1*self.increment[2], self.bin_map

    def update_overlay(self, overlay):
        raycast(*self.var_for_raycast(), overlay, np.zeros(self.lidar_res))

    @property
    def lidar(self):
        out_arr = np.zeros(self.lidar_res)
        raycast(*self.var_for_raycast(), np.zeros(map_size), out_arr)
        return [0,0,0], out_arr
