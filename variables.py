from constants import *
import numpy as np
from KUKA import YouBot
from sim_kuka import Robot


class Vars():
    def __init__(self):
        self.curr_speed = np.array([0.0, 0.0, 0.0])
        self.conf_space = np.zeros((*map_size, ang_res, lidar_res))
        self.prob_arr = np.zeros((*map_size, ang_res)).astype(np.float64)


        self.curr_shift = np.array([0.0, 0.0])
        self.ang_chng = 0
        #self.robot = Robot()
        self.robot = YouBot("192.168.88.21", offline=False, ssh=False, advanced=False, camera_enable=False, ros=True)

v = Vars()

