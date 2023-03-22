from constants import *
import numpy as np
from KUKA import YouBot
from sim_kuka import Robot



last_checked_pressed_keys = []
move_speed_val = 0.4 * scale
curr_speed = np.array([0.0, 0.0, 0.0])

conf_space = np.zeros((*map_size, ang_res, lidar_res))
prob_arr = np.zeros((*map_size, ang_res)).astype(np.float64)


curr_prob_ang = 0
curr_shift = np.array([0.0, 0.0])
ang_chng = 0
robot = Robot()
#robot = YouBot("192.168.88.21", offline=False, ssh=False, advanced=True, camera_enable=False, ros=False)
