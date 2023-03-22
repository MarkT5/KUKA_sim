import math
import time

import numpy as np
import pygame as pg
import scipy
from PygameGUI import Screen
from PygameGUI.Sprites import Mat, Slider, Text
from numba import njit
from robot_sprite import RobotSprite
from map_editor import MapEditor
from KUKA import YouBot
from Markov import *
from constants import *
from variables import *
from convertion import *
from PygameGUI.Sprites import Sprite

last_checked_pressed_keys = []
curr_prob_ang = 0

def update_hot_keys(screen):
    global last_checked_pressed_keys
    global v
    pressed_keys = screen.pressed_keys

    if pg.K_c in pressed_keys:
        move_speed = [0, 0, 0]
        v.robot.move_base(0, 0, 0)
        calc_txt.visible = True
        calc_txt2.visible = True
        screen.wait_step()
        screen.wait_step()
        get_c_space(map_editor.get_bin_map(), v.conf_space)
        if isinstance(v.robot, Robot):
            lid= v.robot.lidar[1]
        else:
            lid = np.array(v.robot.lidar[1][0:680:17])
        v.prob_arr = get_probability_distribution(lid, v.conf_space, v.prob_arr)
        calc_txt.visible = False
        calc_txt2.visible = False

    if pg.K_p in pressed_keys:
        calc_txt.visible = True
        calc_txt2.visible = True
        screen.wait_step()
        screen.wait_step()
        if isinstance(v.robot, Robot):
            lid= v.robot.lidar[1]
        else:
            lid = np.array(v.robot.lidar[1][0:680:17])
        v.prob_arr = get_probability_distribution(lid, v.conf_space, v.prob_arr)
        calc_txt.visible = False
        calc_txt2.visible = False

    if last_checked_pressed_keys != pressed_keys:
        last_checked_pressed_keys = pressed_keys[:]


def update_mov_keys(screen):
    global last_checked_pressed_keys
    global v
    pressed_keys = screen.pressed_keys
    move_speed = np.array([0, 0, 0.0])
    fov = 0
    if pg.K_w in pressed_keys:
        fov += 1
    if pg.K_s in pressed_keys:
        fov -= 1
    move_speed[0] = fov * move_speed_val

    rot = 0
    if pg.K_a in pressed_keys:
        rot += 0.7
    if pg.K_d in pressed_keys:
        rot -= 0.7
    move_speed[2] = rot

    side = 0
    if pg.K_q in pressed_keys:
        side += 1
    if pg.K_e in pressed_keys:
        side -= 1
    move_speed[1] = side * move_speed_val

    if last_checked_pressed_keys != pressed_keys:
        v.robot.move_base(*move_speed)
        v.curr_speed = np.array(move_speed)
        last_checked_pressed_keys = pressed_keys[:]
    if len(pressed_keys) == 0 and (v.robot.move_speed[0] != 0 or v.robot.move_speed[1] != 0 or v.robot.move_speed[2] != 0):
        v.robot.move_base(*move_speed)



def out_prob_mat():
    out = v.prob_arr[:, :, curr_prob_ang]
    ret = (np.stack((out.T, out.T, out.T), axis=2)) * 255 / np.max(v.prob_arr)
    return ret


def change_prob_ang(val):
    global curr_prob_ang
    curr_prob_ang = int(val)

def get_prob_ang_txt():
    return str(round(to_radians(slider.val) * 360 / (2 * math.pi)))


def get_robot_pos_txt():
    return "current robot position: " + \
        str(round(np.array(v.robot.increment[0]) * scale)) + "; " + \
        str(round(np.array(v.robot.increment[1]) * scale)) + "; " + \
        str(round(v.robot.increment[2] % (2 * math.pi) * 360 / (2 * math.pi)))

class Updater(Sprite):
    def __init__(self, par_surf, **kwargs):
        super().__init__(par_surf, **kwargs)

    def draw(self):
        pass
    def update(self):
        robot_sp.dec_pos = (np.array(v.robot.increment[:2]) * scale * mat_scale_factor).astype(int)
        print(v.robot.increment)
        robot_sp.ang = -v.robot.increment[2]
        curr_guess = np.array(np.unravel_index(np.argmax(v.prob_arr, axis=None), v.prob_arr.shape))
        robot_sp_guess.dec_pos = (curr_guess[:2] * mat_scale_factor).astype(int)
        robot_sp_guess.ang = to_radians(curr_guess[2])
        update_mov_keys(screen)



screen = Screen(1200, 1600)
screen.lunch_separate_thread()
screen.pause_update.acquire()
map_editor = MapEditor(screen, name="MapEditor", x=0, y=0, width=1, height=0.48, map_shape=map_size, robot=v.robot)
mat_scale_factor = Mat(screen, name="Mat", x=0, y=0.48, width=1, height=0.48, cv_mat_stream=out_prob_mat).scale_factor
slider = Slider(screen, name="prob_ang", x=0.22, y=0.97, width=0.7, height=0.03, min=0, max=ang_res - 1,
                func=change_prob_ang)
Text(screen, name="slider_label", inp_text=lambda *args: "angle in prob map:", x=0.01, y=0.97)
Text(screen, name="slider_val_label", inp_text=get_prob_ang_txt, x=0.93, y=0.97)
Text(screen, name="robot_pos", inp_text=get_robot_pos_txt, x=0.01, y=0.01, color=(0, 0, 0))
calc_txt = Text(screen, name="Calculating", inp_text=lambda *args: "Calculating...",
                x=0.25, y=0.6, font_size=40, color=(255, 0, 0))
calc_txt2 = Text(screen, name="Calculating2", inp_text=lambda *args: "(this may take a while)",
                 x=0.02, y=0.67, font_size=40, color=(255, 0, 0))
robot_sp = RobotSprite(screen, name="robot", x=0.5, y=0.5, color=(0, 255, 0))
robot_sp_guess = RobotSprite(screen, name="robot_guess", x=0.5, y=0.5, color=(0, 0, 255))
upd = Updater(screen, name="upd")
calc_txt.visible = False
calc_txt2.visible = False
screen.pause_update.release()
