from KUKA import YouBot

from map_editor import MapEditor
import math
from numba import njit
import numpy as np
import pygame as pg
import threading as thr
import time

from PygameGUI import Screen
from PygameGUI.Sprites import Mat, Slider


# robot = YouBot("192.168.88.22", offline=True)

map_size = (100, 70)
ang_res = 360
lidar_res = 50
scale = 5  # discrete per meter

def to_radians(ang_dec):
    ang_dec = ang_dec % ang_res
    if ang_dec < 0:
        ang_dec += ang_res
    return ang_dec/ang_res*(2*math.pi)

def to_descrete(ang):
    ang = ang % (2*math.pi)
    if ang < 0:
        ang += (2*math.pi)
    return ang / (2*math.pi) * (ang_res-1)


@njit(fastmath=True)
def raycast(x, y, ang, bin_map, chack_map, out_array):
    ang_step = math.pi * 240 / (lidar_res * 180)
    ang_offset = 120 * math.pi / 180 - math.pi / 2
    lidar_max = int(5.6 * scale)
    shape = bin_map.shape
    for i in range(lidar_res):
        out_val = 5.6
        for j in range(lidar_max):
            x_c = x + int(j * math.sin(ang - ang_offset + i * ang_step))
            y_c = y + int(j * math.cos(ang - ang_offset + i * ang_step))
            if not 0 < x_c < shape[0] or not 0 < y_c < shape[1]:
                continue
            chack_map[x_c, y_c] = 20
            if bin_map[x_c, y_c] == 1:
                out_val = j / scale
                break
        out_array[i] = out_val



@njit(fastmath=True)
def get_probability_distribution(curr_lidar, c_space, prob_arr):
    print("prob")
    for x in range(prob_arr.shape[0]):
        for y in range(prob_arr.shape[1]):
            for ang in range(prob_arr.shape[2]):
                dist_vect = c_space[x, y, ang, :]
                if np.linalg.norm(dist_vect) * np.linalg.norm(curr_lidar) != 0:
                    prob_arr[x, y, ang] = (np.dot(dist_vect, curr_lidar) / (np.linalg.norm(dist_vect) * np.linalg.norm(curr_lidar)))**100
                    #print(prob_arr[x, y, ang])
                else:
                    prob_arr[x, y, ang] = 0

    return prob_arr


# @njit(fastmath=True)
def get_random_distribution(curr_lidar, bin_map, prob_arr):
    particles = np.random.rand(100, 3)
    particles[:, 0] = (particles[:, 0] * map_size[0]).astype(int)
    particles[:, 1] = (particles[:, 1] * map_size[1]).astype(int)
    particles[:, 2] = (particles[:, 2] * ang_res).astype(int)
    particles = particles.astype(int)
    dummy = np.zeros(map_size)
    for p_i in range(100):
        dist_vect = np.zeros(lidar_res)
        p = particles[p_i]
        raycast(p[0], p[1], p[2], bin_map, dummy, dist_vect)
        prob_arr[p[0], p[1], p[2]] = (np.dot(dist_vect, curr_lidar) / (
                np.linalg.norm(dist_vect) * np.linalg.norm(curr_lidar)))**3

    return prob_arr


@njit(fastmath=True)
def get_c_space(bin_map, c_space):
    dummy = np.zeros(map_size)
    for x in range(c_space.shape[0]):
        for y in range(c_space.shape[1]):
            for ang_dec in range(c_space.shape[2]):
                ang = ang_dec/ang_res * (2 * math.pi)
                dist_vect = np.zeros(lidar_res)
                raycast(x, y, ang, bin_map, dummy, dist_vect)
                c_space[x, y, ang_dec, :] = dist_vect
    return



class Robot:
    def __init__(self, **kwargs):
        self.lidar_res = lidar_res
        self.increment = np.array([0.0, 0.0, 0.0])
        self.move_speed = np.array([0.0, 0.0, 0.0])
        self.bin_map = np.zeros(map_size)
        self.c_space = np.zeros((*map_size, ang_res, lidar_res)).astype(np.float64)

    def move_base(self, f=0.0, s=0.0, r=0.0):
        self.move_speed = np.array([f, s, r])

    def update(self):
        ang = self.increment[2]
        self.increment[:2] += self.move_speed[:2] @ np.array([[math.cos(ang), -math.sin(ang)],
                                                              [math.sin(ang), math.cos(ang)]])
        self.increment[2] += self.move_speed[2]

    def var_for_raycast(self):
        return int(self.increment[0] * scale), int(self.increment[1] * scale), self.increment[2], self.bin_map

    def update_overlay(self, overlay):
        raycast(*self.var_for_raycast(), overlay, np.zeros(self.lidar_res))

    @property
    def lidar(self):
        out_arr = np.zeros(self.lidar_res)
        raycast(*self.var_for_raycast(), np.zeros(map_size), out_arr)
        return out_arr


last_checked_pressed_keys = []
move_speed_val = 0.01 * scale


def update_keys(screen):
    global last_checked_pressed_keys
    global out
    global out_full
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
        rot += 0.1
    if pg.K_d in pressed_keys:
        rot -= 0.1
    move_speed[2] = rot

    side = 0
    if pg.K_q in pressed_keys:
        side += 1
    if pg.K_e in pressed_keys:
        side -= 1
    move_speed[1] = side * move_speed_val

    if pg.K_c in pressed_keys:
        print(thr.get_native_id())
        get_c_space(map_editor.get_bin_map(), robot.c_space)
        print(robot.lidar)
        out_full = get_probability_distribution(robot.lidar, robot.c_space, prob_arr)
        out = out_full[:, :, int(to_descrete(robot.increment[2]))]
    if pg.K_p in pressed_keys:
        out_full = get_probability_distribution(robot.lidar, robot.c_space, prob_arr)

    if last_checked_pressed_keys != pressed_keys:
        robot.move_base(*move_speed)
        last_checked_pressed_keys = pressed_keys[:]



prob_arr = np.zeros((*map_size, ang_res)).astype(np.float64)


def out_prob_mat():
    out = out_full[:, :, curr_prob_ang]
    ret = (np.stack((out.T,out.T,out.T), axis=2)+0.5)*100
    return ret

curr_prob_ang = 0
def change_prob_ang(val):
    global curr_prob_ang
    curr_prob_ang = int(val)

robot = Robot()
out = prob_arr[:, :, int(to_descrete(robot.increment[2]))]
out_full = prob_arr

screen = Screen(1000, 1400)
screen.lunch_separate_thread()
map_editor = screen.sprite(MapEditor, "MapEditor", x=0, y=0, width=1, height=0.45, map_shape=map_size, robot=robot)
screen.sprite(Mat, "Mat", x=0, y=0.45, width=1, height=0.45, cv_mat_stream=out_prob_mat)
screen.sprite(Slider, "prob_ang", x=0, y=0.9, width=1, height=0.1, min=0, max=ang_res-1, func=change_prob_ang)
screen.add_fps_indicator(color=(0,0,0))

while True:
    robot.bin_map = map_editor.get_bin_map()
    update_keys(screen)
    time.sleep(0.01)
