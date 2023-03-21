import math
import time

import numpy as np
import pygame as pg
import scipy
from PygameGUI import Screen
from PygameGUI.Sprites import Mat, Slider, Text
from numba import njit

from map_editor import MapEditor
from robot_sprite import RobotSprite

# robot = YouBot("192.168.88.22", offline=True)

map_size = (100, 70)
ang_res = 30
lidar_res = 50
scale = 5  # discrete per meter


def to_radians(ang_dec):
    ang_dec = ang_dec % ang_res
    if ang_dec < 0:
        ang_dec += ang_res
    return ang_dec / ang_res * (2 * math.pi)


def to_descrete(ang):
    ang = ang % (2 * math.pi)
    if ang < 0:
        ang += (2 * math.pi)
    return ang / (2 * math.pi) * (ang_res - 1)


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
    lidar_prob = np.zeros_like(prob_arr)
    for x in range(prob_arr.shape[0]):
        for y in range(prob_arr.shape[1]):
            for ang in range(prob_arr.shape[2]):
                dist_vect = c_space[x, y, ang, :]
                if np.linalg.norm(dist_vect) * np.linalg.norm(curr_lidar) != 0:
                    lidar_prob[x, y, ang] = ((1 + np.dot(dist_vect, curr_lidar) / (
                            np.linalg.norm(dist_vect) * np.linalg.norm(curr_lidar))) / 2) ** 1000
                else:
                    lidar_prob[x, y, ang] = 0
    buff = prob_arr * lidar_prob
    prob_arr = (buff) / np.sum(buff)
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
                np.linalg.norm(dist_vect) * np.linalg.norm(curr_lidar))) ** 3

    return prob_arr


@njit(fastmath=True)
def get_c_space(bin_map, c_space):
    dummy = np.zeros(map_size)
    for x in range(c_space.shape[0]):
        for y in range(c_space.shape[1]):
            for ang_dec in range(c_space.shape[2]):
                ang = ang_dec / ang_res * (2 * math.pi)
                dist_vect = np.zeros(lidar_res)
                raycast(x, y, ang, bin_map, dummy, dist_vect)
                c_space[x, y, ang_dec, :] = dist_vect
    return


def get_transition_matrix(transition_vector):
    transition_matrix = np.zeros((5, 5))
    # ind = tuple((np.ones((2)) * 2 + transition_vector).astype(np.int32))
    # transition_matrix[ind] = 1
    # mov_prob_dist = np.ones((3, 3, 3)) / 200
    # mov_prob_dist[:, 1, 1] = 1 / 60
    # mov_prob_dist[1, :, 1] = 1 / 60
    # mov_prob_dist[1, 1, :] = 1 / 60
    # mov_prob_dist[1, 1, 1] = 0
    # mov_prob_dist[1, 1, 1] = 1 - np.sum(mov_prob_dist)
    for i in range(5):
        for j in range(5):
            k = 1
            if i == 0 or j == 0 or i == 4 or j == 4:
                k = 0.2
            v = np.array((i, j)) - np.ones(2) * 2
            len_dif = abs(np.linalg.norm(transition_vector) - np.linalg.norm(v))
            if i == 2 and j == 2:
                transition_matrix[i, j] = (6 - len_dif * 3)
                continue
            transition_matrix[i, j] = ((1 + np.dot(v, transition_vector) / (
                    np.linalg.norm(transition_vector) * np.linalg.norm(v))) / 2) * (6 - len_dif * 3)
    transition_matrix = transition_matrix / np.sum(transition_matrix)
    return transition_matrix


def apply_movement_uncertainty_on_layer(extended_prob_arr, transition_vector, ang):
    rot = np.array([[math.cos(ang), -math.sin(ang)],
                    [math.sin(ang), math.cos(ang)]])
    new_vect = transition_vector @ rot
    return scipy.ndimage.convolve(extended_prob_arr,
                                  get_transition_matrix(new_vect))


def apply_movement_uncertainty(prob_arr, transition_vector):
    extended_prob_arr = np.concatenate((prob_arr[:, :, -5:], prob_arr, prob_arr[:, :, :5]), axis=2)
    if transition_vector[2] > ang_res / 2:
        transition_vector[2] -= ang_res
    if transition_vector[2] < -ang_res / 2:
        transition_vector[2] += ang_res
    if (np.abs(transition_vector) > 1).any():
        mult = np.max(np.abs(transition_vector[:2]))
        step = transition_vector / mult
        prev_step = np.zeros_like(step)

        for i in range(1, int(mult)+1):
            for ang in range(extended_prob_arr.shape[2]):
                extended_prob_arr[:, :, ang] = apply_movement_uncertainty_on_layer(extended_prob_arr[:, :, ang],
                                                                          (step * i - prev_step)[:2], to_radians(ang))
            prev_step = step * i
        return extended_prob_arr[:, :, 5:-5]
    else:
        for ang in range(extended_prob_arr.shape[2]):
            extended_prob_arr[:, :, ang] = apply_movement_uncertainty_on_layer(extended_prob_arr[:, :, ang],
                                                                               transition_vector[:2],
                                                                               to_radians(ang))
        return extended_prob_arr[:, :, 5:-5]


class Robot:
    def __init__(self, **kwargs):
        self.lidar_res = lidar_res
        self.increment = np.array([3.0, 3.0, 0])
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
curr_speed = np.array([0.0, 0.0, 0.0])


def update_keys(screen):
    global last_checked_pressed_keys
    global out
    global prob_arr
    global curr_speed
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
        rot += 0.02
    if pg.K_d in pressed_keys:
        rot -= 0.02
    move_speed[2] = rot

    side = 0
    if pg.K_q in pressed_keys:
        side += 1
    if pg.K_e in pressed_keys:
        side -= 1
    move_speed[1] = side * move_speed_val

    if pg.K_c in pressed_keys:
        move_speed = [0, 0, 0]
        robot.move_base(0, 0, 0)
        calc_txt.visible = True
        calc_txt2.visible = True
        screen.wait_step()
        screen.wait_step()
        get_c_space(map_editor.get_bin_map(), robot.c_space)
        prob_arr = get_probability_distribution(robot.lidar, robot.c_space, prob_arr)
        out = prob_arr[:, :, int(to_descrete(robot.increment[2]))]
        calc_txt.visible = False
        calc_txt2.visible = False

    if pg.K_p in pressed_keys:
        calc_txt.visible = True
        calc_txt2.visible = True
        screen.wait_step()
        screen.wait_step()
        prob_arr = get_probability_distribution(robot.lidar, robot.c_space, prob_arr)
        calc_txt.visible = False
        calc_txt2.visible = False

    if last_checked_pressed_keys != pressed_keys:
        robot.move_base(*move_speed)
        curr_speed = np.array(move_speed)
        last_checked_pressed_keys = pressed_keys[:]


def out_prob_mat():
    out = prob_arr[:, :, curr_prob_ang]
    dev = max(np.max(prob_arr), 0.00000000001)
    ret = (np.stack((out.T, out.T, out.T), axis=2)) * 255 / dev
    return ret


robot = Robot()
curr_prob_ang = 0


def change_prob_ang(val):
    global curr_prob_ang
    curr_prob_ang = int(val)


def get_prob_ang_txt():
    return str(round(to_radians(slider.val) * 360 / (2 * math.pi)))


def get_robot_pos_txt():
    return "current robot position: " + \
           str(round(robot.increment[0] * scale)) + "; " + \
           str(round(robot.increment[1] * scale)) + "; " + \
           str(round(robot.increment[2] % (2 * math.pi) * 360 / (2 * math.pi)))


prob_arr = np.zeros((*map_size, ang_res)).astype(np.float64)

out = prob_arr[:, :, int(to_descrete(robot.increment[2]))]

# GUI layout
screen = Screen(600, 800)
screen.lunch_separate_thread()
map_editor = MapEditor(screen, name="MapEditor", x=0, y=0, width=1, height=0.48, map_shape=map_size, robot=robot)
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
calc_txt.visible = False
calc_txt2.visible = False
robot_sp.par_surf = map_editor
robot_sp_guess.par_surf = map_editor

# prob_arr[int(robot.increment[0] * scale), int(robot.increment[1] * scale), int(to_descrete(robot.increment[2]))] = 1
prob_arr[:, :, :] = 1
prob_arr = prob_arr / np.sum(prob_arr)

prev_t = time.time()
curr_shift = np.array([0.0, 0.0])
ang_chng = 0

while True:
    robot_sp.dec_pos = (np.array(robot.increment[:2]) * scale * mat_scale_factor).astype(int)
    robot_sp.ang = -robot.increment[2]
    curr_guess = np.array(np.unravel_index(np.argmax(prob_arr, axis=None), prob_arr.shape))
    robot_sp_guess.dec_pos = (curr_guess[:2] * mat_scale_factor).astype(int)
    robot_sp_guess.ang = to_radians(curr_guess[2])
    print(curr_shift.shape, curr_speed.shape)
    curr_shift += curr_speed[:2] * int((time.time() - prev_t) * scale * 100)
    ang_chng += curr_speed[2] * (time.time() - prev_t)*90
    prev_t = time.time()
    if (curr_shift > 1).any():
        prob_arr = apply_movement_uncertainty(prob_arr, (*curr_shift, ang_chng))
        curr_shift = np.array([0.0, 0.0])
    if abs(ang_chng) > 1:
        prob_arr = np.concatenate((prob_arr[:, :, -1*int(ang_chng):], prob_arr[:, :, :-1*int(ang_chng)]), axis=2)
        ang_chng -= int(ang_chng)
        if time.time() - prev_t > 1:
            curr_speed[2] = 0
    robot.bin_map = map_editor.get_bin_map()
    update_keys(screen)
    time.sleep(0.01)
