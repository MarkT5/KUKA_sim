import math

import numpy as np
import scipy
from numba import njit
from constants import *
from convertion import *

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
            chack_map[x_c, y_c] = 20
            if bin_map[x_c, y_c] == 1 or not (0 < x_c < shape[0]) or not (0 < y_c < shape[1]):
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
                            np.linalg.norm(dist_vect) * np.linalg.norm(curr_lidar))) / 2) ** 100
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
    extended_prob_arr = prob_arr
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
    else:
        for ang in range(extended_prob_arr.shape[2]):
            extended_prob_arr[:, :, ang] = apply_movement_uncertainty_on_layer(extended_prob_arr[:, :, ang],
                                                                               transition_vector[:2],
                                                                               to_radians(ang))
    return extended_prob_arr#[:, :, 5:-5]