from for_gui import *




prob_arr[:, :, :] = 1
prob_arr = prob_arr / np.sum(prob_arr)
out_full = prob_arr

# map difinition
bool_map = np.zeros(map_size[:2])
bool_map[0, :] = 1
bool_map[:, 0] = 1
bool_map[-1, :] = 1
bool_map[:, -1] = 1
bool_map[int(5 * scale):, int(1.5 * scale)] = 1
bool_map[int(5 * scale), :int(1.5 * scale)] = 1
map_editor.full_map = bool_map


robot_sp.offset = np.array([15, 26, 0]) * np.array((*mat_scale_factor, 1))

prev_t = time.time()

while True:
    robot_sp.dec_pos = (np.array(robot.increment[:2]) * scale * mat_scale_factor).astype(int)
    robot_sp.ang = -robot.increment[2]
    curr_guess = np.array(np.unravel_index(np.argmax(prob_arr, axis=None), prob_arr.shape))
    robot_sp_guess.dec_pos = (curr_guess[:2] * mat_scale_factor).astype(int)
    robot_sp_guess.ang = to_radians(curr_guess[2])
    curr_shift += curr_spreed[:2] * (time.time() - prev_t)

    prev_t = time.time()
    if (curr_shift > 1).any():
        prob_arr = apply_movement_uncertainty(prob_arr, robot_discrete_pos - old_robot_discrete_pos)
        out_full = np.copy(prob_arr)
        old_robot_discrete_pos = np.copy(robot_discrete_pos)
    robot_discrete_pos = curr_shift[:2]

    # np.array((*(np.array(robot.increment[:2]) * scale).astype(int), int(to_descrete(-robot.increment[2]))))

    update_keys(screen)
    time.sleep(0.01)
