from for_gui import *
from variables import *



v.prob_arr[:, :, :] = 1
v.prob_arr = v.prob_arr / np.sum(v.prob_arr)
out_full = v.prob_arr

# map difinition
bool_map = np.zeros(map_size[:2])
bool_map[0, :] = 1
bool_map[:, 0] = 1
bool_map[-1, :] = 1
bool_map[:, -1] = 1
bool_map[int(5 * scale):, int(1.5 * scale)] = 1
bool_map[int(5 * scale), :int(1.5 * scale)] = 1
bool_map[int(1.5 * scale), int(-1.5 * scale):] = 1
map_editor.full_map = bool_map

if not isinstance(v.robot, Robot):
    robot_sp.offset = np.array([15, 26, 0]) * np.array((*mat_scale_factor, 1))

prev_t = time.time()

while True:
    v.curr_shift += np.array(v.robot.move_speed[:2]) * (time.time() - prev_t) * scale*0.9
    v.ang_chng += v.robot.move_speed[2] * (time.time() - prev_t)*5#*7.8
    prev_t = time.time()
    if (np.abs(v.curr_shift) > 1).any():
        v.prob_arr = apply_movement_uncertainty(v.prob_arr, [*v.curr_shift, v.ang_chng])
        v.curr_shift = np.array([0.0, 0.0])
    if abs(v.ang_chng) > 1:
        v.prob_arr = np.concatenate((v.prob_arr[:, :, -1*int(v.ang_chng):], v.prob_arr[:, :, :-1*int(v.ang_chng)]), axis=2)
        v.ang_chng -= int(v.ang_chng)
        if time.time() - prev_t > 1:
            v.robot.move_speed[2] = 0
    if isinstance(v.robot, Robot):
        v.robot.bin_map = map_editor.get_bin_map()
        v.robot.update()
    update_hot_keys(screen)
    time.sleep(0.01)
