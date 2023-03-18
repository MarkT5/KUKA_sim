from PygameGUI.Sprites import Sprite
import numpy as np
import pygame as pg


class MapEditor(Sprite):
    def __init__(self, par_surf, /, map_shape=None, robot=None, **kwargs):
        super().__init__(par_surf, **kwargs)
        self.map_shape = map_shape
        self.robot = robot
        self.discrete_per_pixel_width = self.map_shape[0] / self.width
        self.discrete_per_pixel_height = self.map_shape[1] / self.height
        self.discrete_wh = self.width / self.map_shape[0], self.height / self.map_shape[1]
        self.full_map = np.zeros(self.map_shape).astype(np.uint16)
        self.brush = 5
        self.robot_pos = robot.increment
        self.draw_robot = False
        self.old_wheel = 0
        self.overlay = np.zeros(map_shape)

    def change_brush_size(self, val):
        self.brush = int(val)

    def fill_map(self, x, y, setter=1):
        bs = self.brush
        if bs == 1:
            self.full_map[x, y] = setter
            return
        for di in range(-bs, bs):
            for dj in range(-bs, bs):
                i, j = x + di, y + dj
                if di ** 2 + dj ** 2 < bs ** 2 and self.map_shape[0] > i >= 0 and self.map_shape[1] > j >= 0:
                    self.full_map[i, j] = setter

    def release(self, *args, **kwargs):
        pass

    def update(self):
        self.robot.update()
        self.overlay = np.zeros(self.map_shape)
        self.robot.update_overlay(self.overlay)
        delta = self.old_wheel - self.par_surf.mouse_wheel_pos
        self.brush = max(1, min(20, self.brush + delta))
        self.old_wheel = self.par_surf.mouse_wheel_pos

    def pressed(self, *args, **kwargs):
        pos = kwargs["mouse_pos"]
        x = int(pos[0] * self.discrete_per_pixel_width)
        y = int(pos[1] * self.discrete_per_pixel_height)
        if x >= self.full_map.shape[0] or y >= self.full_map.shape[1]:
            return
        if kwargs["btn_id"] == 1:
            self.fill_map(x, y, 1)
        elif kwargs["btn_id"] == 3:
            self.fill_map(x, y, 0)

    def get_bin_map(self):
        return self.full_map == 1

    def dragged(self, *args, **kwargs):
        setter = 1
        if kwargs["btn_id"] == 3:
            setter = 0
        pos = kwargs["mouse_pos"]
        x = int(pos[0] * self.discrete_per_pixel_width)
        y = int(pos[1] * self.discrete_per_pixel_height)
        if x >= self.full_map.shape[0] or y >= self.full_map.shape[1]:
            return
        self.fill_map(x, y, setter)

    def discrete_rect(self, i, j):
        x = i * self.discrete_wh[0]
        y = j * self.discrete_wh[1]
        return (x, y, *self.discrete_wh)

    def get_prepared_surface(self):
        return pg.surfarray.make_surface((self.full_map - 1) * -255 + self.overlay)

    def draw(self):
        self.surface.blit(
                    pg.transform.scale(
                        self.get_prepared_surface(), (self.width, self.height)), (0, 0))

        if self.draw_robot:
            pg.draw.circle(self.surface, (0, 255, 0), self.robot_pos[:2], 30)
