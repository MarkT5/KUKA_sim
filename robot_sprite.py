from PygameGUI.Sprites import Sprite
import pygame as pg
import math
import numpy as np
from map_editor import MapEditor


class RobotSprite(Sprite):
    def __init__(self, par_surf, /, **kwargs):
        super().__init__(par_surf, **kwargs)
        self.dec_pos = np.array([self.x, self.y])
        self.ang = 0
        self.offset = np.array([0, 0, 0])

    def draw(self):
        self.x, self.y = self.dec_pos+self.offset[:2]
        a = self.ang + math.pi / 2 + self.offset[2]
        rot = np.array([[math.cos(a), -math.sin(a)],
                        [math.sin(a), math.cos(a)]])
        points = np.array([[0, 20], [-20, -30], [20, -30]]) @ rot + self.dec_pos+self.offset[:2]
        if isinstance(self.par_surf, MapEditor):
            pg.draw.polygon(self.par_surf.workspace, self.color, points)
            self.surface.blit(self.par_surf.workspace, (0, 0))
        else:
            pg.draw.polygon(self.surface, self.color, points)