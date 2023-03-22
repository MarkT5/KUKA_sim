from constants import *
import math


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