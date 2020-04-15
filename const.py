from enum import Enum

ENV_SIZE = 6  # the length of the 1 dimensional world
REACHABLE = 1  # how far can the intelligence get the award
MAX_DISTANCE = ENV_SIZE - REACHABLE

EPSILON = 1
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3


class Actions(Enum):
    LEFT = 'left',
    RIGHT = 'right'


TERMINATED = -1
