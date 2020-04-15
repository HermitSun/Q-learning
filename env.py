from const import *
import time


def get_env_feedback(state: int, action: str) -> (int, int):
    new_state = TERMINATED
    reward = 0
    if Actions[action] == Actions.RIGHT:
        if state == MAX_DISTANCE:
            reward = 1
        else:
            new_state = state + 1
    else:
        if state == 0:
            new_state = 0
        else:
            new_state = state - 1
    return new_state, reward


def update_env(state: int, episode: int, step_counter: int) -> None:
    env_list = ['-'] * (ENV_SIZE - 1) + ['T']
    if state == TERMINATED:
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)
