from const import *
from env import update_env, get_env_feedback
from epsilon import linear, exponent, slider
import numpy as np
import pandas as pd
from typing import Callable

np.random.seed(7)  # reproducible


def build_q_table(size: int, actions: type(Actions)) -> pd.DataFrame:
    action_names = actions.__members__
    table = pd.DataFrame(
        np.zeros((size, len(action_names))),
        columns=action_names.keys(),
    )
    return table


def choose_best_action(
        state: int,
        actions: type(Actions),
        q_table: pd.DataFrame,
        epsilon: int = EPSILON
) -> str:
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() < epsilon) or ((state_actions == 0).all()):
        action_name = np.random.choice(list(actions.__members__))
    else:
        action_name = state_actions.idxmax()
    return action_name


def rl(strategy: Callable[[int, float], float]) -> pd.DataFrame:
    q_table = build_q_table(ENV_SIZE, Actions)
    epsilon = EPSILON
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0
        has_terminated = False
        update_env(state, episode, step_counter)
        while not has_terminated:
            action = choose_best_action(state, Actions, q_table, epsilon)
            new_state, reward = get_env_feedback(state, action)

            q_predict = q_table.loc[state, action]
            if new_state != TERMINATED:
                q_target = reward + GAMMA * q_table.iloc[new_state, :].max()
            else:
                q_target = reward
                has_terminated = True
            q_table.loc[state, action] += ALPHA * (q_target - q_predict)

            state = new_state
            update_env(state, episode, step_counter + 1)
            step_counter += 1

        epsilon = strategy(episode, epsilon)

    return q_table


q_table_final = rl(linear)
print('\r\nQ-table:\n')
print(q_table_final)
