from typing import Union
import numpy as np


class EnvArgs_5x5:
    env_size: Union[list, tuple, np.ndarray] = (5, 5)
    start_state: Union[list, tuple, np.ndarray] = (0, 3)
    target_state: Union[list, tuple, np.ndarray] = (2, 3)
    forbidden_states: list = [(1, 1), (1, 4), (2, 2), (2, 1), (3, 3), (1, 3)]
    reward_target: float = 10
    reward_forbidden: float = -5
    reward_step: float = -1
    action_space: list = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]
    debug: bool = False
    animation_interval: float = 0.2

class EnvArgs_20x20:
    env_size: Union[list, tuple, np.ndarray] = (20, 20)
    start_state: Union[list, tuple, np.ndarray] = (1, 19)
    target_state: Union[list, tuple, np.ndarray] = (19, 19)
    forbidden_states: list = [(0, 2), (0, 5), (0, 11), (0, 12), (0, 13), (0, 15), (0, 18), (0, 19),
                              (1, 5), (1, 9), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (2, 2), 
                              (2, 3), (2, 5), (2, 12), (2, 16), (2, 17), (3, 2), (3, 12), (4, 3), 
                              (4, 6), (4, 7), (4, 10), (4, 11), (4, 15), (4, 17), (5, 6), (5, 12), 
                              (5, 13), (5, 16), (5, 18), (6, 7), (6, 10), (6, 16), (6, 18), (7, 11),
                              (7, 8), (7, 9), (7, 12), (7, 17), (8, 3), (8, 10), (8, 13), (8, 15), 
                              (8, 16), (8, 17), (9, 1), (9, 2), (9, 4), (9, 6), (9, 7), (9, 10), 
                              (9, 14), (9, 16), (9, 17), (9, 18), (10, 14), (10, 17), (10, 18),
                              (11, 4), (11, 8), (11, 15), (11, 18), (11, 19), (12, 3), (12, 6), 
                              (12, 8), (12, 11), (12, 13), (12, 14), (12, 15), (12, 17), (13, 4), 
                              (13, 6), (13, 7), (13, 8), (13, 10), (13, 13), (13, 15), (14, 6), 
                              (14, 8), (14, 9), (14, 11), (14, 16), (14, 19), (15, 0), (15, 1), 
                              (15, 2), (15, 3), (15, 4), (15, 8), (15, 12), (15, 17), (15, 18), 
                              (15, 19), (16, 0), (16, 6), (16, 7), (17, 3), (17, 5), (17, 6), 
                              (17, 9), (17, 13), (17, 14), (17, 18), (18, 1), (18, 3), (18, 4), 
                              (18, 5), (18, 6), (18, 9), (18, 11), (18, 12), (18, 15), (19, 1), 
                              (19, 3), (19, 4), (19, 6), (19, 7), (19, 8), (19, 12), (19, 15), (19, 16)
                              ]
    reward_target: float = 10
    reward_forbidden: float = -5
    reward_step: float = -1
    action_space: list = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]
    debug: bool = False
    animation_interval: float = 0.2


args = EnvArgs_5x5()


def validate_environment_parameters(env_size, start_state, target_state, forbidden_states):
    if not (isinstance(env_size, tuple) or isinstance(env_size, list) or isinstance(env_size, np.ndarray)) and len(env_size) != 2:
        raise ValueError(
            "Invalid environment size. Expected a tuple (rows, cols) with positive dimensions.")

    for i in range(2):
        assert start_state[i] < env_size[i]
        assert target_state[i] < env_size[i]
        for j in range(len(forbidden_states)):
            assert forbidden_states[j][i] < env_size[i]


try:
    validate_environment_parameters(
        args.env_size, args.start_state, args.target_state, args.forbidden_states)
except ValueError as e:
    print("Error:", e)
