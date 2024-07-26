from typing import Union
import numpy as np


class Args:
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


args = Args()     
def validate_environment_parameters(env_size, start_state, target_state, forbidden_states):
    if not (isinstance(env_size, tuple) or isinstance(env_size, list) or isinstance(env_size, np.ndarray)) and len(env_size) != 2:
        raise ValueError("Invalid environment size. Expected a tuple (rows, cols) with positive dimensions.")
    
    for i in range(2):
        assert start_state[i] < env_size[i]
        assert target_state[i] < env_size[i]
        for j in range(len(forbidden_states)):
            assert forbidden_states[j][i] < env_size[i]
try:
    validate_environment_parameters(args.env_size, args.start_state, args.target_state, args.forbidden_states)
except ValueError as e:
    print("Error:", e)
