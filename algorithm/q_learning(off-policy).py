from importmonkey import add_path
import os
add_path(os.getcwd())

from src import GridWorld
from analysis import Recorder
import numpy as np
import random
from tqdm import tqdm


def train(env: GridWorld):
    discount_rate = 0.9
    v_pi_k = {s: 0 for s in env.state_space}  # v_k(s) <- v_k[s]
    q_k = {s: dict.fromkeys(env.action_space, 0) for s in env.state_space}  # q_k(s, a) <- q_k[s][a]
    behavior_policy = {s: dict.fromkeys(env.action_space, 1/len(env.action_space)) for s in env.state_space}  # pi(a|s) <-  policy[s][a]
    target_policy = behavior_policy.copy()
    
    episode_num = 5
    episode_length = 100000
    alpha = {s: dict.fromkeys(env.action_space, 0.1) for s in env.state_space}

    epsilon = 0.1


    # train
    print("generating episodes ......")
    episodes = generate_episodes(behavior_policy, env, episode_length, episode_num)
    print("generate episodes done")
    for episode in tqdm(episodes):
        for step_index in range(1, len(episode)):
            state, action, reward = episode[step_index-1]
            next_state, _, _ = episode[step_index]
            # update q-value for s_t, a_t
            q_k[state][action] = q_k[state][action]  -  alpha[state][action] * (q_k[state][action]-(reward+discount_rate*max(q_k[next_state].values())))
            v_pi_k[state] = max(q_k[state].values())
            # update policy for s_t
            max_value_action = max(q_k[state], key=lambda _a: q_k[state][_a])
            target_policy[state][max_value_action] = 1
            for a in target_policy[state].keys():
                if a != max_value_action:
                    target_policy[state][a] = 0
    return target_policy, v_pi_k

def generate_episodes(policy, env: GridWorld, length=100000, num=5):
    episodes = []
    for _ in range(num):
        episode = []
        state, _ = env.reset()
        for _ in range(length):
            action = decision(state, policy, train=True)
            step = [state, action, 0]
            state, reward, done, _ = env.step(action)
            step[2] = reward
            episode.append(step)
        episodes.append(episode)
    return episodes

def test(policy):
    # use trained policy
    next_state, _ = env.reset()
    for t in range(1000):
        action = decision(next_state, policy, train=False)
        next_state, reward, done, _ = env.step(action)
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        if done:
            env.render(animation_interval=20)
            break
        else:
            env.render()

def decision(state, policy, train=True):
    if train is True:
        return random.choices(list(policy[state].keys()), weights=list(policy[state].values()), k=1)[0]
    else:
        return max(policy[state], key=policy[state].get)

def mathcal_A(state, env):
    return len(env.action_space)

def show(policy, state_value, delay=20):
    env.reset()
    # Add policy
    # policy_matrix=np.random.rand(env.num_states, len(env.action_space))    
    policy_matrix=np.zeros((env.num_states, len(env.action_space)))    
    for s, actions in policy.items():
        for i, a in enumerate(actions.keys()):
            policy_matrix[s[1]*env.env_size[0]+s[0]][i] = policy[s][a]
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1

    env.add_policy(policy_matrix)

     # Add state values
    # values = np.random.uniform(0,10,(env.num_states,))
    values = np.array([0 for _ in range(len(env.state_space))], dtype=np.float32)
    for s in env.state_space:
        values[s[1]*env.env_size[0]+s[0]] = state_value[s]
    env.add_state_values(values)

    # Render the environment
    env.render(animation_interval=delay)
    
if __name__ == "__main__":             
    env = GridWorld()
    env.reset()
    env.render()
    policy, state_value = train(env)
    test(policy)
    show(policy, state_value)
