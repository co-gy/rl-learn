from importmonkey import add_path
import os
add_path(os.getcwd())

from src import GridWorld
from analysis import RecordStateValue
import numpy as np
import random
from tqdm import trange


def train(env: GridWorld):
    discount_rate = 0.95
    max_iteration = 3000
    q_k = {s: dict.fromkeys(env.action_space, 0) for s in env.state_space}  # q_k(s, a) <- q_k[s][a]
    v_pi_k = {s: 0 for s in env.state_space}  # v_k(s) <- v_k[s]
    policy = {s: dict.fromkeys(env.action_space, 1/len(env.action_space)) for s in env.state_space}  # pi(a|s) <-  policy[s][a]

    num = {s: dict.fromkeys(env.action_space, 0) for s in env.state_space}  # Num(s, a) <- num[s][a]
    return_ = {s: dict.fromkeys(env.action_space, 0) for s in env.state_space}  # Return(s, a) <- return[s][a]
    episode_length = 1000
    epsilon = 0.6

    # train
    last_v_pi_k = v_pi_k.copy()
    for k in (pbar := trange(max_iteration)):
        print("\r", k, end="", flush=True)
        episode = generate_episode(policy, env, episode_length)
        g = 0  # sample of G_t
        epsilon = max(0.1, epsilon-0.003)
        last_v_pi_k = v_pi_k.copy()
        for state, action, reward in reversed(episode):
            g = reward + discount_rate * g
            return_[state][action] += g
            num[state][action] += 1
            # policy evaluation
            q_k[state][action] = return_[state][action] / num[state][action]
            # policy improvement
            max_value_action = max(q_k[state], key=lambda _a: q_k[state][_a])
            policy[state][max_value_action] = 1 - epsilon * (mathcal_A(state, env) - 1) / mathcal_A(state, env)
            for a in policy[state].keys():
                if a != max_value_action:
                    policy[state][a] = epsilon / mathcal_A(state, env)
            v_pi_k[state] = max(q_k[state].values())
        pbar.set_description(f"{k}: {np.linalg.norm(np.array(list(last_v_pi_k.values())) - np.array(list(v_pi_k.values())))}")
    return policy, v_pi_k

def generate_episode(policy, env: GridWorld, length=10000):
    episode = []
    state, _ = env.reset()
    for _ in range(length):
        action = decision(state, policy, train=True)
        step = [state, action, 0]
        state, reward, _, _ = env.step(action)
        step[2] = reward
        episode.append(step)
        # env.render(animation_interval=0.01)
    return episode

def mathcal_A(state, env):
    return len(env.action_space)

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
    # print(values)
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
