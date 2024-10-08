from importmonkey import add_path
import os
add_path(os.getcwd())

from src import GridWorld
from analysis import Recorder
import numpy as np
import random
from tqdm import trange


def train(env: GridWorld):
    discount_rate = 0.9
    max_iteration = 500
    q_k = {s: dict.fromkeys(env.action_space, 0) for s in env.state_space}  # q_k(s, a) <- q_k[s][a]
    policy = {s: dict.fromkeys(env.action_space, 1/len(env.action_space)) for s in env.state_space}  # pi(a|s) <-  policy[s][a]
    v_pi_k = {s: 0 for s in env.state_space}  # v_k(s) <- v_k[s]
    
    episode_length = 300
    alpha = {s: dict.fromkeys(env.action_space, 0.1) for s in env.state_space}

    epsilon = 0.1

    length_recorder = Recorder("./result/sarsa/length.txt")
    reward_recorder = Recorder("./result/sarsa/reward.txt")

    # train
    for k in trange(max_iteration):
        state, _ = env.reset()
        action = decision(state, policy)
        done = False
        total_reward = 0
        for t in range(episode_length):
            # generate sarsa (s_t, a_t, r, s_{t+1}, a_{t+1}) <- (state, action, reward, next_state, next_action)
            if not done:
                next_state, reward, done, _ = env.step(action)
                next_action = decision(next_state, policy)
                # policy evaluation
                q_k[state][action] = q_k[state][action] - alpha[state][action] * (q_k[state][action] - (reward + discount_rate*q_k[next_state][next_action]))
                # policy improvement
                max_value_action = max(q_k[state], key=lambda _a: q_k[state][_a])
                policy[state][max_value_action] = 1 - epsilon * (mathcal_A(state, env) - 1) / mathcal_A(state, env)
                for a in policy[state].keys():
                    if a != max_value_action:
                        policy[state][a] = epsilon / mathcal_A(state, env)
                # update s_t, a_t
                state, action = next_state, next_action
                # record
                total_reward += reward
            else:
                length_recorder.add(t)
                reward_recorder.add(total_reward)
                break
    return policy


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

def show(policy, delay=20):
    env.reset()
    # Add policy
    # policy_matrix=np.random.rand(env.num_states, len(env.action_space))    
    policy_matrix=np.zeros((env.num_states, len(env.action_space)))    
    for s, actions in policy.items():
        for i, a in enumerate(actions.keys()):
            policy_matrix[s[1]*env.env_size[0]+s[0]][i] = policy[s][a]
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1

    env.add_policy(policy_matrix)

    # Render the environment
    env.render(animation_interval=delay)
    
if __name__ == "__main__":             
    env = GridWorld()
    env.reset()
    env.render()
    policy = train(env)
    test(policy)
    show(policy)
    print("done")