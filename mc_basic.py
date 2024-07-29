from src import GridWorld
import numpy as np
import random
from tqdm import trange


def train(env: GridWorld):
    discount_rate = 0.9
    max_iteration = 50
    q_k = {s: dict.fromkeys(env.action_space, 0) for s in env.state_space}  # q_k(s, a) <- q_k[s][a]
    v_pi_k = {s: 0 for s in env.state_space}  # v_k(s) <- v_k[s]
    policy = {s: random.choice(env.action_space) for s in env.state_space}  # pi(s) <-  policy[s] return a
    policy = {s: (0, 1) for s in env.state_space}  # pi(s) <-  policy[s] return a
    episode_length = 30

    v_pi_k_history = ["0\n"]
    v_pi_k_file = open("./result/mc_basic/state_value_history.txt", "w")

    # train
    for k in trange(max_iteration):
        for s in env.state_space:
            for a in env.action_space:
                # collect sufficiently many episodes starting from (s, a) by \pi_k
                episodes = generate_episodes(s, a, policy, env, episode_length)
                # policy evaluation
                q_k[s][a] = sum([episode_return(episode, discount_rate) for episode in episodes]) / len(episodes)  # average return of all episodes
            # policy improvement
        for s in env.state_space:
            for a in env.action_space:
                next_state, reward = env._get_next_state_and_reward(s, a)
                q_k[s][a] = reward + discount_rate * v_pi_k[next_state]
            v_pi_k[s] = max(q_k[s].values())
            max_action_value = max(q_k[s], key=lambda _a: q_k[s][_a])
            policy[s] = max_action_value
        v_pi_k_history.append(f"{np.linalg.norm(list(v_pi_k.values()), ord=1)}\n")
    v_pi_k_file.writelines(v_pi_k_history)
    v_pi_k_file.close()
    return policy, v_pi_k

def generate_episodes(state, action, policy, env: GridWorld, length=30, num=1):
    episodes = []
    for _ in range(num):
        episode = []
        next_state, next_action = state, action
        for _ in range(length):
            visit = [next_state, next_action, 0]
            next_state, reward = env._get_next_state_and_reward(next_state, next_action)
            next_action = policy[next_state]
            visit[2] = reward
            episode.append(visit)
        episodes.append(episode)
    # print(episodes)
    return episodes

def episode_return(episode, discount_rate):
    g = 0  # sample of G_t
    for state, action, reward in reversed(episode):
        g = reward + discount_rate * g
    return g

def test(policy):
    # use trained policy
    next_state = env.start_state
    for t in range(1000):
        action = decision(next_state, policy)
        next_state, reward, done, info = env.step(action)
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        if done:
            env.render(animation_interval=20)
            break
        else:
            env.render()

def decision(state, policy):
    return policy[state]

def show(policy, state_value, delay=20):
    env.reset()
    # Add policy
    # policy_matrix=np.random.rand(env.num_states, len(env.action_space))    
    policy_matrix=np.zeros((env.num_states, len(env.action_space)))    
    for s, a in policy.items():
        policy_matrix[s[1]*env.env_size[0]+s[0]][env.action_space.index(a)] = 1
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