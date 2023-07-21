import gym
import numpy as np
import random
import time
import argparse
import pickle
import datetime
import matplotlib.pyplot as plt


def create_random_policy(env) -> dict:
    '''Create an empty dictionary to store state action values'''
    policy = {}
    for key in range(0, env.observation_space.n):
        p = {}
        for action in range(0, env.action_space.n):
            p[action] = 1 / env.action_space.n
        policy[key] = p
    return policy


def create_state_action_dictionary(env, policy: dict) -> dict:
    '''Create an empty dictionary for storing rewards for each state-action pair'''
    Q = {}
    for key in policy.keys():
        Q[key] = {a: 0.0 for a in range(0, env.action_space.n)}
    return Q


def run_game(env, policy: dict) -> list[list[int]]:
    '''Plays the game and returns a list of [state, action, reward]'''
    env.reset()
    episode = []
    finished = False

    while not finished:
        s = env.env.s

        timestep = []
        timestep.append(s)
        n = random.uniform(0, sum(policy[s].values()))
        top_range = 0
        for prob in policy[s].items():
            top_range += prob[1]
            if n < top_range:
                action = prob[0]
                break
        _, reward, finished, _ = env.step(action)
        timestep.append(action)
        timestep.append(reward)

        episode.append(timestep)

    return episode


def train(env=gym.make("Taxi-v3"),
                       episodes: int = 100,
                       epsilon: float = 0.01) -> dict:
    policy = create_random_policy(env)
    Q = create_state_action_dictionary(env, policy)
    returns = {}
    start_episode = time.time()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"models/policy_{timestamp}.pkl"

    # Variables for plotting
    total_rewards = []
    mean_rewards = []
    episode_durations = []
    epsilon_vec = []

    for e in range(episodes):
        G = 0  # Store cumulative reward in G (initialized at 0)
        episode = run_game(env=env, policy=policy)

        for i in reversed(range(0, len(episode))):
            s_t, a_t, r_t = episode[i]
            state_action = (s_t, a_t)
            G += r_t  # Increment total reward by reward on current timestep

            if not state_action in [(x[0], x[1]) for x in episode[0:i]]:
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]

                Q[s_t][a_t] = sum(returns[state_action]) / len(
                    returns[state_action])  # Average reward across episodes

                Q_list = list(map(
                    lambda x: x[1],
                    Q[s_t].items()))  # Finding the action with maximum value
                indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
                max_Q = random.choice(indices)

                A_star = max_Q

                for a in policy[s_t].items():
                    if a[0] == A_star:
                        policy[s_t][a[0]] = 1 - epsilon + (
                            epsilon / abs(sum(policy[s_t].values())))
                    else:
                        policy[s_t][a[0]] = (epsilon /
                                             abs(sum(policy[s_t].values())))

        # Track episode statistics for plotting
        total_rewards.append(G)
        mean_reward = np.mean(total_rewards[-100:])
        mean_rewards.append(mean_reward)
        episode_durations.append(len(episode))
        epsilon_vec.append(epsilon)

        if e % int(episodes / 10) == 0:
            episode_time = (time.time() - start_episode)
            start_episode = time.time()
            print("[EPISODE {}/{}] - {} secondes - Mean Time Per Episode: {}s".format(
                e, episodes, np.round(episode_time, 4),
                np.round(episode_time / e, 6) if e != 0 else 0))

    with open(path, 'wb') as f:
        pickle.dump(policy, f)

    # Plotting
    plot_durations(episode_durations, total_rewards, mean_rewards, epsilon_vec)

    return policy


def plot_durations(episode_durations: list, total_rewards: list, mean_rewards: list, epsilon_vec: list):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('Training Progress')
    ax1.plot(episode_durations)
    ax1.set_ylabel('Duration')
    ax1.set_xlabel('Episode')
    ax1.set_title('Episode Duration')

    ax2.plot(total_rewards)
    ax2.set_ylabel('Total Reward')
    ax2.set_xlabel('Episode')
    ax2.set_title('Total Reward')

    ax3.plot(mean_rewards)
    ax3.set_ylabel('Mean Reward')
    ax3.set_xlabel('Episode')
    ax3.set_title('Mean Reward')

    plt.tight_layout()
    plt.savefig('graphs/training_progress.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Taxi Driver Using the Q-Learning Algorithm")
    parser.add_argument(
        "--episodes",
        type=int,
        default=500000,
        help="Number of episodes",
    )
    args = parser.parse_args()

    env = gym.make("Taxi-v3")
    policy = train(env, episodes=args.episodes)
