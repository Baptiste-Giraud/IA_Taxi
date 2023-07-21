import os
import sys
import gym
import time
import argparse
import matplotlib
import numpy as np
import pandas as pd
from DQN_Play import play
from DQN import DQN, DQN_2
from itertools import count
import matplotlib.pyplot as plt
from Memory import Transition, ReplayMemory

import torch
import torch.optim as optim
import torch.nn.functional as F

is_notebook = 'inline' in matplotlib.get_backend()

if is_notebook:
    from IPython import display


class TrainingAgent():

    def __init__(self,
                 env=gym.make("Taxi-v3").env,
                 batch_size: int = 128,
                 gamma: float = 0.99,
                 eps_start: float = 1,
                 eps_end: float = 0.1,
                 eps_decay: float = 400,
                 target_update: int = 20,
                 max_steps_per_episode: int = 100,
                 warmup_episode: int = 10,
                 save_freq: int = 1000,
                 lr: float = 0.001,
                 lr_min: float = 0.0001,
                 lr_decay: int = 5000,
                 memory_size: int = 50000,
                 num_episodes: int = 10000,
                 name: str = None,
                 architecture: int = 2,
                 save: bool = True) -> None:
        self.config = {
            "BATCH_SIZE": batch_size,
            "GAMMA": gamma,
            "EPS_START": eps_start,
            "EPS_END": eps_end,
            "EPS_DECAY": eps_decay,
            "TARGET_UPDATE": target_update,
            "MAX_STEPS_PER_EPISODE": max_steps_per_episode,
            "WARMUP_EPISODE": warmup_episode,
            "SAVE_FREQ": save_freq,
            "LR": lr,
            "LR_MIN": lr_min,
            "LR_DECAY": lr_decay,
            "MEMORY_SIZE": memory_size,
            "NUM_EPISODES": num_episodes,
            "SHOULD_WARMUP": True
        }
        self.save_fig = save
        self.episode_durations = []
        self.reward_in_episode = []
        self.epsilon_vec = []

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.id = int(time.time()) if name == None else name
        self.rng = np.random.default_rng(123)
        self.architecture = architecture

    def print_model_info(self) -> None:
        '''Display agent's configuration'''
        print("Model Initialized with params:")
        print(" - EPISODES: {},".format(self.config["NUM_EPISODES"]))
        print(" - BATCH SIZE: {},".format(self.config["BATCH_SIZE"]))
        print(" - GAMMA: {},".format(self.config["GAMMA"]))
        print(" - STARTING EPSILON: {},".format(self.config["EPS_START"]))
        print(" - ENDING EPSILON: {},".format(self.config["EPS_END"]))
        print(" - DECAY EPSILON: {},".format(self.config["EPS_DECAY"]))
        print(" - TARGET UPDATE: {},".format(self.config["TARGET_UPDATE"]))
        print(" - MAX STEPS PER EPISODE: {},".format(
            self.config["MAX_STEPS_PER_EPISODE"]))
        print(" - WARMUP: {},".format(self.config["WARMUP_EPISODE"]))
        print(" - SAVE FREQUENCIE: {},".format(self.config["SAVE_FREQ"]))
        print(" - STARTING LEARNING RATE: {},".format(self.config["LR"]))
        print(" - ENDING LEARNING RATE {},".format(self.config["LR_MIN"]))
        print(" - DECAY LEARNING RATE: {},".format(self.config["LR_DECAY"]))
        print(" - MEMORY: {}.".format(self.config["MEMORY_SIZE"]))
        print()

    def import_model(self, id: str) -> None:
        '''Import an existing agent to finish training based on previously set parameters'''
        try:
            n_actions = self.env.action_space.n
            n_observations = self.env.observation_space.n
            self.id = id
            checkpoint = torch.load(f"./models/{id}/DQN_{id}.pt")

            self.architecture = checkpoint.get("architecture") or 1

            self.model = DQN(n_observations, n_actions).to(
                self.device) if self.architecture == 1 else DQN_2(
                    n_observations, n_actions).to(self.device)
            self.target_model = DQN(n_observations, n_actions).to(
                self.device) if self.architecture == 1 else DQN_2(
                    n_observations, n_actions).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.config["LR"])

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

            self.reward_in_episode = checkpoint["reward_in_episode"]
            self.episode_durations = checkpoint["episode_durations"]
            self.epsilon_vec = checkpoint["epsilon_vec"]

            self.config = checkpoint["config"]

            if len(self.episode_durations) - 1 > self.config["WARMUP_EPISODE"]:
                self.config["SHOULD_WARMUP"] = False
        except:
            print("[WARNING] - Unable to Import Model {}".format(id))
            if input("Do you wish to create a new model ? [y/n] ").lower(
            ) == "y":
                print("Creating model {}...".format(id))
                self.compile()
            else:
                print("Stopping Execution.")
                sys.exit(0)

    def compile(self) -> None:
        '''Compile Agent based on selected architecture'''
        n_actions = self.env.action_space.n
        n_observations = self.env.observation_space.n

        self.model = DQN(n_observations, n_actions).to(
            self.device) if self.architecture == 1 else DQN_2(
                n_observations, n_actions).to(self.device)
        self.target_model = DQN(n_observations, n_actions).to(
            self.device) if self.architecture == 1 else DQN_2(
                n_observations, n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.config["LR"])

    def _get_epsilon(self, episode: int) -> float:
        '''Calculate Epsilon value depending on the number of episode'''
        epsilon = self.config["EPS_END"] + \
                          (self.config["EPS_START"] - self.config["EPS_END"]) * \
                              np.exp(-episode / self.config["EPS_DECAY"])

        return epsilon

    def _get_action_for_state(self, state: int):
        '''Returns an action choosen by the agent based on a state'''
        with torch.no_grad():
            predicted = self.model(torch.tensor([state], device=self.device))
            action = predicted.max(1)[1]

        return action.item()

    def _choose_action(self, state: int, epsilon: int) -> int:
        '''Defines whether to choose an action or explore'''
        if self.rng.uniform() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self._get_action_for_state(state)

        return action

    def _remember(self, state: int, action: int, next_state: int, reward: int,
                  done: bool) -> None:
        '''Store state, action, next_state, reward, and status in memory'''
        self.memory.push(
            torch.tensor([state], device=self.device),
            torch.tensor([action], device=self.device, dtype=torch.long),
            torch.tensor([next_state], device=self.device),
            torch.tensor([reward], device=self.device),
            torch.tensor([done], device=self.device, dtype=torch.bool))

    def _train_model(self) -> None:
        '''Update Target Model's weight based on training model'''
        if len(self.memory) < self.config["BATCH_SIZE"]:

            return
        transitions = self.memory.sample(self.config["BATCH_SIZE"])
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        # Compute predicted Q values
        predicted_q_value = self.model(state_batch).gather(
            1, action_batch.unsqueeze(1))

        # Compute the expected Q values
        next_state_values = self.target_model(next_state_batch).max(1)[0]
        expected_q_values = (~done_batch * next_state_values *
                             self.config["GAMMA"]) + reward_batch

        loss = self.loss(predicted_q_value, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def _adjust_learning_rate(self, episode: int) -> None:
        '''Update Learning Rate based on number of episode'''
        delta = self.config["LR"] - self.config["LR_MIN"]
        base = self.config["LR_MIN"]
        rate = self.config["LR_DECAY"]
        lr = base + delta * np.exp(-episode / rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _update_target(self) -> None:
        '''Save Target Model's dictionary'''
        self.target_model.load_state_dict(self.model.state_dict())

    def fit(self) -> None:
        '''Fit Agent during a number of episode'''
        self.memory = ReplayMemory(self.config["MEMORY_SIZE"])
        self.loss = F.smooth_l1_loss

        episode_done = len(self.episode_durations)
        reward_in_episode = 0
        epsilon = 1 if self.config["SHOULD_WARMUP"] else self.epsilon_vec[-1]
        start = time.time()

        for i_episode in range(self.config["NUM_EPISODES"] - episode_done):
            state = self.env.reset()
            if i_episode >= self.config["WARMUP_EPISODE"] and self.config[
                    "SHOULD_WARMUP"]:
                epsilon = self._get_epsilon(i_episode -
                                            self.config["WARMUP_EPISODE"])
            elif not self.config["SHOULD_WARMUP"]:
                epsilon = self._get_epsilon(
                    len(self.episode_durations) -
                    self.config["WARMUP_EPISODE"])

            for step in count():
                action = self._choose_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)

                self._remember(state, action, next_state, reward, done)

                if i_episode >= self.config[
                        "WARMUP_EPISODE"] or not self.config["SHOULD_WARMUP"]:
                    self._train_model()
                    self._adjust_learning_rate(i_episode -
                                               self.config["WARMUP_EPISODE"] +
                                               1)
                    done = (step == self.config["MAX_STEPS_PER_EPISODE"] -
                            1) or done
                else:
                    done = (step == 5 * self.config["MAX_STEPS_PER_EPISODE"] -
                            1) or done

                state = next_state
                reward_in_episode += reward

                if done:
                    self.episode_durations.append(step + 1)
                    self.reward_in_episode.append(reward_in_episode)
                    self.epsilon_vec.append(epsilon)
                    reward_in_episode = 0
                    self.plot_durations()

                    break

            if i_episode % self.config["TARGET_UPDATE"] == 0:
                self._update_target()

            if i_episode % (self.config["SAVE_FREQ"] / 2) == 0:
                print(
                    "[EPISODE {}/{}] - {}min - Mean reward for last {} Episodes: {} in {} steps - Mean Time Per Episode: {}"
                    .format(
                        i_episode + episode_done, self.config["NUM_EPISODES"],
                        int((time.time() - start) / 60),
                        int(self.config["SAVE_FREQ"] / 2),
                        np.round(
                            np.mean(self.reward_in_episode[
                                -int(self.config["SAVE_FREQ"] / 2):]), 2),
                        np.round(
                            np.mean(self.episode_durations[
                                -int(self.config["SAVE_FREQ"] / 2):]), 2),
                        np.round((time.time() - start) /
                                 i_episode, 6) if i_episode != 0 else 0))
                start = time.time()

            if i_episode % self.config["SAVE_FREQ"] == 0:
                self.save()

            self.last_episode = i_episode

        self.save()

    @staticmethod
    def _moving_average(x: list, periods=5) -> list:
        if len(x) < periods:

            return x

        cumsum = np.cumsum(np.insert(x, 0, 0))
        res = (cumsum[periods:] - cumsum[:-periods]) / periods

        return np.hstack([x[:periods - 1], res])

    def save(self) -> None:
        '''Save Model and graphs'''
        if self.save_fig:
            if not os.path.isdir(f"./models/{self.id}"):
                os.makedirs(f"./models/{self.id}")

            torch.save(
                {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    "reward_in_episode": self.reward_in_episode,
                    "episode_durations": self.episode_durations,
                    "epsilon_vec": self.epsilon_vec,
                    "config": self.config,
                    "architecture": self.architecture
                }, f"./models/{self.id}/DQN_{self.id}.pt")
            plt.savefig(f"./models/{self.id}/DQN_{self.id}_graph.png")

    def plot_durations(self) -> None:
        '''Plot graphs containing Epsilon, Rewards, and Steps per episode over time'''
        lines = []
        fig = plt.figure(1, figsize=(15, 7))
        plt.clf()
        ax1 = fig.add_subplot(111)

        plt.title(f'Training {self.id}...')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Duration & Rewards')
        ax1.set_ylim(-2 * self.config["MAX_STEPS_PER_EPISODE"],
                     self.config["MAX_STEPS_PER_EPISODE"] + 10)
        ax1.plot(self.episode_durations, color="C1", alpha=0.2)
        ax1.plot(self.reward_in_episode, color="C2", alpha=0.2)
        mean_steps = self._moving_average(self.episode_durations, periods=5)
        mean_reward = self._moving_average(self.reward_in_episode, periods=5)
        lines.append(ax1.plot(mean_steps, label="steps", color="C1")[0])
        lines.append(ax1.plot(mean_reward, label="rewards", color="C2")[0])

        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon')
        lines.append(
            ax2.plot(self.epsilon_vec, label="epsilon", color="C3")[0])
        labs = [l.get_label() for l in lines]
        ax1.legend(lines, labs, loc=3)

        if is_notebook:
            display.clear_output(wait=True)
        else:
            plt.show()
        plt.pause(0.001)


def train_agent(env=gym.make("Taxi-v3").env,
                batch_size: int = 128,
                gamma: float = 0.99,
                eps_start: float = 1,
                eps_end: float = 0.1,
                eps_decay: float = 400,
                target_update: int = 20,
                max_steps_per_episode: int = 100,
                warmup_episode: int = 10,
                save_freq: int = 1000,
                lr: float = 0.001,
                lr_min: float = 0.0001,
                lr_decay: int = 5000,
                memory_size: int = 50000,
                num_episodes: int = 10000,
                name: str = None,
                architecture: int = 2,
                save: bool = True) -> None:
    agent = TrainingAgent(env=env,
                          batch_size=batch_size,
                          gamma=gamma,
                          eps_start=eps_start,
                          eps_end=eps_end,
                          eps_decay=eps_decay,
                          target_update=target_update,
                          max_steps_per_episode=max_steps_per_episode,
                          warmup_episode=warmup_episode,
                          save_freq=save_freq,
                          lr=lr,
                          lr_min=lr_min,
                          lr_decay=lr_decay,
                          memory_size=memory_size,
                          num_episodes=num_episodes,
                          name=name,
                          architecture=architecture,
                          save=save)

    agent.fit()
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Taxi Driver Model based on DQN")
    parser.add_argument("--environment",
                        type=str,
                        default="Taxi-v3",
                        help="Environment to train the DQN")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Batch Size")
    parser.add_argument("--gamma", type=float, default=0.99, help="GAMMA")
    parser.add_argument("--eps_start",
                        type=float,
                        default=1.0,
                        help="Epsilon Starting value")
    parser.add_argument("--eps_end",
                        type=float,
                        default=0.1,
                        help="Epsilon minimal value")
    parser.add_argument("--eps_decay",
                        type=float,
                        default=400,
                        help="Epsilon Decay rate")
    parser.add_argument("--target_update",
                        type=int,
                        default=20,
                        help="Number of episodes between dict saving")
    parser.add_argument("--max_steps",
                        type=int,
                        default=100,
                        help="Max Steps Per Episode")
    parser.add_argument("--warmup_episode",
                        type=int,
                        default=10,
                        help="Number of warmup episodes")
    parser.add_argument("--save_freq",
                        type=int,
                        default=1000,
                        help="Number of episode between model saving")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning Rate")
    parser.add_argument("--lr_min",
                        type=float,
                        default=0.0001,
                        help="Learning Rate Minimal Value")
    parser.add_argument("--lr_decay",
                        type=float,
                        default=5000,
                        help="Learning Rate Decay rate")
    parser.add_argument("--memory",
                        type=int,
                        default=50000,
                        help="Size of Memory")
    parser.add_argument("--episodes",
                        type=int,
                        default=10000,
                        help="Number of episodes during training")
    parser.add_argument("--model",
                        type=str,
                        default="",
                        help="Import existing model")
    parser.add_argument("--name",
                        type=str,
                        default=None,
                        help="Name for the model")
    parser.add_argument("--architecture",
                        type=int,
                        default=2,
                        help="Model Architecture to use (1/2)")

    args = parser.parse_args()
    env = gym.make(args.environment).env
    batch_size = args.batch_size
    gamma = args.gamma
    eps_start = args.eps_start
    eps_end = args.eps_end
    eps_decay = args.eps_decay
    target_update = args.target_update
    max_steps_per_episode = args.max_steps
    warmup_episode = args.warmup_episode
    save_freq = args.save_freq
    lr = args.lr
    lr_min = args.lr_min
    lr_decay = args.lr_decay
    memory_size = args.memory
    num_episodes = args.episodes
    existing_model = args.model
    name = args.name
    architecture = args.architecture

    start_time = time.time()

    plt.ion()
    # agent = TrainingAgent(env=env,
    #                       batch_size=batch_size,
    #                       gamma=gamma,
    #                       eps_start=eps_start,
    #                       eps_end=eps_end,
    #                       eps_decay=eps_decay,
    #                       target_update=target_update,
    #                       max_steps_per_episode=max_steps_per_episode,
    #                       warmup_episode=warmup_episode,
    #                       save_freq=save_freq,
    #                       lr=lr,
    #                       lr_min=lr_min,
    #                       lr_decay=lr_decay,
    #                       memory_size=memory_size,
    #                       num_episodes=num_episodes,
    #                       name=name,
    #                       architecture=architecture)

    agent = train_agent(env=env,
                          batch_size=batch_size,
                          gamma=gamma,
                          eps_start=eps_start,
                          eps_end=eps_end,
                          eps_decay=eps_decay,
                          target_update=target_update,
                          max_steps_per_episode=max_steps_per_episode,
                          warmup_episode=warmup_episode,
                          save_freq=save_freq,
                          lr=lr,
                          lr_min=lr_min,
                          lr_decay=lr_decay,
                          memory_size=memory_size,
                          num_episodes=num_episodes,
                          name=name,
                          architecture=architecture)

    if len(existing_model) != 0:
        agent.import_model(existing_model)
    else:
        agent.compile()

    agent.print_model_info()

    agent.fit()

    time_train = time.time() - start_time

    print('Training Complete in {}s - {}min - {}h'.format(
        time_train, np.round(time_train / 60, 2),
        np.round(time_train / 3600, 2)))
    print()
    print("Testing...")

    mean_steps, mean_result, total_failed = 0, 0, 0
    for l in range(1000):
        steps, result, done = play(agent.model, is_loop=True)
        mean_steps += steps
        mean_result += result
        if not done:
            total_failed += 1
    percentage_success = np.round((1 - total_failed / 1000) * 100, 2)

    print("Testing Complete. {}% Win Rate".format(percentage_success))
    print()
    print("Saving Metrics to models.csv...")
    df = pd.read_csv("models.csv", sep=";")

    if not f"{agent.id}/DQN_{agent.id}.pt" in df["name"].unique():
        new_row = [[
            f"{agent.id}/DQN_{agent.id}.pt", agent.config["BATCH_SIZE"],
            agent.config["GAMMA"], agent.config["EPS_START"],
            agent.config["EPS_END"], agent.config["EPS_DECAY"],
            agent.config["TARGET_UPDATE"],
            agent.config["MAX_STEPS_PER_EPISODE"],
            agent.config["WARMUP_EPISODE"], agent.config["SAVE_FREQ"],
            agent.config["LR"], agent.config["LR_MIN"],
            agent.config["LR_DECAY"], agent.config["MEMORY_SIZE"],
            agent.config["NUM_EPISODES"],
            np.round(np.mean(agent.reward_in_episode[-100:]),
                     2), percentage_success
        ]]
        df2 = pd.DataFrame(new_row, columns=df.columns.values)
        new_df = pd.concat([df, df2])
        new_df.set_index('name', drop=True, inplace=True)

        new_df.to_csv("models.csv", sep=";")
        print("Metrics Saved.")
    else:
        print("Model already saved not saving metrics")

    plt.ioff()
