import gym
import sys
import time
import torch
import random
import argparse
import datetime
import numpy as np
try:
    from DQN import DQN, DQN_2
except:
    pass
from IPython import display
import torch.optim as optim

def import_model(path: str) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path)

    n_actions = env.action_space.n
    n_observation = env.observation_space.n
    model = DQN(n_observation, n_actions).to(
        device) if checkpoint.get("architecture") == 1 or checkpoint.get(
            "architecture") == None else DQN_2(n_observation,
                                               n_actions).to(device)
    optimizer = optim.RMSprop(model.parameters())

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return optimizer, model, device


def get_action_for_state(state: int, model, device) -> list:
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        predicted = model(torch.tensor([state], device=device))
        action = predicted.max(1)[1]
    return action.item()


def solve(model,
          device=torch.device("cpu"),
          env=gym.make("Taxi-v3").env,
          render: bool = False,
          max_steps: int = 100,
          slow: bool = False,
          is_loop: bool = False,
          is_time: bool = False) -> tuple[int, int, bool]:
    # Play an episode
    actions_str = ["South", "North", "East", "West", "Pickup", "Dropoff"]

    iteration = 0
    state = env.reset()  # reset environment to a new, random state
    if render:
        env.render()
        print(f"Iter: {iteration} - Action: *** - Reward ***")
    done = False
    total_reward = 0

    while not done:
        action = get_action_for_state(state, model, device)
        iteration += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        display.clear_output(wait=True)
        if render or (random.uniform(0, 1) < 0.3 and not is_loop
                      and not is_time):
            env.render()
            print(
                f"Iter: {iteration} - Action: {action}({actions_str[action]}) - Reward {reward}"
            )
        if iteration == max_steps:
            break
        elif slow and not done:
            input("Press anything to continue...")
            print("\r", end="\r")
    if (not is_loop and not is_time):
        print("[{}/{} MOVES] - Total reward: {}".format(
            iteration, max_steps, total_reward))

    return iteration, total_reward, done


def play(env, model, max, mean_steps, mean_result, total_failed, render, slow,
         is_time, is_loop, device):
    steps, result, done = solve(model,
                                render=render,
                                slow=slow,
                                env=env,
                                max_steps=max,
                                device=device,
                                is_loop=is_loop,
                                is_time=is_time)
    mean_steps += steps
    mean_result += result
    if not done:
        total_failed += 1

    return steps, result, mean_steps, mean_result


def display_data(total, total_failed, start, mean_steps, mean_result):
    print()
    print(
        "[{} LOOP DONE - {}% FAILED - {} SECONDES] - Mean Steps Per Loop: {} - Mean Reward Per Loop: {} - Mean Time Per Loop: {}"
        .format(total, np.round(total_failed / total * 100, 2),
                np.round(time.time() - start, 4),
                np.round(mean_steps / total, 2),
                np.round(mean_result / total, 2),
                np.round((time.time() - start) / total), 6))


def error_args(args):
    path = args.path
    max_steps = args.max
    time = args.time
    loop = args.loop

    try:
        open(path, "r")
    except OSError:
        return 1, "Model path is invalid."

    if max_steps <= 0:
        return 1, "Max number of steps can not be lower than 1."
    if time < 0:
        return 1, "Time can not be negative or null."
    if loop <= 0:
        return 1, "Number of loop can not be negative or null"

    return 0, ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the Taxi Driver Game Using a DQN model")
    parser.add_argument("-p",
                        "--path",
                        type=str,
                        default="./models/reference_2/DQN_reference_2.pt",
                        help="DQN model to use")
    parser.add_argument(
        "-m",
        "--max",
        type=int,
        default=100,
        help=
        "Max Number of Steps the Model is allowed to take to complete the game"
    )
    parser.add_argument(
        "-s",
        "--slow",
        dest="slow",
        action="store_true",
        default=False,
        help="Activate Slow Mode",
    )
    parser.add_argument("-r",
                        "--render",
                        dest="render",
                        action="store_true",
                        default=False,
                        help="Render State for each Step")
    parser.add_argument("-l",
                        "--loop",
                        type=int,
                        help="How many times to play the game",
                        default=1)
    parser.add_argument("-t",
                        "--time",
                        type=int,
                        default=0,
                        help="Run play for x seconds")
    args = parser.parse_args()
    code, msg = error_args(args)

    if code != 0:
        print("[ERROR] - {}".format(msg))

        sys.exit(1)

    path = args.path
    render = args.render
    slow = args.slow
    max = args.max
    loop = args.loop
    max_time = args.time

    if slow:
        render = True

    start = time.time()
    env = gym.make("Taxi-v3").env
    _, model, device = import_model(path)

    mean_steps, mean_result = 0, 0
    total_failed = 0
    is_loop = True if args.loop != 1 else False
    maxrt = datetime.timedelta(seconds=max_time) if max_time != 0 else None

    if maxrt != None:
        stop = datetime.datetime.now() + maxrt
        total = 0
        while datetime.datetime.now() < stop:
            steps, result, mean_steps, mean_result = play(env,
                                                          model,
                                                          max,
                                                          mean_steps,
                                                          mean_result,
                                                          total_failed,
                                                          render=render,
                                                          slow=slow,
                                                          is_time=True,
                                                          is_loop=is_loop,
                                                          device=device)
            total += 1
        display_data(total, total_failed, start, mean_steps, mean_result)
    else:
        for l in range(loop):
            steps, result, mean_steps, mean_result = play(env,
                                                          model,
                                                          max,
                                                          mean_steps,
                                                          mean_result,
                                                          total_failed,
                                                          render=render,
                                                          slow=slow,
                                                          is_time=False,
                                                          is_loop=is_loop,
                                                          device=device)
        if is_loop:
            display_data(loop, total_failed, start, mean_steps, mean_result)
