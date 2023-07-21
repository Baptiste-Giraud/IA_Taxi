import numpy as np
import datetime
import argparse
import random
import time
import gym
import sys


def launch(path: str = "models/qtable.npy",
         slow: bool = False,
         render: bool = False,
         is_loop: bool = False,
         is_time: bool = False):
    env = gym.make("Taxi-v3")

    q_table = np.load(path)
    done = False
    result = 0
    state = env.reset()

    if render:
        print("Initial Environnement:")
        env.render()
    steps = 1

    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)

        result += reward
        state = next_state

        if render or (random.uniform(0, 1) < 0.3 and not is_loop
                      and not is_time):
            print()
            env.render()
        steps += 1

        if steps >= 100:
            break

        if slow:
            input("Press anything to continue...")
            print("\r", end="\r")

    if not is_loop and not is_time:
        print("[{} MOVES] - Total reward: {}".format(steps, result))

    return steps, result


def display_data(total, total_failed, start, mean_steps, mean_result):
    print()
    print(
        "[{} LOOP DONE - {}% FAILED - {} SECONDES] - Mean Steps Per Loop: {} - Mean Reward Per Loop: {} - Mean Time Per Loop : {}"
        .format(total, np.round(total_failed / total * 100, 2),
                np.round(time.time() - start, 4),
                np.round(mean_steps / total, 2),
                np.round(mean_result / total, 2),
                np.round((time.time() - start) / total), 6))


def solve(path, slow, render, mean_steps, mean_result, total_failed, is_loop,
          is_time):
    steps, result = launch(path=path,
                         slow=slow,
                         render=render,
                         is_loop=is_loop,
                         is_time=is_time)
    mean_steps += steps
    mean_result += result
    if steps >= 100:
        total_failed += 1

    return mean_steps, mean_result, total_failed


def error_args(args):
    time = args.time
    loop = args.loop

    if time < 0:
        return 1, "Time can not be negative or null."
    if loop <= 0:
        return 1, "Number of loop can not be negative or null"

    return 0, ""


def play(path_file: str = "models/qtable.npy",
        slow: bool = False,
        render: bool = False,
        loop: int = 1,
        p_time: int = 0) -> tuple[int, int]:
    
    start = time.time()
    mean_steps, mean_result = 0, 0
    total_failed = 0
    is_loop = True if loop != 1 else False
    maxrt = datetime.timedelta(seconds=p_time) if p_time != 0 else None

    if maxrt != None:
        stop = datetime.datetime.now() + maxrt
        total = 0

        while datetime.datetime.now() < stop:
            mean_steps, mean_result, total_failed = solve(
                path_file, slow, render, mean_steps, mean_result,
                total_failed, is_loop, True)
            total += 1

        display_data(total, total_failed, start, mean_steps, mean_result)
    else:
        for l in range(loop):
            mean_steps, mean_result, total_failed = solve(
                path_file, slow, render, mean_steps, mean_result,
                total_failed, is_loop, False)

        if is_loop:
            display_data(loop, total_failed, start, mean_steps,
                         mean_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the Taxi Driver Game Using the SARSA Algorithm")
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
                        help="Render State")
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

    parser.add_argument("-p",
                        "--path",
                        # dest="path",
                        # action="store_true",
                        default="models/qtable.npy",
                        type=str,
                        help="Path file to load")

    args = parser.parse_args()

    code, msg = error_args(args)

    if code != 0:
        print("[ERROR] - {}".format(msg))

        sys.exit(1)

    play(args.path, args.slow, args.render, args.loop, args.time)

