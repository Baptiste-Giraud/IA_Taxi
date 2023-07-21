import numpy as np
import datetime
import argparse
import time
import gym
import sys
import matplotlib.pyplot as plt

def play(env=gym.make("Taxi-v3"), is_loop: bool = False, is_time: bool = False) -> tuple[int, int]:
    total_steps = 0
    total_reward = 0
    passenger_found = False

    state = env.reset()

    # Initialisation des listes pour les données du graphique
    steps_data = []
    rewards_data = []

    while True:
        stop = False
        while not stop:
            new_state, reward, _, _ = env.step(1)

            if state == new_state:
                stop = True

            state = new_state
            total_steps += 1
            total_reward += reward

        if not is_loop and not is_time:
            print("TOP REACHED")
            env.render()

        stop = False
        while not stop:
            new_state, reward, _, _ = env.step(3)
            total_steps += 1
            total_reward += reward

            if new_state == state:
                for s in [0, 0, 3, 3, 1, 1]:
                    new_state, reward, _, _ = env.step(s)
                    total_reward += reward
                    total_steps += 1

                stop = False
                while not stop:
                    new_state, reward, _, _ = env.step(1)

                    if state == new_state:
                        stop = True

                    state = new_state
                    total_steps += 1
                    total_reward += reward

            state = new_state

        if not is_loop and not is_time:
            print("TOP LEFT REACHED")
            env.render()

        if not passenger_found:
            if not is_loop and not is_time:
                print("Attempting Pickup")
            new_state, reward, _, _ = env.step(4)
            total_reward += reward
            total_steps += 1
            state = new_state

            if reward == -1:
                passenger_found = True

                if not is_loop and not is_time:
                    print("Passenger Found...")

        else:
            if not is_loop and not is_time:
                print("Attempting Dropoff")
            new_state, reward, done, _ = env.step(5)
            total_reward += reward
            total_steps += 1

            if done:

                if not is_loop and not is_time:
                    print("Passenger Dropped Off")

                break

        for s in [0, 0, 0, 0]:
            new_state, reward, _, _ = env.step(s)
            total_reward += reward
            total_steps += 1

        if not is_loop and not is_time:
            print("BOTTOM LEFT REACHED")
            env.render()

        if not passenger_found:
            if not is_loop and not is_time:
                print("Attempting Pickup")
            new_state, reward, _, _ = env.step(4)
            total_reward += reward
            total_steps += 1

            if reward == -1:
                passenger_found = True

                if not is_loop and not is_time:
                    print("Passenger Found...")

        else:
            if not is_loop and not is_time:
                print("Attempting Dropoff")
            new_state, reward, done, _ = env.step(5)
            total_reward += reward
            total_steps += 1

            if done:

                if not is_loop and not is_time:
                    print("Passenger Dropped Off")

                break

        for s in [1, 1, 2, 2, 2, 2, 1, 1]:
            new_state, reward, _, _ = env.step(s)
            total_reward += reward
            total_steps += 1

        if not is_loop and not is_time:
            print("TOP RIGHT REACHED")
            env.render()

        if not passenger_found:
            if not is_loop and not is_time:
                print("Attempting Pickup")
            new_state, reward, _, _ = env.step(4)
            total_reward += reward
            total_steps += 1

            if reward == -1:
                passenger_found = True

                if not is_loop and not is_time:
                    print("Passenger Found...")

        else:
            if not is_loop and not is_time:
                print("Attempting Dropoff")
            new_state, reward, done, _ = env.step(5)
            total_reward += reward
            total_steps += 1

            if done:

                if not is_loop and not is_time:
                    print("Passenger Dropped Off")

                break

        for s in [0, 0, 0, 0, 3]:
            new_state, reward, _, _ = env.step(s)
            total_reward += reward
            total_steps += 1

        if not is_loop and not is_time:
            print("BOTTOM RIGHT REACHED")
            env.render()

        if not passenger_found:
            if not is_loop and not is_time:
                print("Attempting Pickup")
            new_state, reward, _, _ = env.step(4)
            total_reward += reward
            total_steps += 1

            if reward == -1:
                passenger_found = True

                if not is_loop and not is_time:
                    print("Passenger Found...")

        else:
            if not is_loop and not is_time:
                print("Attempting Dropoff")
            new_state, reward, done, _ = env.step(5)
            total_reward += reward
            total_steps += 1

            if done:

                if not is_loop and not is_time:
                    print("Passenger Dropped Off")

                break

        if not is_loop and not is_time:
            print("[DONE] {} STEPS TOTAL / {} REWARD TOTAL".format(total_steps, total_reward))

        # Ajouter les données du pas et de la récompense aux listes pour le graphique
        steps_data.append(total_steps)
        rewards_data.append(total_reward)

        # Mettre à jour le graphique en temps réel
        update_live_plot(steps_data, rewards_data)

    return total_steps, total_reward


def display_data(total, start, mean_steps, mean_result):
    print()
    print("[{} LOOP DONE -  {} SECONDES] - Mean Steps Per Loop: {} - Max Steps For a Loop: {} - Mean Reward Per Loop: {} - Mean Time Per Loop: {}s".format(
        total, np.round(time.time() - start, 4), np.round(mean_steps / total, 2), np.max(max_steps),
        np.round(mean_result / total, 2), np.round((time.time() - start) / total), 6))


def error_args(args):
    time = args.time
    loop = args.loop

    if time < 0:
        return 1, "Time can not be negative or null."
    if loop <= 0:
        return 1, "Number of loop can not be negative or null"

    return 0, ""


def update_live_plot(steps_data, rewards_data):
    plt.figure(1)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(steps_data)
    plt.title('Total Steps')
    plt.subplot(2, 1, 2)
    plt.plot(rewards_data)
    plt.title('Total Reward')
    plt.xlabel('Episode')
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the Taxi Driver Game Using the Q-Learning Algorithm"
    )
    parser.add_argument("-t",
                        "--time",
                        type=int,
                        help="Run play for x seconds",
                        default=0)
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

    args = parser.parse_args()

    code, msg = error_args(args)
    if code != 0:
        print("[ERROR] - {}".format(msg))
        sys.exit(1)

    start = time.time()
    maxrt = datetime.timedelta(seconds=args.time) if args.time != 0 else None

    mean_steps, mean_result = 0, 0
    max_steps = []
    is_loop = True if args.loop != 1 else False

    env = gym.make("Taxi-v3")

    if maxrt is not None:
        stop = datetime.datetime.now() + maxrt
        total = 0
        while datetime.datetime.now() < stop:
            steps, result = play(env=env, is_loop=is_loop, is_time=True)
            max_steps.append(steps)
            mean_steps += steps
            mean_result += result
            total += 1
        display_data(total, start, mean_steps, mean_result)
    else:
        for l in range(args.loop):
            steps, result = play(env=env, is_loop=is_loop, is_time=False)
            max_steps.append(steps)
            mean_steps += steps
            mean_result += result
        display_data(args.loop, start, mean_steps, mean_result)

    # Sauvegarde du graphique
    plt.savefig('graphs/graph.png')
    plt.show()
