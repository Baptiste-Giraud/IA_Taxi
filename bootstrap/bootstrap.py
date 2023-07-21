import gym
import numpy as np
import time


# Définir la fonction pour initialiser la table Q
def initialize_Q(env):
    return np.zeros([env.observation_space.n, env.action_space.n])

# Définir la fonction pour choisir une action aléatoire
def random_action(env):
    return env.action_space.sample()

# Définir la fonction pour choisir une action en utilisant une politique epsilon-greedy
def epsilon_greedy(state, Q, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

# Définir la fonction pour mettre à jour la table Q en utilisant l'équation de Q-learning
def update_Q(state, action, reward, next_state, Q, learning_rate, discount_factor):
    Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])
    return Q

# Définir la fonction pour entraîner l'agent en utilisant l'algorithme Q-learning
def q_learning(env, num_episodes, learning_rate, discount_factor, epsilon, epsilon_min, epsilon_decay):
    # Initialiser la table Q
    Q = initialize_Q(env)

    # Boucle sur tous les épisodes
    for i in range(num_episodes):
        # Réinitialiser l'environnement pour un nouvel épisode
        state = env.reset()
        done = False

        # Choix d'une action en utilisant une politique epsilon-greedy
        action = epsilon_greedy(state, Q, epsilon)

        # Afficher la progression de l'entraînement dans le terminal
        if (i+1) % 1000 == 0:
            print("Episode ", i+1, " sur ", num_episodes)

        start_time = time.time()  # Mesure du temps d'exécution pour chaque épisode

        while not done:
            # Exécuter l'action et obtenir la récompense et le nouvel état
            next_state, reward, done, info = env.step(action)

            # Choix de la prochaine action en utilisant une politique epsilon-greedy
            next_action = epsilon_greedy(next_state, Q, epsilon)

            # Mettre à jour la table Q en utilisant l'équation de Q-learning
            Q = update_Q(state, action, reward, next_state, Q, learning_rate, discount_factor)

            # Mettre à jour l'état et l'action actuels
            state = next_state
            action = next_action

        end_time = time.time()  # Mesure du temps d'ex
        episode_time = end_time - start_time  # Calcul du temps nécessaire à l'agent pour atteindre l'état final

        # Mettre à jour le taux d'exploration
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q

# Définir la fonction pour entraîner l'agent en utilisant l'algorithme SARSA
def sarsa(env, num_episodes, learning_rate, discount_factor, epsilon, epsilon_min, epsilon_decay):
    # Initialiser la table Q
    Q = initialize_Q(env)

    # Boucle sur tous les épisodes
    for i in range(num_episodes):
        # Réinitialiser l'environnement pour un nouvel épisode
        state = env.reset()
        done = False

        # Choix d'une action en utilisant une politique epsilon-greedy
        action = epsilon_greedy(state, Q, epsilon)

        # Afficher la progression de l'entraînement dans le terminal
        if (i+1) % 1000 == 0:
            print("Episode ", i+1, " sur ", num_episodes)

        start_time = time.time()  # Mesure du temps d'exécution pour chaque épisode

        while not done:
            # Exécuter l'action et obtenir la récompense et le nouvel état
            next_state, reward, done, info = env.step(action)

            # Choix de la prochaine action en utilisant une politique epsilon-greedy
            next_action = epsilon_greedy(next_state, Q, epsilon)

            # Mettre à jour la table Q en utilisant l'équation SARSA
            Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * Q[next_state, next_action] - Q[state, action])

            # Mettre à jour l'état et l'action actuels
            state = next_state
            action = next_action

        end_time = time.time()  # Mesure du temps d'exécution pour chaque épisode
        episode_time = end_time - start_time  # Calcul du temps nécessaire à l'agent pour atteindre l'état final

        # Mettre à jour le taux d'exploration
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q

# Définir la fonction pour entraîner l'agent en utilisant l'algorithme Monte Carlo
def monte_carlo(env, num_episodes):
    # Initialiser la table Q et le compteur de visites
    Q = initialize_Q(env)
    N = np.zeros([env.observation_space.n, env.action_space.n])

    # Boucle sur tous les épisodes
    for i in range(num_episodes):
        # Initialiser les listes pour les états, actions et récompenses
        states = []
        actions = []
        rewards = []

        # Réinitialiser l'environnement pour un nouvel épisode
        state = env.reset()
        done = False

        # Exécuter l'épisode jusqu'à ce qu'on atteigne l'état final
        while not done:
            # Choisir une action aléatoire et exécuter l'action
            action = random_action(env)
            next_state, reward, done, info = env.step(action)

            # Ajouter l'état, l'action et la récompense aux listes
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # Mettre à jour l'état actuel
            state = next_state

        # Mettre à jour la table Q et le compteur de visites en utilisant les récompenses obtenues pour chaque état et action
        G = 0
        for t in range(len(states)-1, -1, -1):
            state = states[t]
            action = actions[t]
            reward = rewards[t]
            G = reward + G
            N[state, action] += 1
            alpha = 1.0 / N[state, action]
            Q[state, action] += alpha * (G - Q[state, action])

    return Q

# Définir la fonction pour tester l'agent sur des épisodes de test
def test_agent(env, num_test_episodes, Q, epsilon, epsilon_min, epsilon_decay):
    total_rewards = 0
    success_count = 0
    success_time = []

    for i in range(num_test_episodes):
        state = env.reset()
        done = False
        episode_time = 0

        while not done:
            action = np.argmax(Q[state, :])
            next_state, reward, done, info = env.step(action)
            total_rewards += reward
            episode_time += 1
            state = next_state

            if done and reward == 1:
                success_count += 1
                success_time.append(episode_time)

    # Calculer la récompense moyenne, le taux de réussite et le temps moyen pour atteindre l'état final (sur les épisodes réussis)
    average_reward = total_rewards / num_test_episodes
    success_rate = success_count / num_test_episodes * 100.0
    average_success_time = sum(success_time) / len(success_time)

    # Calculer le meilleur et le pire temps pour atteindre l'état final
    if len(success_time) > 0:
        best_time = min(success_time)
        worst_time = max(success_time)

    return (average_reward, success_rate, average_success_time, best_time, worst_time)


# Créer l'environnement Frozen Lake
env = gym.make("FrozenLake-v1")


# Définir le nombre d'épisodes et le taux d'apprentissage
num_episodes = 100000
test_episodes = int(num_episodes * 0.2)  # 20% du nombre total d'épisodes pour les tests
learning_rate = 0.8
discount_factor = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999

# Tester le temps d'exécution de l'algorithme Q-learning
start_time = time.time()
Q_q_learning = q_learning(env, num_episodes, learning_rate, discount_factor, epsilon, epsilon_min, epsilon_decay)
end_time = time.time()
q_learning_time = end_time - start_time

# Tester l'agent entraîné sur des épisodes de test
test_results_q_learning = test_agent(env, test_episodes, Q_q_learning,epsilon, epsilon_min, epsilon_decay)

# Tester le temps d'exécution de l'algorithme SARSA
start_time = time.time()
Q_sarsa = sarsa(env, num_episodes, learning_rate, discount_factor, epsilon, epsilon_min, epsilon_decay)
end_time = time.time()
sarsa_time = end_time - start_time

# Tester l'agent entraîné sur des épisodes de test
test_results_sarsa = test_agent(env, test_episodes, Q_sarsa, epsilon, epsilon_min, epsilon_decay)


# Tester le temps d'exécution de l'algorithme SARSA
start_time = time.time()
Q_monte_carlo = monte_carlo(env, num_episodes)
end_time = time.time()
monte_carlo_time = end_time - start_time


# Tester l'agent entraîné sur des épisodes de test
test_results_monte_carlo = test_agent(env, test_episodes, Q_monte_carlo, epsilon, epsilon_min, epsilon_decay)


# Comparaison des temps d'exécution des trois algorithmes
print("Temps d'exécution pour Q-learning : ", q_learning_time, " secondes")
print("Temps d'exécution pour SARSA : ", sarsa_time, " secondes")
print("Temps d'exécution pour Monte Carlo : ", monte_carlo_time, " secondes")

# Comparaison de la récompense moyenne des trois algorithmes
print("Récompense moyenne pour Q-learning : ", test_results_q_learning[0])
print("Récompense moyenne pour SARSA : ", test_results_sarsa[0])
print("Récompense moyenne pour Monte Carlo : ", test_results_monte_carlo[0])

# Comparaison du taux de réussite des trois algorithmes
print("Taux de réussite pour Q-learning : ", test_results_q_learning[1], "%")
print("Taux de réussite pour SARSA : ", test_results_sarsa[1], "%")
print("Taux de réussite pour Monte Carlo : ", test_results_monte_carlo[1], "%")

# Comparaison du temps moyen pour atteindre l'état final (sur les épisodes réussis) des trois algorithmes
print("Temps moyen pour atteindre l'état final (sur les épisodes réussis) pour Q-learning : ", test_results_q_learning[2], " étapes")
print("Temps moyen pour atteindre l'état final (sur les épisodes réussis) pour SARSA : ", test_results_sarsa[2], " étapes")
print("Temps moyen pour atteindre l'état final (sur les épisodes réussis) pour Monte Carlo : ", test_results_monte_carlo[2], " étapes")

# Comparaison du meilleur temps pour atteindre l'état final des trois algorithmes
print("Meilleur temps pour atteindre l'état final pour Q-learning : ", test_results_q_learning[3], " étapes")
print("Meilleur temps pour atteindre l'état final pour SARSA : ", test_results_sarsa[3], " étapes")
print("Meilleur temps pour atteindre l'état final pour Monte Carlo : ", test_results_monte_carlo[3], " étapes")

# Comparaison du pire temps pour atteindre l'état final des trois algorithmes
print("Pire temps pour atteindre l'état final pour Q-learning : ", test_results_q_learning[4], " étapes")
print("Pire temps pour atteindre l'état final pour SARSA : ", test_results_sarsa[4], " étapes")
print("Pire temps pour atteindre l'état final pour Monte Carlo : ", test_results_monte_carlo[4], " étapes")



