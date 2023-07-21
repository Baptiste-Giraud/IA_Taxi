import pygame
from pygame.locals import *
import numpy as np
import time
import gym
import easygui
import subprocess
import tkinter as tk
from tkinter import filedialog

from QLearning.QLearning_Play import play as play_Q
from MonteCarlo.MC_Play import play as play_MC
from SARSA.SARSA_Play import play as play_S

from QLearning.QLearning_Train import train as train_Q
from MonteCarlo.MC_Train import train as train_MC
from SARSA.SARSA_Train import train as train_S
# Initialisation de Pygame
pygame.init()
pygame.mixer.init()

# Paramètres de la fenêtre
largeur = 800
hauteur = 600
fenetre = pygame.display.set_mode((largeur, hauteur))
pygame.display.set_caption("Taxi V3")
background = pygame.image.load('landcape.jpeg')
background = pygame.transform.scale(background, (largeur, hauteur))  # redimensionner l'image
background_rect = background.get_rect()


# Couleurs
BLANC = (255, 255, 255)
BLEU = (0, 0, 255)
JAUNE = (255, 255, 0)

# Chargement des ressources
taxi = pygame.image.load("taxi.png")
taxi = pygame.transform.scale(taxi, (taxi.get_width() // 2, taxi.get_height() // 2))
taxi_rect = taxi.get_rect()
taxi_rect.center = (0, hauteur - taxi_rect.height // 2)

personne = pygame.image.load("personne.png")
personne = pygame.transform.scale(personne, (personne.get_width() // 3, personne.get_height() // 3))
personne_rect = personne.get_rect()
personne_rect.center = (largeur // 2, hauteur - taxi_rect.height // 2)

# Position et vitesse du taxi
position_taxi = [0, hauteur - taxi_rect.height]
vitesse_taxi = 1

# Variables de jeu
personne_capturee = False
animation_terminee = False

# Variables de temps
attente_personne = 4  # Durée d'attente en secondes lorsque le taxi atteint le personnage
attente_recommencer = 3  # Durée d'attente en secondes avant de recommencer l'animation
attente_sortie_ecran = 2  # Durée d'attente en secondes avant de sortir complètement de l'écran

# Création de l'environnement Taxi V3
env = gym.make("Taxi-v3")
q_table = np.load("QLearning/models/qtable.npy")
state = env.reset()

# # Placeholder function for train_Q(), train_S(), train_MC()
# def train_Q(gamma, epsilon, temps_requis):
#     # Insérez ici votre code d'entraînement pour Q-learning
#     pass

# def train_S(gamma, epsilon, temps_requis):
#     # Insérez ici votre code d'entraînement pour SARSA
#     pass

# def train_MC(gamma, epsilon, temps_requis):
#     # Insérez ici votre code d'entraînement pour Monte Carlo
#     pass

# Boucle de jeu
def main_game():
    scroll_x = 0
    pygame.mixer.music.load('taxi.mp3')
    pygame.mixer.music.play(-1)
    global position_taxi, personne_capturee, animation_terminee, state
    en_cours = True
    clock = pygame.time.Clock()
    while en_cours:
        # Gestion des événements
        rel_x = scroll_x % background.get_rect().width
        fenetre.blit(background, (rel_x - background.get_rect().width, 0))
        if rel_x < largeur:
            fenetre.blit(background, (rel_x, 0))
        scroll_x -= 1
        for event in pygame.event.get():
            if event.type == QUIT:
                en_cours = False
                pygame.quit()
                return

        # Déplacement du taxi
        if not animation_terminee:
            if position_taxi[0] < largeur // 2 - taxi_rect.width // 2:
                position_taxi[0] += vitesse_taxi
            else:
                if not personne_capturee:
                    if pygame.time.get_ticks() > attente_personne * 1000:
                        personne_capturee = True
                        personne_rect.centery = position_taxi[1] + taxi_rect.height // 2
                else:
                    if pygame.time.get_ticks() > attente_recommencer * 1000:
                        position_taxi[0] += vitesse_taxi
                        if position_taxi[0] > largeur:
                            animation_terminee = True
                            time.sleep(attente_sortie_ecran)
                            menu()


        # Création du texte
        font = pygame.font.Font(None, 36)  # Changez la taille de la police si nécessaire
        text = font.render('Taxi V3 IA Epitech', True, BLEU if (pygame.time.get_ticks() // 1000) % 2 == 0 else JAUNE)

        # Affichage du texte
        text_rect = text.get_rect(center=(largeur // 2, 50))  # Ajustez la position Y si nécessaire
        fenetre.blit(text, text_rect)

        # Affichage du taxi
        fenetre.blit(taxi, position_taxi)

        # Affichage de la personne
        if not personne_capturee:
            fenetre.blit(personne, personne_rect)

        # Mettre à jour l'affichage
        pygame.display.flip()

        # Jouer une étape du jeu Taxi V3
        if not animation_terminee:
            action = np.argmax(q_table[state])
            state, _, done, _ = env.step(action)
            if done:
                state = env.reset()

def menu():
    font = pygame.font.Font(None, 36)
    bordure_epaisseur = 5  # Changez cette valeur pour ajuster l'épaisseur des bordures
    couleur_bordure_normale = (255, 0, 0)
    couleur_bordure_survol = (200, 0, 0)
    couleur_texte_normale = (255, 255, 0)  # Jaune
    couleur_texte_survol = (255, 255, 255)

    # Chargement de l'image de fond
    fond = pygame.image.load('fond.jpeg')
    fond = pygame.transform.scale(fond, (largeur, hauteur))  # redimensionne l'image pour qu'elle s'adapte à la fenêtre

    play_text = font.render('Play', 1, couleur_texte_normale)
    play_rect = play_text.get_rect(center=(largeur / 3, hauteur / 2))  # position ajustée

    train_text = font.render('Train', 1, couleur_texte_normale)
    train_rect = train_text.get_rect(center=(2 * largeur / 3, hauteur / 2))  # position ajustée

    en_cours = True
    while en_cours:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == QUIT:
                en_cours = False
                pygame.quit()
                return
            if event.type == MOUSEBUTTONDOWN:
                if play_rect.collidepoint(mouse_pos):
                    root = tk.Tk()
                    root.withdraw()
                    file_path = filedialog.askopenfilename()
                    print("File chosen:", file_path)
                    # Séparation du chemin en parties
                    chemin_parts = file_path.split("/")
                    nom_fichier = chemin_parts[-1]

                    # Trouver la position du répertoire "models"
                    index_models = chemin_parts.index("models")

                    # Extraire la partie du chemin avant "models"
                    chemin_avant_models = "/".join(chemin_parts[:index_models])

                    # Choose play time
                    msg = "Please enter the play time."
                    title = "Play parameters"
                    fieldNames = ["Time"]
                    fieldValues = ["0"]  # default value
                    fieldValues = easygui.multenterbox(msg, title, fieldNames, fieldValues)

                    # make sure that none of the fields was left blank
                    while 1:
                        if fieldValues is None: break
                        errmsg = ""
                        for i in range(len(fieldNames)):
                            if fieldValues[i].strip() == "":
                                errmsg += ('"%s" is a required field.\n\n' % fieldNames[i])
                        if errmsg == "": 
                            break  # no problems found
                        fieldValues = easygui.multenterbox(errmsg, title, fieldNames, fieldValues)

                    # Call play function with entered time
                    temps_partie = int(fieldValues[0])

                    if "QLearning" in chemin_avant_models:
                        play_Q(path="./QLearning/models/qtable.npy", p_time=temps_partie, render=True)
                    if "MonteCarlo" in chemin_avant_models:
                        play_MC(path_file="./MonteCarlo/models/policy.pkl", p_time=temps_partie, render=True)
                    if "SARSA" in chemin_avant_models:
                        play_S(path="./SARSA/models/qtable.npy", p_time=temps_partie, render=True)

                    reset_game()
                    main_game()
                    if file_path:
                        subprocess.run(file_path, shell=True)
                elif train_rect.collidepoint(mouse_pos):
                    train_game()

        # Dessine l'image de fond
        fenetre.blit(fond, (0, 0))

        if play_rect.collidepoint(mouse_pos):
            play_text = font.render('Play', 1, couleur_texte_survol)
            pygame.draw.rect(fenetre, couleur_bordure_survol, play_rect.inflate(20, 20), bordure_epaisseur)
        else:
            play_text = font.render('Play', 1, couleur_texte_normale)
            pygame.draw.rect(fenetre, couleur_bordure_normale, play_rect.inflate(20, 20), bordure_epaisseur)
        fenetre.blit(play_text, play_rect)

        if train_rect.collidepoint(mouse_pos):
            train_text = font.render('Train', 1, couleur_texte_survol)
            pygame.draw.rect(fenetre, couleur_bordure_survol, train_rect.inflate(20, 20), bordure_epaisseur)
        else:
            train_text = font.render('Train', 1, couleur_texte_normale)
            pygame.draw.rect(fenetre, couleur_bordure_normale, train_rect.inflate(20, 20), bordure_epaisseur)
        fenetre.blit(train_text, train_rect)

        pygame.display.flip()


# Reset Game Variables
def reset_game():
    global position_taxi, personne_capturee, animation_terminee, state
    position_taxi = [0, hauteur - taxi_rect.height]
    personne_capturee = False
    animation_terminee = False
    state = env.reset()

# Train function
def train_game():
    episodes = 2000
    gamma = 0.95
    epsilon = 1
    alpha = 0.85
    lr = 0.01

    # Option de formation - Choisissez l'algorithme
    algo_choices = ["Q-learning", "SARSA", "Monte Carlo"]
    algo_choice = easygui.buttonbox("Choisissez l'algorithme d'apprentissage à entraîner.", choices=algo_choices)

    # Choix des paramètres de formation
    msg = "Veuillez entrer les paramètres de formation."
    title = "Paramètres de formation"

    if algo_choice == "SARSA":
        fieldNames = ["Episodes", "Gamma", "Epsilon", "Alpha"]
        fieldValues = [str(episodes), str(gamma), str(epsilon), str(alpha)]  # valeurs par défaut
    elif algo_choice == "Q-learning":
        fieldNames = ["Episodes", "Gamma", "Epsilon", "Learning rate"]
        fieldValues = [str(episodes), str(gamma), str(epsilon), str(lr)]  # valeurs par défaut
    elif algo_choice == "Monte Carlo":
        fieldNames = ["Episodes", "Epsilon"]
        fieldValues = [str(episodes), str(epsilon)]  # valeurs par défaut

    fieldValues = easygui.multenterbox(msg, title, fieldNames, fieldValues)

    # assurez-vous que rien n'a été laissé vide
    while 1:
        if fieldValues is None: break
        errmsg = ""
        for i in range(len(fieldNames)):
            if fieldValues[i].strip() == "":
                errmsg += ('"%s" est un champ requis.\n\n' % fieldNames[i])
        if errmsg == "":
            break  # no problems found
        fieldValues = easygui.multenterbox(errmsg, title, fieldNames, fieldValues)

    # Entraine chaque algorithme avec les paramètres donnés
    if algo_choice == "Q-learning":
        episodes = int(fieldValues[0])
        gamma = float(fieldValues[1])
        epsilon = float(fieldValues[2])
        lr = float(fieldValues[3])
        train_Q(episodes=episodes, gamma=gamma, epsilon=epsilon, lr=lr)
    elif algo_choice == "SARSA":
        episodes = int(fieldValues[0])
        gamma = float(fieldValues[1])
        epsilon = float(fieldValues[2])
        alpha = float(fieldValues[3])
        train_S(episodes=episodes, gamma=gamma, epsilon=epsilon, alpha=alpha)
    elif algo_choice == "Monte Carlo":
        episodes = int(fieldValues[0])
        epsilon = float(fieldValues[1])
        train_MC(episodes=episodes, epsilon=epsilon)


if __name__ == "__main__":
    main_game()
