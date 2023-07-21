import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

from QLearning.QLearning_Play import play as PlayQ
from MonteCarlo.MC_Play import play as play_MC
from SARSA.SARSA_Play import play as play_S

def launch_model(model_combobox, epsilon_text, turn_text, learning_text):
    model = model_combobox.get()
    epsilon = epsilon_text.get("1.0", tk.END).strip()
    turn = turn_text.get("1.0", tk.END).strip()
    learning_rate = learning_text.get("1.0", tk.END).strip()

    # if model == "MonteCarlo":
    #     play_MC()
    # if model == "SARSA":
    #     play_S()
    # if model == "QLearning":
    #     PlayQ()
            
    print("Modèle :", model)
    print("Epsilon :", epsilon)
    print("Turn :", turn)
    print("Learning Rate :", learning_rate)

def train_model():
    for widget in window.winfo_children():
        widget.destroy()

    label = tk.Label(window, text="Choisissez un modèle :", font=("Arial", 24))
    label.pack(pady=10)

    models = ["MonteCarlo", "SARSA", "QLearning"]
    model_combobox = ttk.Combobox(window, values=models)
    model_combobox.pack(pady=10)

    title_label = tk.Label(window, text="Choix des arguments", font=("Arial", 24))
    title_label.pack(pady=50)

    epsilon_label = tk.Label(window, text="Epsilon", font=("Arial", 12))
    epsilon_label.pack()

    epsilon_text = tk.Text(window, height=1, width=30)
    epsilon_text.pack(pady=10)

    turn_label = tk.Label(window, text="Turn", font=("Arial", 12))
    turn_label.pack()

    turn_text = tk.Text(window, height=1, width=30)
    turn_text.pack(pady=10)

    learning_label = tk.Label(window, text="Learning Rate", font=("Arial", 12))
    learning_label.pack()

    learning_text = tk.Text(window, height=1, width=30)
    learning_text.pack(pady=10)

    launch_button = tk.Button(window, text="Lancer l'entraînement", command=lambda: launch_model(model_combobox, epsilon_text, turn_text, learning_text))
    launch_button.pack(pady=10)

def find_train_model(file_path):
    if "SARSA" in file_path:
        path_model = file_path.split("T-AIA-902-MAR_1")[1].strip("/")
        # print("load sarsa", path_model)
        play_S(path=path_model, render=True)
    else:
        print("test")

def load_checkpoint():
    root = tk.Tk()
    root.withdraw()  # Cacher la fenêtre principale de Tkinter

    file_path = filedialog.askopenfilename(initialdir="./", filetypes=[("Checkpoint files", "*.ckpt *.npy *.pkl *.pt")])

    if file_path:
        find_train_model(file_path)
    else:
        print("Aucun fichier sélectionné.")

window = tk.Tk()
window.title("Interface de l'IA")
window.state('zoomed')

title_project = tk.Label(window, text="Choisissez une option :", font=("Arial", 24))
title_project.place(x=window.winfo_screenwidth()//2, y=window.winfo_screenheight()//2-200, anchor=tk.CENTER)

train_button = tk.Button(window, text="Lancer l'entraînement", command=train_model, width=25)
train_button.place(x=window.winfo_screenwidth()//2-250, y=window.winfo_screenheight()//2-10)

load_button = tk.Button(window, text="Charger un checkpoint", command=load_checkpoint, width=25)
load_button.place(x=window.winfo_screenwidth()//2+50, y=window.winfo_screenheight()//2-10)

window.mainloop()