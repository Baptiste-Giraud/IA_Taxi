import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog

from SARSA.SARSA_Train import train as train_S
from QLearning.QLearning_Train import train as train_QL
from MonteCarlo.MC_Train import monte_carlo_e_soft as train_MC

from QLearning.QLearning_Play import play as PlayQ
from MonteCarlo.MC_Play import play as play_MC
from SARSA.SARSA_Play import play as play_S


class HCI(QWidget):

    def __init__(self):
        super().__init__()
        self.args = {}
        self.mode = self.__getMode()
        self.algo = self.__getAlgo()
        self.setParam(self.algo)
        if self.mode == "Training":
            self.askParam()
            print("Using Training mode with {} algorithm.".format(self.algo))
            print("Selected Parameters: ", self.args)
            self.train(self.algo)
        elif self.mode == "Play":
            print("Using Play mode with {} algorithm.".format(self.algo))
            self.play(self.algo)

    def __getMode(self) -> str:
        '''
        Display a dropdown selector with 2 modes (Training and Play)
        
        Returns the selected mode
        '''
        return self.__getChoice("Mode", "Mode:", ("Training", "Play"))

    def __getAlgo(self) -> str:
        '''
        Display a dropdown selector with a list of available algorithms
        
        Returns the name of the selected algorithm
        '''
        return self.__getChoice(
            "Get item", "Algorithm:", ("Monte Carlo", "Q-Learning", "SARSA"))

    def setParam(self, algo: str):
        '''
        Based on the given algorithm name, creates a list of parameters and display an input field for each param.
        '''
        if algo == "Q-Learning" or algo == "SARSA":
            self.args = {
                "lr": 0.01 if algo == "Q-Learning" else 0.85,
                "episodes": 25000 if algo == "Q-Learning" else 10000,
                "gamma": 0.99,
                "epsilon": 1,
                "min_epsilon": 0.001,
                "decay_rate": 0.01
            }
        elif algo == "Monte Carlo":
            self.args = {"episodes": 500000, "epsilon": 0.01}

    def askParam(self):
        for key in self.args:
            if key == "episodes":
                self.__getInteger("Number of Episodes",
                                  key,
                                  default=self.args[key],
                                  min=1,
                                  max=1000000)
            else:
                self.__getDouble(key,
                                 key,
                                 default=self.args[key],
                                 min=0,
                                 max=1,
                                 decimals=4)

    def train(self, algo: str):
        '''
        Based on the selected algorithm, trains it with the previously selected parameters.
        '''
        plt.ion()
        if algo == "Monte Carlo":
            train_MC(episodes=self.args["episodes"],
                     epsilon=self.args["epsilon"],
                     path="MonteCarlo/policy.pkl")
        elif algo == "Q-Learning":
            train_QL(episodes=self.args["episodes"],
                     lr=self.args["lr"],
                     gamma=self.args["gamma"],
                     epsilon=self.args["epsilon"],
                     max_epsilon=self.args["epsilon"],
                     min_epsilon=self.args["min_epsilon"],
                     epsilon_decay=self.args["decay_rate"],
                     show_empty=False,
                     path_table="QLearning/qtable",
                     path_graph="QLearning/QLearning_graph.png")
        elif algo == "SARSA":
            train_S(episodes=self.args["episodes"],
                    gamma=self.args["gamma"],
                    epsilon=self.args["epsilon"],
                    max_epsilon=self.args["epsilon"],
                    min_epsilon=self.args["min_epsilon"],
                    epsilon_decay=self.args["decay_rate"],
                    alpha=self.args["lr"],
                    path_table="SARSA/qtable",
                    path_graph="SARSA/SARSA_graph.png")
        plt.ioff()

    def play(self, algo: str):
        '''
        Based on the selected algorithm, plays the game using the trained model.
        '''
        play_action = self.__getChoice("Select Play Action", "Action:", ("PlayQ", "PlayMC", "PlayS"))

        if algo == "Monte Carlo":
            if play_action == "PlayMC":
                play_MC()
        elif algo == "Q-Learning":
            if play_action == "PlayQ":
                PlayQ()
        elif algo == "SARSA":
            if play_action == "PlayS":
                play_S()

    def __getChoice(self, title, name, items: tuple[str]) -> str:
        item, okPressed = QInputDialog.getItem(self, title, name, items, 0,
                                               False)
        if okPressed and item:
            return item

    def __getDouble(self, placeholder: str, target: str, default: float,
                    min: float, max: float, decimals: int):
        d, okPressed = QInputDialog.getDouble(self, placeholder, placeholder,
                                              default, min, max, decimals)
        if okPressed:
            self.args[target] = d

    def __getInteger(self, placeholder: str, target: str, default: int,
                     min: int, max: int):
        d, okPressed = QInputDialog.getInt(self, placeholder, "Value:",
                                           default, min, max, 100)
        if okPressed:
            self.args[target] = d


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = HCI()
    sys.exit(app.exec_())
