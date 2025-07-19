from .nn import NN
from FlapPyBird.src.entities.player import Player
from FlapPyBird.src.entities.score import Score
from FlapPyBird.src.entities.player import PlayerMode

class AIBird:
    def __init__(self, config):
        self.config = config
        self.player = Player(config)
        self.score  = Score(config)
        self.model  = NN()
        self.fitness = 0
        self.alive   = True
        self.frames_survived = 0
        self.player.set_mode(PlayerMode.NORMAL)