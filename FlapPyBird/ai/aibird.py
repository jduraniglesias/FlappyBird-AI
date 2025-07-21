from .nn import NN
from FlapPyBird.src.entities.player import Player
from FlapPyBird.src.entities.score import Score
from FlapPyBird.src.entities.player import PlayerMode

class AIBird:
    def __init__(self, config):
        self.player = Player(config)
        self.score  = Score(config)
        self.model  = NN()
        self.frames_survived = 0
        self.alive = True
        self.fitness = 0
    @classmethod
    def from_model(cls, model, config):
        bird = cls(config)
        bird.model = model
        return bird