from .nn import NN
from FlapPyBird.src.entities.player import Player
from FlapPyBird.src.entities.score import Score
from FlapPyBird.src.entities.player import PlayerMode

class AIBird:
    def __init__(self, config):
        self.player = Player(config)
        self.model  = NN()
        self.fitness = 0
        self.frames_survived = 0
        self.alive = True

    @classmethod
    def from_model(cls, model: NN, config):
        """Create a new AIBird using the given NN, resetting all runtime state."""
        bird = cls(config)
        bird.model = model
        bird.fitness = 0
        bird.frames_survived = 0
        bird.alive = True
        return bird