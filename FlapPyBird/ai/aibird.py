from .nn import NN
from FlapPyBird.src.entities.player import Player
from FlapPyBird.src.entities.score import Score
from FlapPyBird.src.entities.player import PlayerMode

class AIBird:
    def __init__(self, config):
        # Game entity
        self.player = Player(config)
        self.player.set_mode(PlayerMode.NORMAL)
        # Tracking score and fitness
        self.score = Score(config)
        self.frames_survived = 0
        self.alive = True
        self.fitness = 0
        # Neural network brain
        self.model = NN()

    @classmethod
    def from_model(cls, model: NN, config):
        """
        Create a new AIBird using an existing NN instance (child),
        resetting runtime state but preserving the brain.
        """
        bird = cls(config)
        bird.model = model
        return bird
