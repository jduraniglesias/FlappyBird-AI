# FlapPyBird/src/flappy_env.py
import pygame
import numpy as np

from .entities import Background, Floor, Pipes, Player, Score, PlayerMode
from .utils import GameConfig

class FlappyEnv:
    def __init__(self, config: GameConfig, render: bool = False):
        self.config = config
        self.render = render
        self.background = None
        self.floor = None
        self.pipes = None
        self.player = None
        self.score = None
        self.frames = 0
        self.flap_count = 0

    def _state(self):
        window = self.config.window
        # find next pipe
        next_pipe = None
        for p in self.pipes.upper:
            if p.x + p.w > self.player.x:
                next_pipe = p
                break

        if next_pipe is None:
            pipe_x = self.config.window.width
            pipe_top_y = window.height * 0.3
            pipe_bottom_y = window.height * 0.7
        else:
            pipe_x = next_pipe.x
            pipe_top_y = next_pipe.y + next_pipe.h
            pipe_bottom_y = pipe_top_y + self.pipes.pipe_gap

        norm_y = self.player.y / window.height
        vel_range = self.player.max_vel_y - self.player.min_vel_y
        norm_vel = (self.player.vel_y - self.player.min_vel_y) / vel_range if vel_range else 0.5
        pipe_dx = (pipe_x - self.player.x) / window.width
        norm_pipe_top = pipe_top_y / window.height
        norm_pipe_bottom = pipe_bottom_y / window.height
        return np.array([norm_y, norm_vel, pipe_dx, norm_pipe_top, norm_pipe_bottom], dtype=np.float32)

    def reset(self):
        # fresh world
        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.pipes = Pipes(self.config)
        self.player = Player(self.config)
        self.score = Score(self.config)
        self.player.set_mode(PlayerMode.NORMAL)
        self.frames = 0
        self.flap_count = 0

        # warm-up one frame so everything has consistent positions
        self.background.tick(); self.floor.tick(); self.pipes.tick(); self.player.tick(); self.score.tick()
        if self.render:
            for event in pygame.event.get(): pass
            pygame.display.update()

        return self._state()

    def step(self, action: int):
        # action: 0=do nothing, 1=flap
        if action == 1:
            self.player.flap()
            self.flap_count += 1

        # tick game
        self.background.tick()
        self.floor.tick()
        self.pipes.tick()
        self.player.tick()
        self.score.tick()
        self.frames += 1

        # reward shaping
        reward = 0.01  # live reward
        # pipe pass
        for p in list(self.pipes.upper):
            if self.player.crossed(p):
                self.score.add()
                reward += 1.0

        done = False
        if self.player.collided(self.pipes, self.floor):
            reward -= 1.0
            done = True

        reward -= 0.01 if action == 1 else 0.0

        if self.render:
            for event in pygame.event.get(): pass
            pygame.display.update()
            # keep Pygame responsive but don't slow training if render=False
            self.config.tick()

        return self._state(), float(reward), bool(done), {"score": self.score.score, "frames": self.frames}
