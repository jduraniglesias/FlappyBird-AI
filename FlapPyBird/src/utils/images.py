import random
import os
from typing import List, Tuple

import pygame

from .constants import BACKGROUNDS, PIPES, PLAYERS


class Images:
    numbers: List[pygame.Surface]
    game_over: pygame.Surface
    welcome_message: pygame.Surface
    base: pygame.Surface
    background: pygame.Surface
    player: Tuple[pygame.Surface]
    pipe: Tuple[pygame.Surface]

    def __init__(self) -> None:
        base_path = os.path.dirname(__file__)
        sprite_path = os.path.abspath(os.path.join(base_path, "../../assets/sprites"))
        self.numbers = [
            pygame.image.load(os.path.join(sprite_path, f"{num}.png")).convert_alpha()
            for num in range(10)
        ]

        # game over sprite
        self.game_over = pygame.image.load(
            os.path.join(sprite_path, "gameover.png")
        ).convert_alpha()

        # welcome_message sprite for welcome screen
        self.welcome_message = pygame.image.load(
            os.path.join(sprite_path, "message.png")
        ).convert_alpha()

        # base (ground) sprite
        self.base = pygame.image.load(
            os.path.join(sprite_path, "base.png")
        ).convert_alpha()

        self.randomize(sprite_path)

    def randomize(self, sprite_path: str):
        # select random indices
        rand_bg = random.randint(0, len(BACKGROUNDS) - 1)
        rand_player = random.randint(0, len(PLAYERS) - 1)
        rand_pipe = random.randint(0, len(PIPES) - 1)

        # Load background image
        self.background = pygame.image.load(
            os.path.join(sprite_path, BACKGROUNDS[rand_bg])
        ).convert()

        # Load player animation frames
        self.player = (
            pygame.image.load(os.path.join(sprite_path, PLAYERS[rand_player][0])).convert_alpha(),
            pygame.image.load(os.path.join(sprite_path, PLAYERS[rand_player][1])).convert_alpha(),
            pygame.image.load(os.path.join(sprite_path, PLAYERS[rand_player][2])).convert_alpha(),
        )

        # Load pipe image and flipped version
        pipe_img_path = os.path.join(sprite_path, PIPES[rand_pipe])
        self.pipe = (
            pygame.transform.flip(pygame.image.load(pipe_img_path).convert_alpha(), False, True),
            pygame.image.load(pipe_img_path).convert_alpha(),
        )

