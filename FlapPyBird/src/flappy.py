import asyncio
import sys

import pygame
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT
from FlapPyBird.ai.nn import NN

from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, Sounds, Window


class Flappy:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        window = Window(288, 512)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images()

        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            sounds=Sounds(),
        )

    async def start(self):
        while True:
            self.background = Background(self.config)
            self.floor = Floor(self.config)
            self.player = Player(self.config)
            self.welcome_message = WelcomeMessage(self.config)
            self.game_over_message = GameOver(self.config)
            self.pipes = Pipes(self.config)
            self.score = Score(self.config)
            await self.splash()
            await self.play()
            await self.game_over()

    async def splash(self):
        """Shows welcome splash screen animation of flappy bird"""

        self.player.set_mode(PlayerMode.SHM)

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    return

            self.background.tick()
            self.floor.tick()
            self.player.tick()
            self.welcome_message.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    def check_quit_event(self, event):
        if event.type == QUIT or (
            event.type == KEYDOWN and event.key == K_ESCAPE
        ):
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (
            event.key == K_SPACE or event.key == K_UP
        )
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap
    async def play(self):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)
        model = NN()
        next_pipe = None

        while True:
            if self.player.collided(self.pipes, self.floor):
                return

            for i, pipe in enumerate(self.pipes.upper):
                if self.player.crossed(pipe):
                    self.score.add()
            
            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            
            def normalize_values(player, pipe_x, pipe_top_y, pipe_bottom_y, window):
                # For vertical position
                norm_y = player.y / window.height
                # For vertical velocity (assumes NORMAL mode settings)
                vel_range = player.max_vel_y - player.min_vel_y
                norm_vel = (player.vel_y - player.min_vel_y) / vel_range if vel_range != 0 else 0.5
                # Horizontal distance to pipe
                pipe_dx = (pipe_x - player.x) / window.width
                # Pipe gap positions
                norm_pipe_top = pipe_top_y / window.height
                norm_pipe_bottom = pipe_bottom_y / window.height
                return [norm_y, norm_vel, pipe_dx, norm_pipe_top, norm_pipe_bottom]    

            next_pipe = None
            # Setup state contents
            for pipe in self.pipes.upper:
                if pipe.x + pipe.w > self.player.x:
                    next_pipe = pipe
                    break
            if next_pipe:
                pipe_x = next_pipe.x
                pipe_top_y = next_pipe.y + next_pipe.h
                pipe_gap = self.pipes.pipe_gap
                pipe_bottom_y = pipe_top_y + pipe_gap
                state = normalize_values(self.player, pipe_x, pipe_top_y, pipe_bottom_y, self.config.window)
                output = model.forward(state)
                if output > 0.5:
                    self.player.flap()

            # Skipped since we are now using AI to control
            # for event in pygame.event.get():
            #     self.check_quit_event(event)
                
            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    async def game_over(self):
        """crashes the player down and shows gameover image"""

        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    if self.player.y + self.player.h >= self.floor.y - 1:
                        return

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()

            self.config.tick()
            pygame.display.update()
            await asyncio.sleep(0)
