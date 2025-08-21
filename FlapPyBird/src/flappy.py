import asyncio
import sys
import numpy as np
import pygame
import contextlib
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT
from FlapPyBird.ai.nn import NN
from FlapPyBird.ai.aibird import AIBird
from FlapPyBird.ai.evolution import evolve_population
import os

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
        self.ckpt_dir = "checkpoints"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_ever_fitness = float("-inf")
        self.best_ever_gen = 0
        self.champion_path = os.path.join(self.ckpt_dir, "champion.npz")
        self._should_quit = False
        self._quit_task = None
        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            sounds=Sounds(),
        )

    async def _stdin_quit_watcher(self):
        loop = asyncio.get_running_loop()
        try:
            while True:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    continue
                if line.strip().lower() == "quit":
                    print("Quitting…")
                    self._should_quit = True
                    break
        except Exception as e:
            print(f"quit watcher stopped: {e}")

    async def _stop_quit_watcher(self):
        if self._quit_task and not self._quit_task.done():
            self._quit_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._quit_task

    async def replay_champion(self, path: str | None = None):
        path = path or self.champion_path
        if not os.path.exists(path):
            print("No champion checkpoint found, train first")
            return

        data = np.load(path, allow_pickle=False)
        meta_gen = int(data["meta_gen"]) if "meta_gen" in data.files else 0
        meta_fit = float(data["meta_fitness"]) if "meta_fitness" in data.files else 0.0

        net = NN()
        net.w1, net.b1 = data["w1"], data["b1"]
        net.w2, net.b2 = data["w2"], data["b2"]

        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.pipes = Pipes(self.config)

        champ = AIBird.from_model(net, self.config)
        champ.alive = True
        champ.score.reset()

        print(f"Replaying champion (gen {meta_gen}, fitness {meta_fit})")
        await self.play([champ])


    async def start(self, mode="train"):
        self._should_quit = False
        self._quit_task = asyncio.create_task(self._stdin_quit_watcher())
        try:
            if mode == "replay":
                print("Replay Mode")
                await self.replay_champion()
                return

            population_size = 100
            population = [AIBird(self.config) for _ in range(population_size)]
            gen = 0
            while not self._should_quit:
                gen += 1
                print(f"=== Generation {gen} ===")

                self.background = Background(self.config)
                self.floor = Floor(self.config)
                self.pipes = Pipes(self.config)

                await self.play(population)
                if self._should_quit:
                    break

                elite = sorted(population, key=lambda b: b.fitness, reverse=True)[:int(len(population) * 0.2)]
                print(f"Top birds before evolution: {[b.fitness for b in elite]}")
                population = evolve_population(population, self.config)
                print(f"Top birds after evolution:  {[b.fitness for b in population[:len(elite)]]}")
                best = max(population, key=lambda b: b.fitness)
                print(f"Gen {gen} best fitness = {best.fitness}")
                if best.fitness > self.best_ever_fitness:
                    self.best_ever_fitness = best.fitness
                    self.best_ever_gen = gen
                    np.savez(
                        self.champion_path,
                        w1=best.model.w1, b1=best.model.b1,
                        w2=best.model.w2, b2=best.model.b2,
                        meta_gen=np.array(gen, dtype=np.int32),
                        meta_fitness=np.array(best.fitness, dtype=np.float32),
                    )
                    print(f"Saved champion from gen {gen} @ fitness {best.fitness} -> {self.champion_path}")
        finally:
            await self._stop_quit_watcher()
            pygame.quit()
            sys.exit()

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
    # Todo:
    # So the issue is that the bird thats sideways is not ticking/drawing correctly
    # The sideways bird should fly straight up fast but instead it just falls slowly while making the fly noise
    # The other issue is that sometimes the bird will go straight but in reality should be falling (no fly noise made)

    async def play(self, birds: list[AIBird]):
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

        # --- initialize per-bird stats ---
        for bird in birds:
            bird.score.reset()
            bird.frames_survived = 0
            bird.alive = True

            # keep game window alive
            for event in pygame.event.get():
                self.check_quit_event(event)
            # new stats for penalty
            bird.flap_count = 0
            bird.penalty = 0.0
            bird.hover_frames = 0

        while any(bird.alive for bird in birds):
            if self._should_quit:
                return
            self.background.tick()
            self.floor.tick()
            self.pipes.tick()

            for bird in birds:
                if not bird.alive:
                    continue

                # find the next pipe
                next_pipe = None
                for pipe in self.pipes.upper:
                    if pipe.x + pipe.w > bird.player.x:
                        next_pipe = pipe
                        break

                # make decision
                if next_pipe:
                    pipe_x = next_pipe.x
                    pipe_top_y = next_pipe.y + next_pipe.h
                    pipe_gap = self.pipes.pipe_gap
                    pipe_bottom_y = pipe_top_y + pipe_gap

                    state = normalize_values(
                        bird.player, pipe_x, pipe_top_y, pipe_bottom_y, self.config.window
                    )
                    output = bird.model.forward(state)
                    if output > 0.5:
                        bird.player.flap()
                        bird.flap_count += 1

                # tick physics & scoring
                bird.player.tick()
                bird.score.tick()
                bird.frames_survived += 1

                # --- apply per-frame punishment ---
                # 1) Too high: into top 10% of screen
                if bird.player.y < 0.1 * self.config.window.height:
                    bird.penalty += 1.0

                # 2) Hovering: very small velocity for many frames
                if abs(bird.player.vel_y) < 1.0:
                    bird.hover_frames += 1
                    # penalize after 30 consecutive hover frames
                    if bird.hover_frames > 30:
                        bird.penalty += 0.5
                else:
                    bird.hover_frames = 0

                # count pipe passes
                for i, pipe in enumerate(self.pipes.upper):
                    if bird.player.crossed(pipe):
                        bird.score.add()

                # collision check
                if bird.player.collided(self.pipes, self.floor):
                    bird.alive = False
                    # compute final fitness with penalties
                    base_fitness = bird.frames_survived + bird.score.score * 100
                    flap_penalty = bird.flap_count * 0.2
                    bird.fitness = base_fitness - bird.penalty - flap_penalty
                 
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
