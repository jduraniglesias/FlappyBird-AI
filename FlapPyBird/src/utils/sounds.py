import sys
import os

import pygame


class Sounds:
    die: pygame.mixer.Sound
    hit: pygame.mixer.Sound
    point: pygame.mixer.Sound
    swoosh: pygame.mixer.Sound
    wing: pygame.mixer.Sound

    def __init__(self):
        base = os.path.dirname(__file__)
        audio_path = os.path.abspath(os.path.join(base, "../../assets/audio"))

        if "win" in sys.platform:
            ext = "wav"
        else:
            ext = "ogg"

        self.die = pygame.mixer.Sound(os.path.join(audio_path, f"die.{ext}"))
        self.hit = pygame.mixer.Sound(os.path.join(audio_path, f"hit.{ext}"))
        self.point = pygame.mixer.Sound(os.path.join(audio_path, f"point.{ext}"))
        self.swoosh = pygame.mixer.Sound(os.path.join(audio_path, f"swoosh.{ext}"))
        self.wing = pygame.mixer.Sound(os.path.join(audio_path, f"wing.{ext}"))