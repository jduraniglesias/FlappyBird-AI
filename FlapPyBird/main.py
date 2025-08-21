# main.py
import sys
import asyncio
from FlapPyBird.src.flappy import Flappy

def main():
    mode = "train"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()  # e.g. "replay" from: python -m FlapPyBird.main replay
    asyncio.run(Flappy().start(mode=mode))

if __name__ == "__main__":
    main()
