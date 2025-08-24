import argparse
import asyncio
import os

from FlapPyBird.src.flappy import Flappy

# DQN helpers (make sure you created these files from our earlier steps)
from FlapPyBird.src.dqn_train import train_dqn, play_dqn


def main():
    p = argparse.ArgumentParser(
        description="Run FlappyBird-AI with NEAT-style evolution or DQN."
    )
    p.add_argument(
        "mode",
        choices=["neat-train", "neat-replay", "dqn-train", "dqn-play"],
        help="What to run."
    )
    # Common-ish knobs
    p.add_argument("--episodes", type=int, default=1000,
                   help="Number of episodes (DQN train/play).")
    p.add_argument("--ckpt", default=None,
                   help="Checkpoint path. "
                        "For NEAT replay defaults to checkpoints/champion.npz; "
                        "for DQN train/play defaults to checkpoints/dqn.pt.")
    p.add_argument("--render", action="store_true",
                   help="Render during DQN training (slower). Ignored for NEAT train.")
    args = p.parse_args()

    # Instantiate the Pygame app/config once
    app = Flappy()

    # ------- NEAT / Evolution modes -------
    if args.mode == "neat-train":
        # Your existing async training loop with evolution
        return asyncio.run(app.start(mode="train"))

    if args.mode == "neat-replay":
        # Champion replay; allow overriding the checkpoint path
        ckpt = args.ckpt or os.path.join("checkpoints", "champion.npz")
        return asyncio.run(app.replay_champion(path=ckpt))

    # ------- DQN modes -------
    if args.mode == "dqn-train":
        ckpt = args.ckpt or os.path.join("checkpoints", "dqn.pt")
        os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
        # Headless by default for speed (render only if you pass --render)
        train_dqn(
            config=app.config,
            episodes=args.episodes,
            render=args.render,
            save_path=ckpt,
            batch_size=64,
            warmup_steps=2000,
        )
        return

    if args.mode == "dqn-play":
        ckpt = args.ckpt or os.path.join("checkpoints", "dqn.pt")
        # Play N episodes greedily with rendering
        play_dqn(
            config=app.config,
            load_path=ckpt,
            episodes=args.episodes,
        )
        return


if __name__ == "__main__":
    main()
