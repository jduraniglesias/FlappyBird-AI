# Trainer for DQN
import os
import numpy as np
from .flappy_env import FlappyEnv
from ..ai.dqn import DQNAgent

def train_dqn(
    config,
    episodes: int = 1000,
    render: bool = False,
    save_path: str = "checkpoints/dqn.pt",
    batch_size: int = 64,
    warmup_steps: int = 2000,
):
    """
    runs training for dqn
    config = GameConfig
    render = true shows the bird and false is used for quick training
    save_path = best agent weights
    batch_size = minibatch size for updates
    warmup steps = steps before heavy learning
    """

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    # Create flappy environment
    env = FlappyEnv(config, render=render)
    agent = DQNAgent(state_dim=5, action_dim=2, lr=1e-3, gamma=0.99)

    # tracks most pipe passed
    best_pipes = -1
    total_steps = 0

    for ep in range(1, episodes + 1):
        # start new episode
        s = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        last_loss = None

        while True:
            # pick action
            a, eps = agent.select_action(s)

            # increment step env
            s2, r, done, info = env.step(a)
            ep_reward += r
            ep_steps += 1
            total_steps += 1

            # store transition
            agent.buffer.push(s, a, r, s2, done)
            s = s2

            # start learning
            if total_steps >= warmup_steps:
                loss = agent.optimize(batch_size=batch_size, loss_fn="huber")
                if loss is not None:
                    last_loss = loss
            
            # end of episode
            if done:
                pipes_cleared = info.get("score", 0)
                if pipes_cleared > best_pipes:
                    best_pipes = pipes_cleared
                    agent.save(save_path)
                
                # logs
                print(
                    f"[DQN] Ep {ep:04d}  steps={ep_steps:4d}  "
                    f"pipes={pipes_cleared:3d}  reward={ep_reward:7.2f}  "
                    f"eps={eps:0.02f}  loss={last_loss if last_loss is not None else '-'}"
                )
                break

    # save
    agent.save(save_path)
    print(f"[DQN] Training complete. Best pipes={best_pipes}. Saved to {save_path}")


# demo loop without learning
def play_dqn(config, load_path: str = "checkpoints/dqn.pt", episodes: int = 5):
    """
    Run a few visual episodes with a trained agent (pure greedy; no training).
    """
    import torch

    env = FlappyEnv(config, render=True)
    agent = DQNAgent()
    agent.load(load_path)

    for ep in range(1, episodes + 1):
        s = env.reset()
        total = 0.0
        while True:
            # Greedy action: choose the highest Q-value (no exploration)
            with torch.no_grad():
                import torch as _t
                q = agent.q(_t.tensor(s, dtype=_t.float32).unsqueeze(0))
                a = int(_t.argmax(q, dim=1).item())

            s, r, done, info = env.step(a)
            total += r
            if done:
                print(f"[DQN-PLAY] Ep {ep}  pipes={info.get('score',0)}  reward={total:.2f}")
                break