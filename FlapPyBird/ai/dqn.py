import random, math
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# whats stored in the replay buffer (state, action, reward, nextState, done flag)
Transition = namedtuple("Transition", ("s", "a", "r", "s2", "done"))

# neural network that predicts q values for each action
# state_dim = features, action_dim = noflap or flap
class QNet(nn.Module):
    def __init__(self, state_dim=5, action_dim=2):
        super().__init__()
        # executes layers in order: state -> linear -> relu -> linear ...
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), #input -> hidden -> relu
            nn.Linear(128, 128), nn.ReLU(), # hidden -> hidden -> relu
            nn.Linear(128, action_dim) # output layer
        )
    def forward(self, x):
        return self.net(x)

# used to store previous experiences to learn from
class ReplayBuffer:
    def __init__(self, capacity=50_000):
        # drops oldest entries
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        # add transition to buffer 
        self.buf.append(Transition(*args))
    def __len__(self):
        # to call len(buffer)
        return len(self.buf)
    
    def sample(self, batch_size):
        # choose random batch size B
        batch = random.sample(self.buf, batch_size)
        # list-of-transitions to transition-of-lists
        b = Transition(*zip(*batch))
        # states are arrays -> stack to (b, state_dim)
        s = torch.tensor(np.array(b.s), dtype=torch.float32)
        # actions ints -> long tensor, use unsqueeze to match gather's expected shape
        a = torch.tensor(b.a, dtype=torch.long).unsqueeze(1)
        # rewards scalars -> float tensor with shape (B, 1)
        r = torch.tensor(b.r, dtype=torch.float32).unsqueeze(1)
        # same as states
        s2 = torch.tensor(np.array(b.s2), dtype=torch.float32)
        # dones bools -> floats and make shape (B, 1)
        done = torch.tensor(b.done, dtype=torch.float32).unsqueeze(1)
        return s, a, r, s2, done

# used for picking actions, storing transitions in replay, training q network, and saving/loading model weights
class DQNAgent:
    """
    online q networks = model to update each step
    target tqt networks = slow copy to compute stable targets
    ε-greedy = pick random action with probability ε else pick argmax
    """
    def __init__(
        self,
        state_dim=5,
        action_dim=2,
        lr=1e-3, # learning rate
        gamma=0.99, # discount factor for future rewards
        buffer_capacity=50_000,
        eps_start=1.0, # exploration rate start
        eps_end=0.05, # exploration rate end
        eps_decay_steps=100_000, # epsilon steps decay rate  
        target_update_every=1000 # copy q network to tgt network every n steps
    ):
        # use gpu if available else cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # online q network 
        self.q = QNet(state_dim, action_dim).to(self.device)
        # target tgt network
        self.tgt = QNet(state_dim, action_dim).to(self.device)
        # make sure both start identical
        self.tgt.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_update_every = target_update_every
        self.steps = 0
        self.action_dim = action_dim
    
    def _epsilon(self):
        # decays epsilon from start to end over decay steps
        frac = min(1.0, self.steps / float(self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)
    
    def select_action(self, state):
        self.steps += 1
        eps = self._epsilon()
        # explore random actions
        if random.random() < eps:
            return random.randrange(self.action_dim), eps
        
        # else pick action with highest q value
        # dont use gradient
        with torch.nograd():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q(s)
            action = int(torch.argmax(q_values, dim=1).item())
            return action, eps
        
    # learning function
    def optimize(self, batch_size=64, loss_fn="huber"):
        """
        1) sample batch from replay
        2) compute target q vals with target net
        3) compute loss (huber or MSE)
        4) backprop on online q net
        5) periodically sync target net
        returns loss or none
        """

        if len(self.buffer) < batch_size:
            return None
        
        # sample and move to correct device
        s, a, r, s2, done = self.buffer.sample(batch_size)
        s, a, r, s2, done = (
            s.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), done.to(self.device)
        )

        # curr q values are actions taken
        q_values = self.q(s).gather(1, a)

        # compute target if terminal: target = r
        # else target = r + (1.0 - done) * self.gamma * max_next_q
        with torch.no_grad():
            max_next_q = self.tgt(s2).max(1, keepdim=True)[0]
            target = r + (1.0 - done) * self.gamma * max_next_q
        
        # loss between predicted q val action and target
        if loss_fn == "huber":
            loss = nn.SmoothL1Loss()(q_values, target)
        else:
            loss = nn.MSELoss()(q_values, target)
        
        # backprop on q network
        self.opt.zero_grad()
        loss.backward()
        # clip large gradients for stability
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=5.0)
        self.opt.step()

        # copy q network to tgt network
        if self.steps % self.target_update_every == 0:
            self.tgt.load_state_dict(self.q.state_dict())

        return float(loss.item())

    # saves model weights and step counter to later resume/play
    def save(self, path):
        torch.save(
            {
                "q_state_dict": self.q.state_dict(),
                "steps": self.steps,
                "eps_start": self.eps_start,
                "eps_end": self.eps_end,
                "eps_decay_steps": self.eps_decay_steps,
            },
            path,
        )

    # loads model weights
    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.q.load_state_dict(data["q_state_dict"])
        self.tgt.load_state_dict(self.q.state_dict())
        self.steps = int(data.get("steps", 0))
        self.eps_start = float(data.get("eps_start", self.eps_start))
        self.eps_end = float(data.get("eps_end", self.eps_end))
        self.eps_decay_steps = int(data.get("eps_decay_steps", self.eps_decay_steps))
