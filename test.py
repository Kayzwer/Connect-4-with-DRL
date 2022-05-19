import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Connect4 import Connect4


class Network(nn.Module):
    def __init__(self, output_dim: int, learning_rate: float) -> None:
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(42, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.loss = nn.SmoothL1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


if __name__ == "__main__":
    env = Connect4()
    net = Network(env.action_dim, 0.001)

    state = torch.tensor(env.reset(), dtype = torch.float32).unsqueeze(0)
    q_values = net.forward(state)

    print(q_values)
    print(F.softmax(q_values, dim = -1))

    q_values -= q_values.max()
    print(q_values)
    print(F.softmax(q_values, dim = -1).squeeze())
