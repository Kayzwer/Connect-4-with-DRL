from typing import Dict
from Connect4 import Connect4
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Network(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float
    ) -> None:
        super(Network, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Mish()
        )
        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, output_dim)
        )
        self.value_layer = nn.Sequential(
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, 1)
        )
        self.optimizer = optim.RMSprop(self.parameters(), learning_rate)
        self.loss = nn.SmoothL1Loss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_layer.forward(x.float())
        advantage = self.advantage_layer(feature)
        value = self.value_layer(feature)
        return value + advantage - advantage.mean(dim = -1, keepdim = True)


class Epsilon_Controller:
    def __init__(
        self,
        init_eps: float,
        eps_dec_rate: str,
        min_eps: float
    ) -> None:
        self.eps = init_eps
        self.eps_dec_rate = eps_dec_rate
        self.min_eps = min_eps
        self._deci_place = self._get_deci_place()
    
    def decay(self) -> None:
        self.eps = round(self.eps - float(self.eps_dec_rate), self._deci_place) if self.eps > self.min_eps else self.min_eps

    def _get_deci_place(self) -> int:
        after_dot = False
        count = 0
        for char in self.eps_dec_rate:
            if char == ".":
                after_dot = True
            if after_dot:
                count += 1
        return count


class Replay_Buffer:
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        input_dim: int
    ) -> None:
        self.ptr = 0
        self.cur_size = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_memory = np.zeros((buffer_size, input_dim), dtype = np.float32)
        self.action_memory = np.zeros(buffer_size, dtype = np.int8)
        self.reward_memory = np.zeros(buffer_size, dtype = np.float32)
        self.next_state_memory = np.zeros((buffer_size, input_dim), dtype = np.float32)

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray
    ) -> None:
        self.state_memory[self.ptr] = state
        self.action_memory[self.ptr] = action
        self.reward_memory[self.ptr] = reward
        self.next_state_memory[self.ptr] = next_state
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)
    
    def sample(self) -> Dict[str, np.ndarray]:
        indexes = np.random.choice(self.cur_size, self.batch_size, False)
        states = self.state_memory[indexes]
        actions = self.action_memory[indexes]
        rewards = self.reward_memory[indexes]
        next_states = self.next_state_memory[indexes]
        return dict(
            states = torch.tensor(states, dtype = torch.float32),
            actions = torch.tensor(actions, dtype = torch.long),
            rewards = torch.tensor(rewards, dtype = torch.float32),
            next_states = torch.tensor(next_states, dtype = torch.float32)
        )
    
    def is_ready(self) -> bool:
        return self.cur_size >= self.batch_size


class Agent:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        init_eps: float,
        eps_dec_rate: str, 
        min_eps: float,
        gamma: float,
        c: int
    ) -> None:
        self.network = Network(input_dim, output_dim, learning_rate)
        self.target_network = Network(input_dim, output_dim, learning_rate)
        self.replay_buffer = Replay_Buffer(buffer_size, batch_size, input_dim)
        self.epsilon_controller = Epsilon_Controller(init_eps, eps_dec_rate, min_eps)
        self.output_dim = output_dim
        self.gamma = gamma
        self.c, self.update_count = c, 0
        self.update_target_network()

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())

    def choose_action_train(self, state: torch.Tensor, env: Connect4) -> int:
        if np.random.random() < self.epsilon_controller.eps:
            action = np.random.choice(self.output_dim)
        else:
            action = self.network.forward(torch.tensor(state)).argmax().item()
        if env._check_valid(action):
            return action
        else:
            while True:
                action = np.random.choice(self.output_dim)
                if env._check_valid(action):
                    return action

    def choose_action_test(self, state: torch.Tensor, env: Connect4) -> int:
        action = self.network.forward(torch.tensor(state)).argmax().item()
        if env._check_valid(action):
            return action
        else:
            while True:
                action = np.random.choice(self.output_dim)
                if env._check_valid(action):
                    return action

    def train(self) -> torch.Tensor:
        batch = self.replay_buffer.sample()
        states = batch.get("states")
        actions = batch.get("actions")
        rewards = batch.get("rewards")
        next_states = batch.get("next_states")
        batch_index = range(self.replay_buffer.batch_size)


        q_pred = self.network.forward(states)[batch_index, actions]
        q_next_argmax_action = self.network.forward(next_states).argmax(dim = 1)
        q_next = rewards + self.gamma * self.target_network.forward(next_states)[batch_index, q_next_argmax_action]
        loss = self.network.loss(q_pred, q_next)
        loss.backward()
        self.network.optimizer.step()
        self.update_count += 1
        if self.update_count % self.c == 0:
            self.update_target_network()
        self.epsilon_controller.decay()
        return loss


if __name__ == "__main__":
    env = Connect4()
    agent = Agent(
        env.state_dim,
        env.action_dim, 
        0.0001, 10000, 512,
        1.0, "0.0001", 0.001,
        0.99, 1024
    )
    iteration = 1000
    for i in range(iteration):
        state = env.reset()
        done = False
        score = 0
        loss = 0
        while not done:
            action = agent.choose_action_train(state, env)
            next_state, reward, done = env.step(action)
            agent.replay_buffer.store(state, action, reward, next_state)
            score += reward
            state = next_state
            if agent.replay_buffer.is_ready():
                loss = agent.train()
        print(f"Iteration: {i + 1}, Epsilon: {agent.epsilon_controller.eps}, Loss: {loss}, Last Game Reward: {score}")
    
    with open("Connect 4 agent.pickle", "wb") as f:
        pickle.dump(agent, f)
