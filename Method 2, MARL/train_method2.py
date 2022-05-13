from Connect4_two_nn_env import Connect4
from typing import Dict
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Network(nn.Module):
    def __init__(self, output_dim: int, learning_rate: float) -> None:
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 4),
            nn.ReLU(),
            nn.Conv2d(64, 64, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 2),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        self.optimizer = optim.RMSprop(self.parameters(), learning_rate)
        self.loss = nn.SmoothL1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


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
        input_dim: tuple
    ) -> None:
        self.ptr = 0
        self.cur_size = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_memory = np.zeros((buffer_size, *input_dim), dtype = np.float32)
        self.action_memory = np.zeros(buffer_size, dtype = np.int8)
        self.reward_memory = np.zeros(buffer_size, dtype = np.float32)
        self.next_state_memory = np.zeros((buffer_size, *input_dim), dtype = np.float32)

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
        input_dim: tuple,
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
        self.network = Network(output_dim, learning_rate)
        self.target_network = Network(output_dim, learning_rate)
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
            action = self.network.forward(torch.tensor(state, dtype = torch.float32).unsqueeze(0)).argmax().item()
        if env._check_valid(action):
            return action
        else:
            while True:
                action = np.random.choice(self.output_dim)
                if env._check_valid(action):
                    return action

    def choose_action_test(self, state: torch.Tensor, env: Connect4) -> int:
        action = self.network.forward(torch.tensor(state, dtype = torch.float32).unsqueeze(0)).argmax().item()
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

        self.network.zero_grad()
        q_pred = self.network.forward(states)[batch_index, actions]
        q_next_argmax_action = self.network.forward(next_states).argmax(dim = 1)
        q_target = rewards + self.gamma * self.target_network.forward(next_states)[batch_index, q_next_argmax_action]
        loss = self.network.loss(q_pred, q_target)
        loss.backward()
        self.network.optimizer.step()
        self.update_count += 1
        if self.update_count % self.c == 0:
            self.update_target_network()
        self.epsilon_controller.decay()
        return loss
    
    def test(self, n_game: int, env: Connect4) -> None:
        for i in range(n_game):
            state = env.reset()
            done = False
            score = 0
            while not done:
                action = self.choose_action_test(state, env)
                state, reward, done = env.step(action)
                print(env)
                score += reward
            print(f"Game: {i + 1}, Score: {score}")


if __name__ == "__main__":
    env = Connect4()
    agent1 = Agent(
        env.state_shape,
        env.action_dim, 
        0.0001, 20000, 512,
        1.0, "0.00001", 0.001,
        0.99, 2048
    )
    agent2 = Agent(
        env.state_shape,
        env.action_dim, 
        0.0001, 20000, 512,
        1.0, "0.00001", 0.001,
        0.99, 2048
    )
    iteration = 1000
    for i in range(iteration):
        state = env.reset()
        loss = 0
        winner = ""
        done = False
        after_first = False
        agent1_transition = []
        agent2_transition = []

        if i % 2 == 0:
            while not done:
                action1 = agent1.choose_action_train(state, env)
                agent1_transition += [state, action1]
                state, reward_1, reward_2, done = env.step(action1, 1)
                if done:
                    agent1_transition += [reward_1, state]
                    agent1.replay_buffer.store(*agent1_transition)
                    agent1_transition.clear()
                    agent2_transition += [reward_2, state]
                    agent2.replay_buffer.store(*agent2_transition)
                    agent2_transition.clear()
                    if reward_1 == 0.0:
                        winner = "Draw"
                    elif reward_1 == 1.0:
                        winner = "Player 1"
                    break
                if after_first:
                    agent2_transition += [reward_2, state]
                    agent2.replay_buffer.store(*agent2_transition)
                    agent2_transition.clear()
            
                action2 = agent2.choose_action_train(state, env)
                agent2_transition += [state, action2]
                state, reward_1, reward_2, done = env.step(action2, -1)
                if done:
                    agent1_transition += [reward_1, state]
                    agent1.replay_buffer.store(*agent1_transition)
                    agent1_transition.clear()
                    agent2_transition += [reward_2, state]
                    agent2.replay_buffer.store(*agent2_transition)
                    agent2_transition.clear()
                    if reward_2 == 0.0:
                        winner = "Draw"
                    elif reward_2 == 1.0:
                        winner = "Player 2"
                    break
                agent1_transition += [reward_1, state]
                agent1.replay_buffer.store(*agent1_transition)
                agent1_transition.clear()
                after_first = True

                if agent1.replay_buffer.is_ready() and agent2.replay_buffer.is_ready():
                    loss += agent1.train()
                    loss += agent2.train()
        else:
            while not done:
                action2 = agent2.choose_action_train(state, env)
                agent2_transition += [state, action2]
                state, reward_1, reward_2, done = env.step(action2, -1)
                if done:
                    agent2_transition += [reward_2, state]
                    agent2.replay_buffer.store(*agent2_transition)
                    agent2_transition.clear()
                    agent1_transition += [reward_1, state]
                    agent1.replay_buffer.store(*agent1_transition)
                    agent1_transition.clear()
                    if reward_2 == 0.0:
                        winner = "Draw"
                    elif reward_2 == 1.0:
                        winner = "Player 2"
                    break
                if after_first:
                    agent1_transition += [reward_1, state]
                    agent1.replay_buffer.store(*agent1_transition)
                    agent1_transition.clear()
            
                action1 = agent1.choose_action_train(state, env)
                agent1_transition += [state, action1]
                state, reward_1, reward_2, done = env.step(action1, 1)
                if done:
                    agent2_transition += [reward_2, state]
                    agent2.replay_buffer.store(*agent2_transition)
                    agent2_transition.clear()
                    agent1_transition += [reward_1, state]
                    agent1.replay_buffer.store(*agent1_transition)
                    agent1_transition.clear()
                    if reward_1 == 0.0:
                        winner = "Draw"
                    elif reward_1 == 1.0:
                        winner = "Player 1"
                    break
                agent2_transition += [reward_2, state]
                agent2.replay_buffer.store(*agent2_transition)
                agent2_transition.clear()
                after_first = True

                if agent1.replay_buffer.is_ready() and agent2.replay_buffer.is_ready():
                    loss += agent1.train()
                    loss += agent2.train()
        
        if (i + 1) % 5 == 0:
            print(f"Iteration: {i + 1}, Winner: {winner}, Total Loss: {loss}")

    with open("CNN DDQN Connect 4 Agent1.pickle", "wb") as f:
        pickle.dump(agent1, f)
    
    with open("CNN DDQN Connect 4 Agent2.pickle", "wb") as f:
        pickle.dump(agent2, f)
