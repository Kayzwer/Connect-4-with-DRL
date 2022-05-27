from Connect4 import Connect4
from train import Agent, Network, Epsilon_Controller, Replay_Buffer
import pickle


env = Connect4()
with open("DDQN Connect 4 Agent1.pickle", "rb") as f:
    agent1: Agent = pickle.load(f)
with open("DDQN Connect 4 Agent2.pickle", "rb") as f:
    agent2: Agent = pickle.load(f)

state = env.reset()
done = False
while not done:
    action1 = agent1.choose_action_test(state, env)
    state, reward1, reward2, done = env.step(action1, 1)
    if done:
        print(env)
        if reward1 == 0:
            print("DRAW")
        else:
            print("Player 1")
        break
    print(env)

    action2 = agent2.choose_action_test(state, env)
    state, reward1, reward2, done = env.step(action2, 1)
    if done:
        print(env)
        if reward2 == 0:
            print("DRAW")
        else:
            print("Player 2")
        break
    print(env)
