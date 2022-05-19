import pickle
from Connect4 import Connect4

with open("DDQN Connect 4 Agent1.pickle", "rb") as f:
    agent1 = pickle.load(f)
with open("DDQN Connect 4 Agent2.pickle", "rb") as f:
    agent2 = pickle.load(f)

env = Connect4()
state = env.reset()
done = False
while not done:
    action1 = agent1.choose_action_test(state, env)
    state, reward1, reward2, done = env.step(action1, 1)
    if done:
        print(env)
        if reward1 == 0.0:
            print("Draw")
            
        elif reward1 == 1.0:
            print("Player 1 win")
        break
    print(env)

    action2 = agent2.choose_action_test(state, env)
    state, reward1, reward2, done = env.step(action2, -1)
    if done:
        print(env)
        if reward2 == 0.0:
            print("Draw")
        elif reward2 == 1.0:
            print("Player 2 win")
        break
    print(env)