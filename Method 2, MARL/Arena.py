from Connect4_two_nn_env import Connect4
from train_method2 import Agent, Network, Replay_Buffer, Epsilon_Controller
import pickle


if __name__ == "__main__":
    with open("/home/kayzwer/Project/Reinforcement Learning/Connect 4/Method 2, MARL/Dueling DDQN Connect 4 Agent1.pickle", "rb") as f:
        agent1 = pickle.load(f)
    
    with open("/home/kayzwer/Project/Reinforcement Learning/Connect 4/Method 2, MARL/Dueling DDQN Connect 4 Agent2.pickle", "rb") as f:
        agent2 = pickle.load(f)
    
    env = Connect4()
    state = env.reset()
    done = False
    winner = ""
    while not done:
        action1 = agent1.choose_action_test(state, env)
        state, reward_1, reward_2, done = env.step(action1, 1)
        if done:
            print(env)
            if reward_1 == 1.0:
                print("Player 1 win")
            elif reward_1 == 0.0:
                print("Draw")
            break
        print(env)

        action2 = agent2.choose_action_test(state, env)
        state, reward_1, reward_2, done = env.step(action2, -1)
        if done:
            print(env)
            if reward_2 == 1.0:
                print("Player 2 win")
            elif reward_2 == 0.0:
                print("Draw")
            break
        print(env)
