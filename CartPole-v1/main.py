import gym
import numpy as np
from rl import QAgentWithAMemory

# Create the Cart-Pole game environment
env = gym.make('CartPole-v1')

# Number of possible actions
print('Number of possible actions:', env.action_space.n)

actions = [] # actions that the agent selects
rewards = [] # obtained rewards

env.reset()

while True:
    action = env.action_space.sample()  # choose a random action
    state, reward, done, _ = env.step(action) 
    rewards.append(reward)
    actions.append(action)
    if done:
        break

print('Actions:', actions)
print('Rewards:', rewards)

# Network parameters
state_size = 4
action_size = 2
hidden_size = 64               # number of units in each Q-network hidden layer
learning_rate = 0.0001         # Q-network learning rate

with QAgentWithAMemory(__file__, state_size, action_size, learning_rate, hidden_size) as agent:

    agent.learn(env, 350)

    state = env.reset()
    rewards_list = 0
    while True:
        env.render()
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        rewards_list += reward
        if done:
            print("Episode finished after {0} steps".format(rewards_list))
            break