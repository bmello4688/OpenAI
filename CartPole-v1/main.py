import gym
import numpy as np
import matplotlib.pyplot as plt
from rl import DuelingDQNetworkGraph, DoubleDQNetworkGraph, train_and_save

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

network = DoubleDQNetworkGraph('cartpolevduel', __file__, state_size, action_size, learning_rate, hidden_size)

# Now train with experiences
if network.are_weights_saved():
    network.load_weights()
else:
    rewards_list = train_and_save(env, network)


    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / N 

    eps, rews = np.array(rewards_list).T
    smoothed_rews = running_mean(rews, 10)
    plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
    plt.plot(eps, rews, color='grey', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

state = env.reset()
rewards_list = 0
while True:
    env.render()
    action = network.get_action(state)
    state, reward, done, _ = env.step(action)
    rewards_list += reward
    if done:
        print("Episode finished after {0} steps".format(rewards_list))
        break

network.close()