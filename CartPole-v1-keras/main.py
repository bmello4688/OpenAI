import gym
import numpy as np
import matplotlib.pyplot as plt
from rl import QNetwork, train_and_save, are_weights_saved, load_model

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
hidden_size = 64               # number of units in each Q-network hidden layer
learning_rate = 0.0001         # Q-network learning rate

mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate)

# Now train with experiences
with tf.Session() as sess:
    if are_weights_saved():
        load_model(sess)
    else:
        rewards_list = train_and_save(env, sess, mainQN)


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

    rewards_list = []
    while True:
        env.render()
        action = mainQN.get_action(sess, state)
        state, reward, done, _ = env.step(action)
        rewards_list.append(reward)
        if done:
            print("Episode finished after {0} steps".format(len(rewards_list)))
            break