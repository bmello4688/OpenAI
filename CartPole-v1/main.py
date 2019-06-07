import gym
import matplotlib.pyplot as plt
import numpy as np
from rl import QAgent

def _running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

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

max_steps = 200                # max steps in an episode

#You are welcome (and encouraged!) to take the time to extend this code to implement 
# some of the improvements that we discussed in the lesson, to include fixed  𝑄  targets, 
# double DQNs, prioritized replay, and/or dueling networks.
# Atari games paper : http://www.davidqiu.com:8888/research/nature14236.pdf.  

with QAgent(__file__, state_size, action_size, learning_rate, hidden_size) as agent:

    train_episodes = 350
    # Start new episode
    env.reset()

    # Take one random step to get the pole and cart moving
    state, reward, done, _ = env.step(env.action_space.sample())

    loss = None
    rewards_list = []
    for ep in range(1, train_episodes):
        total_reward = 0
        t = 0
        while t < max_steps:
            
            # Uncomment this next line to watch the training
            # env.render() 

            # Take action, get new state and reward
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
        
            total_reward += reward
                
            if done:
                # the episode ends so no next state
                next_state = np.zeros(state.shape)
                t = max_steps
                    
                print('Episode: {}'.format(ep),
                        'Total reward: {}'.format(total_reward),
                        'Training loss: {:.4f}'.format(loss),
                        'Explore P: {:.4f}'.format(agent.get_exploration_probability()))
                rewards_list.append((ep, total_reward))
                    
                # Add experience to memory
                agent.store_experience((state, action, reward, next_state))
                agent.update_target()
                    
                # Start new episode
                env.reset()
                    # Take one random step to get the pole and cart moving
                state, reward, done, _ = env.step(env.action_space.sample())

            else:
                # Add experience to memory
                agent.store_experience((state, action, reward, next_state))
                state = next_state
                t += 1
                
            loss = agent.learn_from_experience()

    agent.save()

    if len(rewards_list) > 2:
        eps, rews = np.array(rewards_list).T
        smoothed_rews = _running_mean(rews, 10)
        plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
        plt.plot(eps, rews, color='grey', alpha=0.3)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show()

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