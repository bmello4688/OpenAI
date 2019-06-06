import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque
from base import QLearningLossFunction, Memory, DeepQNetworkSubgraph, NetworkGraph, QAgent

class DoubleDQNetworkGraph(NetworkGraph):
    def __init__(self, name, weight_path, state_size, action_size, learning_rate, hidden_size):
        self._state_size = state_size
        self._action_size = action_size
        self._learning_rate = learning_rate
        self._hidden_size = hidden_size
        super().__init__(name, weight_path)
    def _define(self):
        main_name = 'main'
        with tf.variable_scope(main_name):
            main_inputs = tf.placeholder(tf.float32, [None, self._state_size])
                    
            # ReLU hidden layers
            mfc1 = tf.contrib.layers.fully_connected(main_inputs, self._hidden_size)
            mfc2 = tf.contrib.layers.fully_connected(mfc1, self._hidden_size)
            mfc3 = tf.contrib.layers.fully_connected(mfc2, self._hidden_size)

            # Linear output layer
            main_output = tf.contrib.layers.fully_connected(mfc3, self._action_size, 
                                                                    activation_fn=None)
            self._mainDQN = DeepQNetworkSubgraph(main_name, self, main_inputs, main_output)

        target_name = 'target'
        with tf.variable_scope(target_name):
            target_inputs = tf.placeholder(tf.float32, [None, self._state_size])
                    
            # ReLU hidden layers
            tfc1 = tf.contrib.layers.fully_connected(target_inputs, self._hidden_size)
            tfc2 = tf.contrib.layers.fully_connected(tfc1, self._hidden_size)
            tfc3 = tf.contrib.layers.fully_connected(tfc2, self._hidden_size)

            # Linear output layer
            target_output = tf.contrib.layers.fully_connected(tfc3, self._action_size, 
                                                                    activation_fn=None)
            self._targetDQN = DeepQNetworkSubgraph(target_name, self, target_inputs, target_output)

        copy_to_target = [t.assign(m) for t, m in zip(self._targetDQN.get_weights(), self._mainDQN.get_weights())]
        self._loss_function = QLearningLossFunction(self, self._mainDQN.input_layer, self._mainDQN.output_layer, self._learning_rate, self._mainDQN.get_weights())
        self._loss_function.add_operations_to_run([copy_to_target])
    def get_action(self, state):
        return self._mainDQN.get_action(state)

    def train_on_experience(self, experiences, gamma):
        states, actions, rewards, next_states = zip(*experiences)

        td_target = self._targetDQN.get_target_Q_value(rewards, gamma, next_states)

        loss, priority = self._loss_function.run(states, actions, td_target)

        return loss, priority

class DuelingDQNetworkGraph(NetworkGraph):
    def __init__(self, name, weight_path, state_size, action_size, learning_rate, hidden_size):
        self._state_size = state_size
        self._action_size = action_size
        self._learning_rate = learning_rate
        self._hidden_size = hidden_size
        super().__init__(name, weight_path)
    def _define(self):
        self.inputs = tf.placeholder(tf.float32, [None, self._state_size], name='inputs')
                
        # ReLU hidden layers
        self.afc1 = tf.contrib.layers.fully_connected(self.inputs, self._hidden_size)
        self.afc2 = tf.contrib.layers.fully_connected(self.afc1, self._hidden_size)
        self.afc3 = tf.contrib.layers.fully_connected(self.afc2, self._hidden_size)

        # Linear output layer
        self.value_output = tf.contrib.layers.fully_connected(self.afc3, 1, 
                                                                activation_fn=None)

        # ReLU hidden layers
        self.vfc1 = tf.contrib.layers.fully_connected(self.inputs, self._hidden_size)
        self.vfc2 = tf.contrib.layers.fully_connected(self.vfc1, self._hidden_size)
        self.vfc3 = tf.contrib.layers.fully_connected(self.vfc2, self._hidden_size)

        # Linear output layer
        self.advantage_output = tf.contrib.layers.fully_connected(self.vfc3, self._action_size, 
                                                                activation_fn=None)
        # aggregate output layer
        self.aggregation_layer = self.value_output + (self.advantage_output - tf.reduce_mean(self.advantage_output, axis=1, keepdims=True))           

        self.duelDQN = DeepQNetworkSubgraph('duel', self, self.inputs, self.aggregation_layer)
        self.loss_function = QLearningLossFunction(self, self.inputs, self.aggregation_layer, self._learning_rate)
    def get_action(self, state):
        action = self.duelDQN.get_action(state)
        return action
    def train_on_experience(self, experiences, gamma):
        states, actions, rewards, next_states = zip(*experiences)

        td_target = self.duelDQN.get_target_Q_value(rewards, gamma, next_states)

        loss, priority = self.loss_function.run(states, actions, td_target)

        return loss, priority

class QAgentWithReplay(QAgent):
    def __init__(self, weights_path, state_size, action_size, learning_rate, hidden_size, graph_results=False):
        self.network = DoubleDQNetworkGraph('agent', weights_path, state_size, action_size, learning_rate, hidden_size)
        self.graph_results = graph_results
        return super().__init__(self.network)
    def _pretrain_memory(self, env, memory, pretrain_length=20):
        state = env.reset()
        # Make a bunch of random actions and store the experiences
        for ii in range(pretrain_length):

            # Make a random action
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            if done:
                # The simulation fails so no next state
                next_state = np.zeros(state.shape)
                # Add experience to memory
                memory.add((state, action, reward, next_state))
            
                # Start new episode
                env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = env.step(env.action_space.sample())
            else:
                # Add experience to memory
                memory.add((state, action, reward, next_state))
                state = next_state

    def _running_mean(self, x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0)) 
            return (cumsum[N:] - cumsum[:-N]) / N 


    def train(self, env):
        train_episodes = 100          # max number of episodes to learn from
        max_steps = 200                # max steps in an episode
        gamma = 0.99                   # future reward discount

        # Exploration parameters
        explore_start = 1.0            # exploration probability at start
        explore_stop = 0.01            # minimum exploration probability 
        decay_rate = 0.0001            # exponential decay rate for exploration prob

        # Memory parameters
        memory_size = 10000            # memory capacity
        batch_size = 20                # experience mini-batch size

        #You are welcome (and encouraged!) to take the time to extend this code to implement 
        # some of the improvements that we discussed in the lesson, to include fixed  𝑄  targets, 
        # double DQNs, prioritized replay, and/or dueling networks.
        # Atari games paper : http://www.davidqiu.com:8888/research/nature14236.pdf.

        memory = Memory(max_size=memory_size)

        self._pretrain_memory(env, memory, batch_size)

        # Initialize the simulation
        env.reset()
        # Take one random step to get the pole and cart moving
        state, reward, done, _ = env.step(env.action_space.sample())

        rewards_list = []
        step = 0
        loss = None
        for ep in range(1, train_episodes):
            total_reward = 0
            t = 0
            while t < max_steps:
                step += 1
                # Uncomment this next line to watch the training
                # env.render() 
                
                # Explore or Exploit
                explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step) 
                if explore_p > np.random.rand():
                    # Make a random action
                    action = env.action_space.sample()
                else:
                    action = self.choose_action(state)
                
                # Take action, get new state and reward
                next_state, reward, done, _ = env.step(action)
        
                total_reward += reward
                
                if done:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)
                    t = max_steps
                    
                    print('Episode: {}'.format(ep),
                        'Total reward: {}'.format(total_reward),
                        'Training loss: {:.4f}'.format(loss),
                        'Explore P: {:.4f}'.format(explore_p))
                    rewards_list.append((ep, total_reward))
                    
                    # Add experience to memory
                    memory.add((state, action, reward, next_state))
                    
                    # Start new episode
                    env.reset()
                    # Take one random step to get the pole and cart moving
                    state, reward, done, _ = env.step(env.action_space.sample())

                else:
                    # Add experience to memory
                    memory.add((state, action, reward, next_state))
                    state = next_state
                    t += 1
                
                # Sample mini-batch from memory
                experiences = memory.sample(batch_size)
                
                loss, priority = self.network.train_on_experience(experiences, gamma)

        self.network.save_weights()

        if self.graph_results:
            eps, rews = np.array(rewards_list).T
            smoothed_rews = self.running_mean(rews, 10)
            plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
            plt.plot(eps, rews, color='grey', alpha=0.3)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.show()