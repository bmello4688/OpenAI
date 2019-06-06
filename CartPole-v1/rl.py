import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque
from base import QLearningLossFunction, Memory, DeepQNetworkSubgraph, NetworkGraph, QAgent

def _get_dqn_graph_definition(_state_size, _action_size, _hidden_size):
    inputs = tf.placeholder(tf.float32, [None, _state_size])
                    
    # ReLU hidden layers
    mfc1 = tf.contrib.layers.fully_connected(inputs, _hidden_size)
    mfc2 = tf.contrib.layers.fully_connected(mfc1, _hidden_size)
    mfc3 = tf.contrib.layers.fully_connected(mfc2, _hidden_size)

    # Linear output layer
    output = tf.contrib.layers.fully_connected(mfc3, _action_size, 
                                                            activation_fn=None)
    return inputs, output

def _get_duel_dqn_graph_definition(_state_size, _action_size, _hidden_size):
    inputs = tf.placeholder(tf.float32, [None, _state_size])
      
    # ReLU hidden layers
    afc1 = tf.contrib.layers.fully_connected(inputs, _hidden_size)
    afc2 = tf.contrib.layers.fully_connected(afc1, _hidden_size)
    afc3 = tf.contrib.layers.fully_connected(afc2, _hidden_size)

    # Linear output layer
    value_output = tf.contrib.layers.fully_connected(afc3, 1, activation_fn=None)

    # ReLU hidden layers
    vfc1 = tf.contrib.layers.fully_connected(inputs, _hidden_size)
    vfc2 = tf.contrib.layers.fully_connected(vfc1, _hidden_size)
    vfc3 = tf.contrib.layers.fully_connected(vfc2, _hidden_size)

    # Linear output layer
    advantage_output = tf.contrib.layers.fully_connected(vfc3, _action_size, activation_fn=None)
    # aggregate output layer
    aggregation_layer = value_output + (advantage_output - tf.reduce_mean(advantage_output, axis=1, keepdims=True))
    return inputs, aggregation_layer

class DuelDoubleDQNetworkGraphWithFixedTarget(NetworkGraph):
    def __init__(self, name, weight_path, state_size, action_size, learning_rate, hidden_size):
        self._state_size = state_size
        self._action_size = action_size
        self._learning_rate = learning_rate
        self._hidden_size = hidden_size
        super().__init__(name, weight_path)
    def _define(self):
        main_name = 'main'
        with tf.variable_scope(main_name):
            main_inputs, main_output = _get_duel_dqn_graph_definition(self._state_size, self._action_size, self._hidden_size)
            self._mainDQN = DeepQNetworkSubgraph(main_name, self, main_inputs, main_output)

        target_name = 'target'
        with tf.variable_scope(target_name):
            target_inputs, target_output = _get_duel_dqn_graph_definition(self._state_size, self._action_size, self._hidden_size)
            self._targetDQN = DeepQNetworkSubgraph(target_name, self, target_inputs, target_output)

        self._update_target_network_op = [t.assign(m) for t, m in zip(self._targetDQN.get_weights(), self._mainDQN.get_weights())]
        self._loss_function = QLearningLossFunction(self, self._mainDQN.input_layer, self._mainDQN.output_layer, self._learning_rate, self._mainDQN.get_weights())
    def get_action(self, state):
        return self._mainDQN.get_action(state)

    def update_target_network(self):
        self.apply_operation(self._update_target_network_op)

    def train_on_experience(self, experiences, gamma):
        states, actions, rewards, next_states = zip(*experiences)

        #single learning
        #td_target = self._targetDQN.get_target_Q_value(rewards, gamma, next_states)

        #double learning
        predicted_next_actions = self._mainDQN.get_action(next_states)
        td_target = self._targetDQN.get_target_Q_value(rewards, gamma, next_states, predicted_next_actions)

        loss, priority = self._loss_function.run(states, actions, td_target)

        return loss, priority

class QAgentWithReplay(QAgent):
    def __init__(self, weights_path, state_size, action_size, learning_rate, hidden_size, graph_results=False):
        self._network = DuelDoubleDQNetworkGraphWithFixedTarget('agent', weights_path, state_size, action_size, learning_rate, hidden_size)
        self.graph_results = graph_results
        return super().__init__(self._network)
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


    def train(self, env, train_episodes = 100):
        # max number of episodes to learn from
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
                
                loss, priority = self._network.train_on_experience(experiences, gamma)

            self._network.update_target_network()

        self._network.save_weights()

        if self.graph_results:
            eps, rews = np.array(rewards_list).T
            smoothed_rews = self.running_mean(rews, 10)
            plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
            plt.plot(eps, rews, color='grey', alpha=0.3)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.show()