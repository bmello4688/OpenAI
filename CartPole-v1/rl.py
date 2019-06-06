import tensorflow as tf
import numpy as np
import os
from collections import deque
from base import LossFunction, Memory, NetworkGraph, NetworkSubgraph

class DoubleDQNLossFunction(LossFunction):
    def __init__(self, networkGraph, mainDQN, targetDQN, action_size, learning_rate):
        self._mainDQN = mainDQN
        self._targetDQN = targetDQN
        self._action_size = action_size
        self._learning_rate = learning_rate
        super().__init__(networkGraph)
    def define(self):
        with tf.variable_scope('loss'):    
            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, self._action_size)
                
            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            Q = tf.reduce_sum(tf.multiply(self._mainDQN.output, one_hot_actions), axis=1)
            
            td_error = tf.square(self.targetQs_ - Q)
            self.loss = tf.reduce_mean(td_error)
            self.opt = tf.train.AdamOptimizer(self._learning_rate).minimize(self.loss, var_list=self._mainDQN.get_weights())
            self.copy_to_target = [t.assign(m) for t, m in zip(self._targetDQN.get_weights(), self._mainDQN.get_weights())]


    def run(self, states, actions, td_target):
        loss, _, _ = self._networkGraph.apply_operation([self.loss, self.opt, self.copy_to_target],
                                feed_dict={self._mainDQN.inputs: states,
                                           self.targetQs_: td_target,
                                           self.actions_: actions})
        return loss

class DuelingDQNLossFunction(LossFunction):
    def __init__(self, networkGraph, inputs, aggregationLayer, action_size, learning_rate):
        self._aggregationLayer = aggregationLayer
        self._inputs = inputs
        self._action_size = action_size
        self._learning_rate = learning_rate
        super().__init__(networkGraph)
    def define(self):
        with tf.variable_scope('loss'):    
            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, self._action_size)
                
            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            Q = tf.reduce_sum(tf.multiply(self._aggregationLayer, one_hot_actions), axis=1)

            td_error = tf.square(self.targetQs_ - Q)
            self.loss = tf.reduce_mean(td_error)
            self.opt = tf.train.AdamOptimizer(self._learning_rate).minimize(self.loss)


    def run(self, states, actions, td_target):
        feed_dict=dict()
        for k in self._inputs:
            feed_dict[k] = states
        feed_dict[self.targetQs_] = td_target
        feed_dict[self.actions_] = actions
        loss, _ = self._networkGraph.apply_operation([self.loss, self.opt],
                                feed_dict=feed_dict)
        return loss

class DoubleDQNetworkGraph(NetworkGraph):
    def __init__(self, name, weight_path, state_size, action_size, learning_rate, hidden_size):
        self._state_size = state_size
        self._action_size = action_size
        self._learning_rate = learning_rate
        self._hidden_size = hidden_size
        super().__init__(name, weight_path)
    def define(self):
        self.mainDQN = DeepQNetworkSubgraph('DeepQNetwork', self, self._state_size, self._action_size, self._hidden_size)
        self.targetDQN = DeepQNetworkSubgraph('FixedTargetQNetwork', self, self._state_size, self._action_size, self._hidden_size)
        self.lostFunction = DoubleDQNLossFunction(self, self.mainDQN, self.targetDQN, self._action_size, self._learning_rate)
    def get_action(self, state):
        return self.mainDQN.get_action(state)

    def train_on_experience(self, experiences, gamma):
        states, actions, rewards, next_states = zip(*experiences)

        td_target = self.targetDQN.get_Q_value(rewards, gamma, next_states)

        loss = self.lostFunction.run(states, actions, td_target)

        return loss

class DuelingDQNetworkGraph(NetworkGraph):
    def __init__(self, name, weight_path, state_size, action_size, learning_rate, hidden_size):
        self._state_size = state_size
        self._action_size = action_size
        self._learning_rate = learning_rate
        self._hidden_size = hidden_size
        super().__init__(name, weight_path)
    def define(self):
        self.valueDQN = DeepQNetworkSubgraph('Value', self, self._state_size, 1, self._hidden_size)
        self.advantageDQN = DeepQNetworkSubgraph('Advantage', self, self._state_size, self._action_size, self._hidden_size)
        self.inputs = {self.valueDQN.inputs: [], self.advantageDQN.inputs: []}
        self.aggregation_layer = self.valueDQN.output + (self.advantageDQN.output - tf.reduce_mean(self.advantageDQN.output, axis=1, keepdims=True))
        self.lostFunction = DuelingDQNLossFunction(self, self.inputs, self.aggregation_layer, self._action_size, self._learning_rate)
    def get_Q_values(self, states):
        """ Get actions from Q-network """
        states = np.asarray(states)
        if states.ndim < 2:
            states = states.reshape((1, *states.shape))
        feed = {}
        for key in self.inputs:
            feed[key] = states
        Qs = self.apply_operation(self.aggregation_layer, feed_dict=feed)

        # Set target_Qs to 0 for states where episode ends
        episode_ends = (states == np.zeros(states[0].shape)).all(axis=1)
        Qs[episode_ends] = np.zeros((self._action_size, ))

        return Qs
    def get_action(self, state):
        actions = self.get_Q_values(state)
        action = np.argmax(actions)
        return action
    def train_on_experience(self, experiences, gamma):
        states, actions, rewards, next_states = zip(*experiences)

        td_value = self.valueDQN.get_value_function(states)
        td_advantage = self.advantageDQN.get_advantage_function(rewards, gamma, next_states, states)
        td_target = td_value + (td_advantage - np.mean(td_advantage))

        loss = self.lostFunction.run(states, actions, td_target)

        return loss

class DeepQNetworkSubgraph(NetworkSubgraph):
    def __init__(self, name, networkGraph, input_size, 
                 output_size, hidden_size):
        self._input_size = input_size #state_size
        self._output_size = output_size #action_size
        self._hidden_size = hidden_size
        super().__init__(name, networkGraph)
    def define(self):
        # state inputs to the Q-network
        with tf.variable_scope(self._name):
            self.inputs = tf.placeholder(tf.float32, [None, self._input_size], name='inputs')
            
            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs, self._hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, self._hidden_size)
            self.fc3 = tf.contrib.layers.fully_connected(self.fc2, self._hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc3, self._output_size, 
                                                            activation_fn=None)

    def get_weights(self):
        return self._networkGraph.get_weights(self._name)

    def get_action(self, state):
        Qs = self.get_Q_values(state)
        action = np.argmax(Qs)
        return action

    def get_advantage_function(self, rewards, gamma, next_states, states):
        advantage = self.get_Q_value(rewards, gamma, next_states) - self.get_value_function(states)
        return advantage

    def get_Q_values(self, states):
        """ Get actions from Q-network """
        states = np.asarray(states)
        if states.ndim < 2:
            states = states.reshape((1, *states.shape))
        feed = {self.inputs: states}
        Qs = self._networkGraph.apply_operation(self.output, feed_dict=feed)

        # Set target_Qs to 0 for states where episode ends
        episode_ends = (states == np.zeros(states[0].shape)).all(axis=1)
        Qs[episode_ends] = np.zeros((self._output_size, ))

        return Qs

    def get_Q_value(self, reward, gamma, next_state):
        target = reward + gamma * self.get_value_function(next_state)
        return target

    def get_value_function(self, states):
        """ Get value function from Q-network """
        Qs = self.get_Q_values(states)

        value = np.max(Qs, axis=1)
        return value

def _pretrain_memory(env, memory, pretrain_length=20):

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

def train_and_save(env, network):

    train_episodes = 1000          # max number of episodes to learn from
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

    _pretrain_memory(env, memory, batch_size)

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
                action = network.get_action(state)
            
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
            
            loss = network.train_on_experience(experiences, gamma)

    network.save_weights()
        
    return rewards_list
            