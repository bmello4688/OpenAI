import tensorflow as tf
import numpy as np
import os
from collections import deque
from base import QLearningLossFunction, Memory, DeepQNetworkSubgraph, NetworkGraph, Agent

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

    def train_on_memories(self, memories, gamma):
        _, experiences, experience_importances = memories
        states, actions, rewards, next_states = zip(*experiences)

        #single learning
        #td_target = self._targetDQN.get_target_Q_value(rewards, gamma, next_states)

        #double learning
        predicted_next_actions = self._mainDQN.get_action(next_states)
        td_target = self._targetDQN.get_target_Q_value(rewards, gamma, next_states, predicted_next_actions)

        loss, abs_td_error = self._loss_function.run(states, actions, td_target, experience_importances)

        return loss, abs_td_error

class QAgent(Agent):
    def __init__(self, weights_path, state_size, action_size, learning_rate, hidden_size):
        network = DuelDoubleDQNetworkGraphWithFixedTarget('agent', weights_path, state_size, action_size, learning_rate, hidden_size)
        # Memory parameters
        memory_batch_size = 20
        memory_size = 10000            # memory capacity
        return super().__init__(network, action_size, memory_batch_size, memory_size, weights_path, 50)
        