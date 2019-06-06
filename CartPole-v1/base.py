import os
import tensorflow as tf
import numpy as np
from collections import deque
from abc import ABC, abstractmethod

class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]

class NetworkGraph(ABC):
    def _get_caller_path(self, path):
        if os.path.isfile(path):
            path = os.path.dirname(os.path.realpath(path)) + '\\'
        return path
    def __init__(self, name, weight_path):
        self.name = name
        self.weight_path = weight_path
        self._caller_path = self._get_caller_path(weight_path)
        self._weightsFilePath = '{}checkpoints/{}.ckpt'.format(self._caller_path, name)
        self._graph = tf.Graph()
        # state inputs to the Q-network
        with self._get_context():
            with tf.variable_scope(self.name):
                self._define()
        self._finalize_graph_creation()
    @abstractmethod
    def _define(self):
        pass
    def _get_context(self):
        return self._graph.as_default()
    def _write_graph(self):
        #save graph for tensorboard
        writer = tf.summary.FileWriter(self._caller_path + 'tensorboard')
        writer.add_graph(self._graph)
        writer.flush()
        writer.close()
    
    def get_weights(self, scope=None):
        return self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/{1}'.format(self.name, scope))

    def save_weights(self):
        self.saver.save(self.session, self._weightsFilePath)
        print("Model saved")

    def load_weights(self):
        self.saver.restore(self.session, self._weightsFilePath)
        print("Model restored")

    def are_weights_saved(self):
        return os.path.exists(self._weightsFilePath+ '.meta')

    def _finalize_graph_creation(self):
        with self._get_context():
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
        self.apply_operation = self.session.run

    def close(self):
        self._write_graph()
        self.session.close()

class NetworkSubgraph(ABC):
    def __init__(self, name, networkGraph):
        self._name = name
        self._networkGraph = networkGraph
    def get_weights(self):
        return self._networkGraph.get_weights(self._name)

class LossFunction(ABC):
    def __init__(self, networkGraph):
        self._networkGraph = networkGraph
        with self._networkGraph._get_context():
            with tf.variable_scope('loss'):
                self.define()
    @abstractmethod
    def define(self):
        pass
    @abstractmethod
    def run(self):
        pass

class QLearningLossFunction(LossFunction):
    def __init__(self, networkGraph, input_layer, output_layer, learning_rate, weight_list=None):
        self._input_layer = input_layer
        self._output_layer = output_layer
        self._action_size = output_layer.shape.dims[output_layer.shape.ndims-1].value
        self._learning_rate = learning_rate
        self.weight_list = weight_list
        super().__init__(networkGraph)

    def define(self):
        # One hot encode the actions to later choose the Q-value for the action
        self._actions = tf.placeholder(tf.int32, [None], name='actions')
        one_hot_actions = tf.one_hot(self._actions, self._action_size)
                
        # Target Q values for training
        self._targetQs = tf.placeholder(tf.float32, [None], name='target')

        ### Train with loss (targetQ - Q)^2
        # output has length 2, for two actions. This next line chooses
        # one value from output (per row) according to the one-hot encoded actions.
        Q = tf.reduce_sum(tf.multiply(self._output_layer, one_hot_actions), axis=1)
                
        self.loss = tf.reduce_mean(tf.square(self._targetQs - Q))
        self.opt = tf.train.AdamOptimizer(self._learning_rate).minimize(self.loss, var_list=self.weight_list)
        self.operations_to_run = [self.loss, self.opt]

    def add_operations_to_run(self, operations):
        self.operations_to_run.extend(operations)

    def run(self, states, actions, td_target):
        return_operations = self._networkGraph.apply_operation(self.operations_to_run,
                                feed_dict={self._input_layer: states,
                                           self._targetQs: td_target,
                                           self._actions: actions})
        loss = return_operations[0]
        return loss

class DeepQNetworkSubgraph(NetworkSubgraph):
    def __init__(self, name, networkGraph, input_layer, output_layer):
        """
            input_layer: states
            output_layer: actions(Q-values)
        """
        self.input_layer = input_layer
        self.output_layer = output_layer
        self._output_size = output_layer.shape.dims[output_layer.shape.ndims-1].value
        super().__init__(name, networkGraph)

    def get_action(self, state):
        Qs = self.get_Q_values(state)
        action = np.argmax(Qs)
        return action

    def get_advantage_function(self, rewards, gamma, next_states, states):
        advantage = self.get_target_Q_value(rewards, gamma, next_states) - self.get_value_function(states)
        return advantage

    def get_Q_values(self, states):
        """ Get actions from Q-network """
        states = np.asarray(states)
        if states.ndim < 2:
            states = states.reshape((1, *states.shape))
        feed = {self.input_layer: states}
        Qs = self._networkGraph.apply_operation(self.output_layer, feed_dict=feed)

        # Set target_Qs to 0 for states where episode ends
        episode_ends = (states == np.zeros(states[0].shape)).all(axis=1)
        Qs[episode_ends] = np.zeros((self._output_size, ))

        return Qs

    def get_target_Q_value(self, reward, gamma, next_state):
        target = reward + gamma * self.get_value_function(next_state)
        return target

    def get_value_function(self, states):
        """ Get value function from Q-network """
        Qs = self.get_Q_values(states)

        value = np.max(Qs, axis=1)
        return value