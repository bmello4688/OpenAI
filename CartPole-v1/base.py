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

class NetworkSubgraph(ABC):
    def __init__(self, name, networkGraph):
        self._name = name
        self._networkGraph = networkGraph
        with self._networkGraph.get_context():
            self.define()
        self._networkGraph.subgraphs.append(self)
    @abstractmethod
    def define(self):
        pass
    def get_weights(self):
        self._networkGraph.get_weights(self.name)

class LossFunction(ABC):
    def __init__(self, networkGraph):
        self._networkGraph = networkGraph
        with self._networkGraph.get_context():
            self.define()
        self._networkGraph.losses.append(self)
    @abstractmethod
    def define(self):
        pass
    @abstractmethod
    def run(self):
        pass

class RLLossFunction(LossFunction):
    def __init__(self, networkGraph, inputs, output, weight_list, action_size, learning_rate):
        self._action_size = action_size
        self._learning_rate = learning_rate
        super().__init__(networkGraph, inputs, output, weight_list)
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
            Q = tf.reduce_sum(tf.multiply(self._output, one_hot_actions), axis=1)
                
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - Q))
            self.opt = tf.train.AdamOptimizer(self._learning_rate).minimize(self.loss)
    def run(self, states, actions, td_target):
        loss, _ = self.apply_operation([self.loss, self.opt],
                                feed_dict={self._inputs: states,
                                           self.targetQs_: td_target,
                                           self.actions_: actions})

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
        self.subgraphs = []
        self.losses = []
        self._graph = tf.Graph()
        self.define()
        self._finalize_graph_creation()
    @abstractmethod
    def define(self):
        pass
    def get_context(self):
        return self._graph.as_default()
    def _write_graph(self):
        #save graph for tensorboard
        writer = tf.summary.FileWriter(self._caller_path + 'tensorboard')
        writer.add_graph(self._graph)
        writer.flush()
        writer.close()
    
    def get_weights(self, scope=None):
        return self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def save_weights(self):
        self.saver.save(self.session, self._weightsFilePath)
        print("Model saved")

    def load_weights(self):
        self.saver.restore(self.session, self._weightsFilePath)
        print("Model restored")

    def are_weights_saved(self):
        return os.path.exists(self._weightsFilePath+ '.meta')

    def _finalize_graph_creation(self):
        with self.get_context():
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
        self.apply_operation = self.session.run

    def close(self):
        self._write_graph()
        self.session.close()