import os
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import pickle
from algorithms import sum_tree_with_data

class Memory():
    def __init__(self, path_dir, max_size=1000):
        self.PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.PER_b = 0.4  # importance-sampling, from initial value increasing to 1
        self.PER_b_increment_per_sampling = 0.001
        self.absolute_error_upper = 1.  # clipped abs error

        self._save_path = '{}checkpoints/{}'.format(path_dir, 'memory.p')
        self._tree = sum_tree_with_data(max_size)

    def get_number_of_memories(self):
        return self._tree.count
    
    def store(self, experience):
        # Find the max priority
        max_priority = self._tree.get_max_leaf()
        
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        #set new experience as max priority so it is used
        self._tree.add(max_priority, experience)
            
    def get_memories(self, batch_size):
        # Create a sample array that will contains the minibatch
        experiences = []
        memory_count = self.get_number_of_memories()
        if memory_count < batch_size:
            batch_size = memory_count
        
        experienceLocationIndices, experience_importances = np.empty((batch_size,), dtype=np.int32), np.empty((batch_size, 1), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self._tree.total_sum / batch_size       # priority segment
    
        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([self.absolute_error_upper, self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        
        # Calculating the max_weight
        p_min = self._tree.get_min_leaf() / self._tree.total_sum
        max_weight = (p_min * batch_size) ** (-self.PER_b) if p_min > 0 else 1
        
        for i in range(batch_size):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            """
            Experience that correspond to each value is retrieved
            """
            index, priority, experience = self._tree.get_leaf(value)
            
            #P(i)
            importance_probability = priority / self._tree.total_sum
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            importance_sampling_weights = np.power(batch_size * importance_probability, -self.PER_b)

            
            experience_importances[i, 0] = importance_sampling_weights / max_weight
                                   
            experienceLocationIndices[i]= index
            
            experiences.append(experience)
        
        return experienceLocationIndices, experiences, experience_importances
    
    """
    Update the priorities on the tree
    """
    def update_memory_importances(self, experienceLocationIndices, abs_td_errors):
        abs_td_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_td_errors, self.absolute_error_upper)
        priorities = np.power(clipped_errors, self.PER_a)

        for eli, p in zip(experienceLocationIndices, priorities):
            self._tree.update(eli, p)

    def load(self):
        if os.path.isfile(self._save_path):
            self.__dict__ = pickle.load( open(self._save_path, "rb" ))
    def save(self):
        pickle.dump(self.__dict__, open(self._save_path, "wb" ))
        

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
        self.hyper_params_file = '{}checkpoints/hypers.p'.format(self._caller_path)
        self.hyper_params = {}
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
        pickle.dump(self.hyper_params, open(self.hyper_params_file, "wb"))
        print("Model saved")

    def load_weights(self):
        self.saver.restore(self.session, self._weightsFilePath)
        self.hyper_params = pickle.load(open(self.hyper_params_file, "rb"))
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
        
        # Add importance sampling weights place holder
        # so we can correct the bias of prioritized experience replays
        self._importance_sampling_weights = tf.placeholder(tf.float32, [None, 1], name='isw')

        td_error = self._targetQs - Q
        abs_td_error = tf.abs(td_error)
        # loss is modified because of PER bias
        loss = tf.reduce_mean(self._importance_sampling_weights * tf.square(td_error))
        opt = tf.train.AdamOptimizer(self._learning_rate).minimize(loss, var_list=self.weight_list)
        self._operations_to_run = [loss, abs_td_error, opt]

    def run(self, states, actions, td_target, importances_of_experience):
        return_operations = self._networkGraph.apply_operation(self._operations_to_run,
                                feed_dict={self._input_layer: states,
                                           self._targetQs: td_target,
                                           self._actions: actions,
                                           self._importance_sampling_weights: importances_of_experience})
        loss = return_operations[0]
        abs_td_error = return_operations[1]
        return loss, abs_td_error

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
        action = np.argmax(Qs, axis=1)
        if len(action) == 1:
            action = action[0]
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

    def get_target_Q_value(self, reward, gamma, next_state, predicted_next_actions=None):
        if predicted_next_actions is None:#if none chooses max Q-value
            target = reward + gamma * self.get_value_function(next_state)
        else:
            target = reward + gamma * self.get_Q_value(next_state, predicted_next_actions)
        return target

    def get_Q_value(self, states, actions):
        """ Get value function from Q-network """
        Qs = self.get_Q_values(states)

        q_value = np.take_along_axis(Qs, actions.reshape(-1,1), axis=1)

        return q_value.flatten()

    def get_value_function(self, states):
        """ Get value function from Q-network """
        Qs = self.get_Q_values(states)

        value = np.max(Qs, axis=1)
        return value

class Agent(ABC):
    def __init__(self, network, action_size, memory_size, path):
        self._network = network
        self.action_size = action_size
        self._dir_path = self._get_caller_path(path)
        self.memory = Memory(self._dir_path, max_size=memory_size)
        if self.is_trained():
            self._network.load_weights()
            self.memory.load()
    def _get_caller_path(self, path):
        if os.path.isfile(path):
            path = os.path.dirname(os.path.realpath(path)) + '\\'
        return path
    def is_trained(self):
        return self._network.are_weights_saved()
    def choose_random_action(self):
        return np.random.randint(self.action_size)
    def choose_action(self, state):
        return self._network.get_action(state)
    def stop(self):
        self._network.close()
    #for using 'with'
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.stop()

    @abstractmethod
    def learn():
        pass