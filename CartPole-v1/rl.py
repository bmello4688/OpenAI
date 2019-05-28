import tensorflow as tf
import numpy as np
import os
from collections import deque

class QNetwork:
    def __init__(self, name, weight_path, learning_rate=0.01, state_size=4, 
                 action_size=2, hidden_size=10, 
                 ):
        def _get_caller_path(weight_path):
            if os.path.isfile(weight_path):
                weight_path = os.path.dirname(os.path.realpath(weight_path)) + '\\'
            return weight_path
        self._caller_path = _get_caller_path(weight_path)
        self._weightsFilePath = '{}checkpoints/{}.ckpt'.format(self._caller_path, name)
        self.graph = tf.Graph()
        # state inputs to the Q-network
        with self.graph.as_default():
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            
            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)
            self.fc3 = tf.contrib.layers.fully_connected(self.fc2, hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc3, action_size, 
                                                            activation_fn=None)
            
            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)
            
            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            self.session = tf.Session()
            #Initialize variables
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
        
        #save graph for tensorboard
        writer = tf.summary.FileWriter(self._caller_path + 'tensorboard')
        writer.add_graph(self.graph)
        writer.flush()
        writer.close()

    def save_weights(self):
        self.saver.save(self.session, self._weightsFilePath)
        print("Model saved")

    def load_weights(self):
        self.saver.restore(self.session, self._weightsFilePath)
        print("Model restored")

    def are_weights_saved(self):
        return os.path.exists(self._weightsFilePath+ '.meta')

    def get_Qs(self, state):
        # Get action from Q-network
        feed = {self.inputs_: state}
        Qs = self.session.run(self.output, feed_dict=feed)
        return Qs

    def get_action(self, state):
        Qs = self.get_Qs(state.reshape((1, *state.shape)))
        action = np.argmax(Qs)
        return action

    def train_on_batch(self, states, actions, rewards, next_states, gamma=0.99):
        # Train network
        target_Qs = self.get_Qs(next_states)
            
        # Set target_Qs to 0 for states where episode ends
        episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
        target_Qs[episode_ends] = (0, 0)
            
        targets = rewards + gamma * np.max(target_Qs, axis=1)

        loss, _ = self.session.run([self.loss, self.opt],
                                feed_dict={self.inputs_: states,
                                           self.targetQs_: targets,
                                           self.actions_: actions})

        return loss

    def close(self):
        self.session.close()


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

def train_and_save(env, mainQN):

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
                action = mainQN.get_action(state)
            
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
            batch = memory.sample(batch_size)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])
            
            loss = mainQN.train_on_batch(states, actions, rewards, next_states, gamma)

    mainQN.save_weights()
        
    return rewards_list
            