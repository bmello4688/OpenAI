from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.losses import mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.utils import to_categorical
import numpy as np
import os
from collections import deque

def _get_path():
    current_dir = os.path.abspath('.')
    if 'CartPole-v1-keras' in current_dir:
        current_dir = current_dir.replace('CartPole-v1-keras', '')
    path = current_dir + "/CartPole-v1-keras/model-weights/cartpole.h5"
    return path

_weightsFilePath = _get_path()

class QNetwork:
    
    def __init__(self, learning_rate=0.01, state_size=4, 
                 action_size=2, hidden_size=10):

        state_input = Input(shape=(state_size, ))
        actions_input = Input(shape=(action_size, ))
        targetQs_input = Input(shape=(1, ))

        # a layer instance is callable on a tensor, and returns a tensor
        fc1 = Dense(hidden_size, activation='relu')(state_input)
        fc2 = Dense(hidden_size, activation='relu')(fc1)
        action_output = Dense(action_size)(fc2)
        Q = keras.layers.dot(axis=1)([action_output, actions_input])

        self.model = Model(inputs=[state_input, actions_input, targetQs_input], outputs=[action_output, Q])

        self. model.compile(optimizer=Adam(lr=learning_rate),
              loss=mean_squared_error,
              metrics=[mean_absolute_error])

        self.gamma = 0.99                   # future reward discount

    def get_Qs(self, state):
        # Get action from Q-network
        Qs = self.model.predict(state)
        return Qs

    def get_action(self, state):
        action = np.argmax(self.get_Qs(state.reshape((1, *state.shape))))
        return action

    def train(self, states, actions, rewards, next_states):
        # Train network
            #target_Qs = self.get_Qs(next_states)
                
            # Set target_Qs to 0 for states where episode ends
            #episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
            #target_Qs[episode_ends] = (0, 0)
                
            #targets = rewards + self.gamma * np.max(target_Qs, axis=1)
        return self.model.train_on_batch(states, actions)


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

            mainQN.train(states, actions, rewards, next_states)
        
    self.model.save(_weightsFilePath)
    print("Model saved")
    return rewards_list

def load_model():
    self.model = load_model(_weightsFilePath)
    print("Model restored")

def are_weights_saved():
    return os.path.exists(_weightsFilePath+ '.meta')
            