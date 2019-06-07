import numpy as np
import os
import pickle

class sum_tree_with_data(object):
    """
    This SumTree code is modified version of Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """
    def __init__(self, capacity):
        self._capacity = capacity # Number of leaf nodes (final nodes) that contains experiences
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self._tree = np.zeros(2 * capacity - 1)
        self._data_pointer = 0
        self._is_at_capacity = False
        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """
        
        # Contains the experiences (so the size of data is capacity)
        self._data = np.zeros(capacity, dtype=object)
    
    
    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """
    def add(self, value, data):
        # Look at what index we want to put the experience
        tree_index = self._data_pointer + self._capacity - 1
        
        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """
        
        # Update data frame
        self._data[self._data_pointer] = data
        
        # Update the leaf
        self.update (tree_index, value)
        
        # Add 1 to data_pointer
        self._data_pointer += 1
        
        if self._data_pointer >= self._capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self._is_at_capacity = True
            self._data_pointer = 0
            
    
    """
    Update the leaf priority score and propagate the change through tree
    """
    def update(self, tree_index, value):
        # Change = new priority score - former priority score
        change = value - self._tree[tree_index]
        self._tree[tree_index] = value
        
        # then propagate the change through tree
        while tree_index != 0:    # this method is faster than the recursive loop in the reference code
            
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
            
                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 
            
            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self._tree[tree_index] += change
    
    
    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """
    def get_leaf(self, value):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0
        
        while True: # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self._tree):
                leaf_index = parent_index
                break
            
            else: # downward search, always search for a higher priority node
                
                if value <= self._tree[left_child_index]:
                    parent_index = left_child_index
                    
                else:
                    value -= self._tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self._capacity + 1

        return leaf_index, self._tree[leaf_index], self._data[data_index]
    
    @property
    def total_sum(self):
        return self._tree[0] # Returns the root node

    @property
    def count(self):
        return self._data_pointer if not self._is_at_capacity else self._capacity

    def get_max_leaf(self):
        return np.max(self._tree[-self._capacity:])

    def get_min_leaf(self):
        return np.min(self._tree[-self._capacity:])

    def load(self, save_path):
        if os.path.isfile(save_path):
            self.__dict__ = pickle.load( open(save_path, "rb" ))
    def save(self, save_path):
        pickle.dump(self.__dict__, open(save_path, "wb" ))

    def serialize(self):
        return self.__dict__
    def deserialize(self, serialized):
        self.__dict__ = serialized