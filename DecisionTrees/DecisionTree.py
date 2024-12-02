from abc import ABC, abstractmethod
import random
from math import sqrt, trunc
#Decision Tree with all necessary functions for a decision tree 

class DecisionTree(ABC):
    def __init__(self, root=None, max_depth=13, intervals=5, min_samples_split=20, random_feature_sampling=False):
        """Constructor of the decision tree"""
        self.root = root #root node of the tree
        self.max_depth = max_depth #maximum depth of the tree 
        self.intervals = intervals #number of intervals for numerical features
        self.min_samples_split = min_samples_split #minimum number of samples for a leaf node
        self.random_feature_sampling = random_feature_sampling #if true the tree samples a random subset of features for each split
      
    #region Abstract Methods  
    @abstractmethod
    def _contrstuct_tree(self, X, y, depth=0):
        """Construct the tree recursively"""
        pass
    
    @abstractmethod
    def _split(self, X, y):
        """Return the best split of a dataset"""
        pass
    
    @abstractmethod
    def predict_single_input(self, single_input):
        """Return the prediction for a single input"""
        pass
    #endregion
        
    #region Public Methods
    def predict(self, X):
        """Return the predictions for the dataset X"""
        return [self.predict_single_input(single_input) for single_input in X]

    
    def fit(self, X, y): 
        """Fit the decision tree to the dataset"""
        self.root = self._contrstuct_tree(X, y)
    #endregion
    
    #region Private Methods        
    def _determine_indecies(self, X_column, split_value):
        """Return the indices of the left and right child"""
        left_indices = [index for index,element in enumerate(X_column) if element < split_value]
        right_indices = [index for index,element in enumerate(X_column) if element >= split_value]
        return left_indices, right_indices
    
    def _feature_sample(self, no_features):
        """Returns the features that are considered for the split"""  
        if self.random_feature_sampling:
            return random.sample(range(no_features), trunc(sqrt(no_features)))
        else:
            return range(no_features)
    #endregion
    
