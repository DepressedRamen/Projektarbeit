import numpy
from abc import ABC, abstractmethod
from Node import Node #Node class for the decision tree
#Decision Tree with all necessary functions for a decision tree 

class DecisionTree(ABC):
    def __init__(self, root=None, max_depth=13, intervalls=5, min_samples_split=20):
        """Constructor of the decision tree"""
        self.root = root #root node of the tree
        self.max_depth = max_depth #maximum depth of the tree 
        self.intervals = intervalls #number of intervalls for numerical features
        self.min_samples_split = min_samples_split #minimum number of samples for a leaf node
      
    #region Abstract Methods  
    @abstractmethod
    def _contrstuct_tree(self, X, y, depth=0):
        """Construct the tree recursively"""
        pass
    
    @abstractmethod
    def _split(self, X, y):
        """Return the best split of a dataset"""
        pass
    #endregion

    #region Private Methods
    def _predict(self, single_input):
        """Return the prediction for a single input"""
        node = self.root
        #traverse tree until we reach a leaf
        while not node.is_leaf():
            if single_input[node.feature_index] < node.split_value:
                node = node.left_child
            else:
                node = node.right_child
        #return the value of the leaf as a prediction
        return node.value
    #endregion
        
    #region Public Methods
    def predict(self, X):
        """Return the predictions for the dataset X"""
        return [self._predict(single_input) for single_input in X]
    
    def fit(self, X, y): 
        """Fit the decision tree to the dataset"""
        self.root = self._contrstuct_tree(X, y)
    #endregion