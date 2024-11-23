from abc import ABC, abstractmethod
from numbers import Number
#Decision Tree with all necessary functions for a decision tree 

class DecisionTree(ABC):
    def __init__(self, root=None, max_depth=13, intervals=5, min_samples_split=20):
        """Constructor of the decision tree"""
        self.root = root #root node of the tree
        self.max_depth = max_depth #maximum depth of the tree 
        self.intervals = intervals #number of intervals for numerical features
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


    #region Public Methods
    def predict_single_input(self, single_input):

        """Return the prediction for a single input"""
        node = self.root
        #traverse tree until we reach a leaf
        while not node.is_leaf():
            feature_value = single_input[node.feature_index]
            #check if the feature value is numerical or categorical
            #if it is numerical we compare the value to the split value and go left if it is smaller than the node value 
            #if it is categorical we go left if the value is equal to the node value
            if (isinstance(feature_value, Number) and feature_value < node.split_value) or \
               (not isinstance(feature_value, Number) and feature_value == node.split_value): 
                node = node.left_child
            #otherwise we go to the right child of the node 
            else:
                node = node.right_child
        #return the value of the leaf as a prediction
        return node.value
        

    def predict(self, X):
        """Return the predictions for the dataset X"""
        return [self.predict_single_input(single_input) for single_input in X]

    
    def fit(self, X, y): 
        """Fit the decision tree to the dataset"""
        self.root = self._contrstuct_tree(X, y)
        
    def _determine_indecies(self, X_column, split_value):
        """Return the indices of the left and right child"""
        left_indices = [index for index,element in enumerate(X_column) if element < split_value]
        right_indices = [index for index,element in enumerate(X_column) if element >= split_value]
        return left_indices, right_indices
    #endregion