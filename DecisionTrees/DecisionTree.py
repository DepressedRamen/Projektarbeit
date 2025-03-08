from abc import ABC, abstractmethod
import random
from math import sqrt
from copy import deepcopy
import numpy
#Parent class for the decision tree models

class DecisionTree(ABC):
    def __init__(self, root=None, max_depth=13, intervals=256, min_samples_split=20, random_feature_sampling=False, prune_alpha=0.0):
        """
        Constructor of the decision tree.

        Parameters
        ----------
            root (Node) : Root node of the tree.
            max_depth (int) : Maximum depth of the tree.
            intervals (int) : Number of intervals for numerical features.
            min_samples_split (int) : Minimum number of samples for a leaf node.
            random_feature_sampling (bool) : If True, the tree samples a random subset of features for each split.
            prune_alpha (float) : Alpha value for pruning the tree.
            
        Raises
        ------
            ValueError : If any of the input parameters do not meet their respective constraints.
        """
        # Check the input parameters
        if max_depth < 1:
            raise ValueError("The maximum depth must be at least 1")
        if intervals is not None and intervals < 1:
            raise ValueError("The number of intervals must be at least 1")
        if min_samples_split < 2:
            raise ValueError("The minimum number of samples for a leaf node must be at least 1")
        if prune_alpha < 0:
            raise ValueError("The alpha value for pruning must be at least 0")
        
        self.root = root 
        self.max_depth = max_depth 
        self.intervals = intervals 
        self.min_samples_split = min_samples_split 
        self.random_feature_sampling = random_feature_sampling 
        self.prune_alpha = prune_alpha 
        self.prune_alphas = [] # List to store the  alpha values for pruning the tree
      
    #region Abstract Methods  
    @abstractmethod
    def _contstruct_tree(self, X, y, depth=0):
        """
        Construct the decision tree recursively.

        Parameters
        ----------
            X (array-like) : The input samples.
            y (array-like) : The target values.
            depth (int, default=0) : The current depth of the tree. 
        """
        pass
    
    @abstractmethod
    def _split(self, X, y):
        """
        Determine the best split for the dataset.

        Parameters
        ----------
            X (array-like) : The input samples.
            y (array-like) : The target values.

        Returns
        -------
            split (tuple) : A tuple containing:
                - left_indices: Indices of the samples in the left child.
                - right_indices: Indices of the samples in the right child.
                - best_split_value: The value to split the feature on.
                - best_feature_index: The index of the feature to split on.
        """
        pass
    
    @abstractmethod
    def predict_single_input(self, single_input):
        """
        Return the prediction for a single input.

        Parameters
        ----------
            single_input (Any) : The input data for which the prediction is to be made.

        Returns
        -------
            prediction (Any) : The predicted output for the given input.
        """
        pass
    #endregion
        
    #region Public Methods
    def predict(self, X):
        """
        Return the predictions for the input samples.

        Parameters
        ----------
            X (array-like) : The input samples.

        Returns
        ------- 
            predictions (list) : The predictions for the input samples.
        """
        return [self.predict_single_input(single_input) for single_input in X]

    
    def fit(self, X, y): 
        """
        Fit the decision tree to the dataset.

        Parameters
        ----------
            X (array-like) : Feature matrix of shape (n_samples, n_features).
            
            y (array-like) : Target vector of shape (n_samples).

        """
        
       #verify that the input data is in the correct format
        if not isinstance(X, numpy.ndarray):
            X = self._convert_to_numpy(X)
        if not isinstance(y, numpy.ndarray):
            y = self._convert_to_numpy(y)
        if y.ndim != 1:
            raise ValueError("The target vector y must be one-dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be equal")
        if X.ndim != 2:
            raise ValueError("The feature matrix X must be two-dimensional")
        
        #construct the tree
        self.root = self._contstruct_tree(X, y)
        
        #prune the tree if the alpha value is greater than 0
        if self.prune_alpha > 0:
            self._prune_tree(self.root, self.prune_alpha)
            
        
    def get_pruning_path(self):
        """
        Determine possible pruning alphas for the tree.

        Returns
        -------
            datapoints (list) : A list of alpha values for pruning the tree.
        """
        #copy the current tree and store the alphas 
        current_tree = deepcopy(self.root)
        prune_alphas = [0]
        
        while not current_tree.is_leaf():
            #iterate over all nodes of the tree and store the alpha values and the nodes that are pruned
            candidate_nodes = []
            candidate_alphas = []
            #recompute the cost complexity pruning values of the tree in every iteration
            self._compute_ccp(current_tree)
            #determine the nodes of the pruned tree
            nodes = self._iterate_tree(current_tree)
            #iterate over all nodes of the pruned tree and store all alpha values
            for node in nodes:
                if not node.is_leaf():
                    #store the alpha value and the node that is pruned
                    candidate_alphas.append(node.effective_alpha)
                    candidate_nodes.append(node)

            #determine the index of the node that will be pruned by finding to lowest alpha
            min_index = candidate_alphas.index(min(candidate_alphas))
            candidate_nodes[min_index]._prune_node()
            prune_alphas.append(candidate_alphas[min_index])
        #store the nodes that are pruned and the alpha values
        self.prune_alphas = prune_alphas
    #endregion
    
    #region Private Methods        
    def _determine_indicies(self, X_column, split_value):
        """
        Determine the indices of the left and right child nodes based on a split value.

        Parameters
        ----------
            X_column (list or array-like) : The column of data to split.
        
            split_value (float or int) : The value to split the data on.

        Returns
        -------
            split_indices (tuple) : A tuple containing two lists:
                - left_indices (list): Indices of elements less than the split value.
                - right_indices (list): Indices of elements greater than or equal to the split value.
        """
        left_indices = [index for index,element in enumerate(X_column) if element < split_value]
        right_indices = [index for index,element in enumerate(X_column) if element >= split_value]
        return left_indices, right_indices
    
    def _feature_sample(self, no_features):
        """
        Returns the features that are considered for the split.

        Parameters
        ----------
            no_features (int) : The total number of features available.

        Returns
        -------
            sample (list or range) : A list of randomly sampled feature indices if random_feature_sampling is True,
                       otherwise a range object containing all feature indices.
        """
        
        if self.random_feature_sampling:
            # Determine the sample size based on the square root of the number of features
            sample_size = int(sqrt(no_features))
            # Return a random sample of features when random_feature_sampling is True
            return random.sample(range(no_features), sample_size)
        else:
            # Return all features when random_feature_sampling is False
            return range(no_features)
        
    def _compute_ccp(self, node):
        """
        Compute the cost complexity pruning (CCP) value of the tree recursively.

        Parameters
        ----------
            node (Node) : The current node for which the CCP value is being computed.

        Returns
        -------
            pruning_cost (tuple) : A tuple containing the CCP value of the subtree and the size of the subtree.
        """
        
        #if the node is a leafe abort the recursion
        if node.is_leaf():
            node.ccp_subtree = 0
            node.leaf_size = 1
            return node.ccp_node, 1
        
        #if the node is not a leaf compute the cost complexity pruning value of the subtree recursively
        else:
            ccp_left, size_left = self._compute_ccp(node.left_child)
            ccp_right, size_right = self._compute_ccp(node.right_child)
            node.leaf_size = size_left + size_right
            node.ccp_subtree = ccp_left + ccp_right
            
            #because we use use the sample weigthed impurity we have to divide the ccp value by the number of samples in the parent node 
            effective_ccp_subtree = (float)(node.ccp_subtree) / (float)(node.samples_amount)
            effectiv_ccp_node = (float)(node.ccp_node) / (float)(node.samples_amount)
            
            #compute the effective alpha of the node and store it in the node
            node.effective_alpha = (float)(effectiv_ccp_node - effective_ccp_subtree) / (float)(node.leaf_size - 1)
            return node.ccp_subtree, node.leaf_size
        
    def _iterate_tree(self, node):
        """
        Recursively iterates over all nodes of the tree starting from the given node.

        Parameters
        ----------
            node (Node) : The starting node to begin iteration from.

        Returns
        -------
            nodes (list) : A list of all nodes in the tree.
        """
        nodes = [node]
        if not node.is_leaf():
            nodes.extend(self._iterate_tree(node.left_child))
            nodes.extend(self._iterate_tree(node.right_child))
        return nodes
        
    def _prune_tree(self, root, alpha):
        """
        Prune the tree based on the cost complexity pruning (CCP) value.
        
        Parameters
        ----------
            root (Node) : The root node of the tree to be pruned.
            alpha (float) : The alpha value used for pruning. Nodes with an effective alpha greater than this value will not be pruned.

        """
        while True:
            #If the tree consists of a single node abort the pruning
            if root.is_leaf():
                return
            #recompute the cost complexity pruning values of the tree
            self._compute_ccp(root)
            nodes = self._iterate_tree(root)
            candidate_alphas = []
            candidate_nodes = []
            #iterate through all nodes of the tree and store the alpha values
            for node in nodes:
                if not node.is_leaf():
                    #store the alpha value and the node that is pruned
                    candidate_alphas.append(node.effective_alpha)
                    candidate_nodes.append(node)
                
            #determine the index of the node that is to be pruned      
            min_index = candidate_alphas.index(min(candidate_alphas))
            #if the alpha value is higher than the alpha value for pruning abort
            if candidate_alphas[min_index] > alpha:
                return
            
            #prune the node
            candidate_nodes[min_index]._prune_node()
            
    def _convert_to_numpy(self, data):
        """
        Convert the input data to a numpy array.

        Parameters
        ----------
            data (Any) : The input data to be converted.

        Returns
        -------
            converted_data (numpy.ndarray) : The converted numpy array.

        Raises
        ------
            ValueError : If the input data could not be converted to a numpy array.
        """
        """Convert the input data to a numpy array"""
        try:
            return numpy.array(data)
        except Exception as e:
            raise ValueError("The input data could not be converted to a numpy array") from e
    #endregion