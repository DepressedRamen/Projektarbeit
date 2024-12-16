from abc import ABC, abstractmethod
import random
from math import sqrt, trunc
from copy import deepcopy
#Decision Tree with all necessary functions for a decision tree 

class DecisionTree(ABC):
    def __init__(self, root=None, max_depth=13, intervals=5, min_samples_split=20, random_feature_sampling=False, prune_alpha=0.0):
        """Constructor of the decision tree"""
        self.root = root #root node of the tree
        self.max_depth = max_depth #maximum depth of the tree 
        self.intervals = intervals #number of intervals for numerical features
        self.min_samples_split = min_samples_split #minimum number of samples for a leaf node
        self.random_feature_sampling = random_feature_sampling #if true the tree samples a random subset of features for each split
        self.prune_alpha = prune_alpha #alpha value for pruning the tree
        self.prune_alphas = [] #list to store the alpha values for pruning the tree
      
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
        if self.prune_alpha > 0:
            self._prune_tree(self.root, self.prune_alpha)
            
        
    def pruning_path(self):
        """Determine possible pruning alphas for the tree"""
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
        
    def _compute_ccp(self, node):
        """Compute the cost complexity pruning value of the tree recursively"""
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
            effective_ccp_subtree = (float)(node.ccp_subtree) / (float)(node.n)
            effectiv_ccp_node = (float)(node.ccp_node) / (float)(node.n)
            #compute the effective alpha of the node and store it in the node
            node.effective_alpha = (float)(effectiv_ccp_node - effective_ccp_subtree) / (float)(node.leaf_size - 1)
            return node.ccp_subtree, node.leaf_size
        
    def _iterate_tree(self, node):
        """Iterate over all nodes of the tree and return the nodes"""
        nodes = [node]
        if not node.is_leaf():
            nodes.extend(self._iterate_tree(node.left_child))
            nodes.extend(self._iterate_tree(node.right_child))
        return nodes
        
    def _prune_tree(self, root, alpha):
        """Prune the tree"""
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
    #endregion