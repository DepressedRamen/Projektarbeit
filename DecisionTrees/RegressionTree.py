from DecisionTrees.DecisionTree import DecisionTree #import the DecisionTree class
from Node import Node #import the Node class
from numbers import Number
import numpy 


class RegressionTree(DecisionTree): #inherit from the DecisionTree class
    #region Implement Abstract Methods
    def _contrstuct_tree(self, X, y, depth=0):
        """Construct the tree recursively"""
        #calculate the cost complexity pruning value of the node
        n = len(y)
        node_value = numpy.mean(y)
        #multiply the rss of the node with the number of samples and store it in the ccp_node attribute
        #multiplication with the samples is done to later determine the sample weighted purity of the node/subtree
        ccp_node = n * numpy.sum((y - node_value)**2)
        
        #check if the current node is a leaf node
        #by determing if the maximum depth is reached or the minimum amount of samples is reached
        if (depth == self.max_depth) or (len(y) <= self.min_samples_split):
            #return a leaf node with the average of the labels as value
            if len(y) == 0:
                return Node(value = 0, ccp_node=ccp_node, n=n)
            else:
                average = numpy.mean(y)
                return Node(value = average, ccp_node=ccp_node, n=n)
        #by determining if the node is pure  
        if numpy.allclose(y, y[0]):
            #return the leaf node by taking the first label as value as the dataset is pure
            return Node(value = y[0], ccp_node=ccp_node, n=n)
        
        #determine the features that are used for the split
        feature_indeces = self._feature_sample(X.shape[1])
        
        #split the dataset
        left_indices, right_indices, split_value, feature_index = self._split(X, y, feature_indeces)
        
        #create left and right child
        left_child = self._contrstuct_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._contrstuct_tree(X[right_indices], y[right_indices], depth + 1)
        
        #create the current node
        return Node(left_child=left_child, 
                    right_child=right_child, 
                    split_value=split_value, 
                    feature_index=feature_index, 
                    ccp_node=ccp_node, 
                    value=node_value, 
                    n=n)
    
    def _split(self, X, y, feature_indices): 
        """Return the best split of a dataset"""
        #initialize variables for the best split
        best_mean_squared_error = None
        best_split_value = None 
        best_feature_index = None
        
        #iterate over all features 
        for feature_index in feature_indices:
            #initialize a list to store the information gains of the current feature
            mean_squared_errors = []
            #get all values of the current feature
            X_column = X[:,feature_index]
            
            #Feature is numerical and has more than unique values than the specified amount of intervals
            if isinstance(X_column[0], Number) and len(numpy.unique(X_column)) > self.intervals:
                #determine the minimum and maximum value of the current feature
                min_value = min(X_column)
                max_value = max(X_column)
                
                #determine the split values for the current feature
                split_values = numpy.linspace(min_value, max_value, self.intervals+1) #determine the split values including the minimum and maximum
                split_values = split_values[1:-1] #remove the minimum and maximum from the split_values   
            else: 
                """Feature is categorical or has less unique numerical values than the specified amount of intervals"""
                #determine all possible split values for the current feature 
                split_values = numpy.unique(X_column)
                
            for value in split_values: 
                #calculate the information gain of the current split
                mean_squared_errors.append(self._mean_squared_error(X_column, y,  value))
                
            #determine the best split for the current feature
            best_mse_for_feature = min(mean_squared_errors)
            if best_mean_squared_error is None or best_mse_for_feature < best_mean_squared_error:
                #check if the current feature is the best split so far
                best_mean_squared_error = best_mse_for_feature
                best_split_value = split_values[mean_squared_errors.index(best_mse_for_feature)]
                best_feature_index = feature_index
        
        #split the dataset according to the best split
        X_column_best_feature = X[:,best_feature_index]
        left_indices, right_indices = self._determine_indecies(X_column_best_feature, best_split_value)
        
        return left_indices, right_indices, best_split_value, best_feature_index
    
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
    #endregion
    
    #region Private Methods
    def _mean_squared_error(self, X_column, y, split_value):
        """Return the mean squared error of a split"""
        #determine the indices of the left and right child
        left_indices, right_indices = self._determine_indecies(X_column, split_value)
        
        len_left = len(left_indices)
        len_right = len(right_indices)
        
         #check if the left or right child is empty
        if len_left == 0 or len_right == 0:
            return float('inf')
        
        #determine the labels of the left and right child
        y_left = y[left_indices]
        y_right = y[right_indices]
        
        #calculate the mean of the labels of the left and right child
        mean_left = numpy.mean(y_left)
        mean_right = numpy.mean(y_right)

        #calculate the residuals of the left and right child
        mse_left = numpy.array((y_left - mean_left)**2)
        mse_right = numpy.array((y_right - mean_right)**2)

        #calculate the mean squared error of the split weighted by the number of samples
        len_total = len_left + len_right
        mean_squared_error = (len_left/len_total) * numpy.sum(mse_left) + (len_right/len_total) * numpy.sum(mse_right)
        
        return mean_squared_error

    
    def _traverse(self, node, single_input, node_id = 0):
        '''Recursively traverse the tree to calculate the node id of a leaf'''
        if node.is_leaf(): 
            return node_id
        
        feature_value = single_input[node.feature_index]
        
        if (isinstance(feature_value, Number) and feature_value < node.split_value) or \
               (not isinstance(feature_value, Number) and feature_value == node.split_value):
            return self._traverse(node.left_child, single_input, node_id * 2 + 1)
        else:
            return self._traverse(node.right_child, single_input, node_id * 2 + 2)
    
    def get_regions(self, X): 
        '''Retrieves the id's of the terminal regions of the tree'''
        
        region_indices = numpy.zeros(X.shape[0], dtype=int)
        
        
        for i, single_input in enumerate(X): 
            region_indices[i] = self._traverse(self.root, single_input) 
            
        return region_indices
#endregion
