from DecisionTrees.DecisionTree import DecisionTree 
from Node import Node
from numbers import Number
import numpy 
#Class for regression problems using decision trees

class RegressionTree(DecisionTree): #inherit from the DecisionTree class
    #region Abstract Methods
    def _contstruct_tree(self, X, y, depth=0):
        """
        Construct the regression tree recursively.
        
        Parameters
        ----------
            X (numpy.ndarray) : The input features of the dataset.
            y (numpy.ndarray) : The target values of the dataset.
            depth (int) : The current depth of the tree (default is 0).
        
        Returns
        -------
            Node (node) : The constructed node of the regression tree.
        """

        #calculate the cost complexity pruning value of the node
        samples_amount = len(y)
        node_value = numpy.mean(y)
        #multiply the rss of the node with the number of samples and store it in the ccp_node attribute
        #multiplication with the samples is done to later determine the sample weighted purity of the node/subtree
        ccp_node = samples_amount * numpy.sum((y - node_value)**2)
        
        #check if the current node is a leaf node
        #by determing if the maximum depth is reached or the minimum amount of samples is reached
        if (depth == self.max_depth) or (len(y) <= self.min_samples_split):
            #return a leaf node with the average of the labels as value
            if len(y) == 0:
                return Node(value = 0, ccp_node=ccp_node, samples_amount=samples_amount)
            else:
                average = numpy.mean(y)
                return Node(value = average, ccp_node=ccp_node, samples_amount=samples_amount)
        #by determining if the node is pure  
        if numpy.allclose(y, y[0]):
            #return the leaf node by taking the first label as value as the dataset is pure
            return Node(value = y[0], ccp_node=ccp_node, samples_amount=samples_amount)
        
        #determine the features that are used for the split
        feature_indeces = self._feature_sample(X.shape[1])
        
        #split the dataset
        left_indices, right_indices, split_value, feature_index = self._split(X, y, feature_indeces)
        
        #check if the best split is valid
        if left_indices is None or right_indices is None: 
            return Node(value = node_value, ccp_node=ccp_node, samples_amount=samples_amount)
        
        #create left and right child
        left_child = self._contstruct_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._contstruct_tree(X[right_indices], y[right_indices], depth + 1)
        
        #create the current node
        return Node(left_child=left_child, 
                    right_child=right_child, 
                    split_value=split_value, 
                    feature_index=feature_index, 
                    ccp_node=ccp_node, 
                    value=node_value, 
                    samples_amount=samples_amount)
        
    def _split(self, X, y, feature_indices): 
        """
        Return the best split of a dataset using prefix sums for optimization.
        
        Parameters
        ----------
            X (numpy.ndarray) : The feature matrix of shape.
            y (numpy.ndarray) : The target values of shape.
            feature_indices (list) : List of feature indices to consider for splitting.
        
        Returns
        -------
            best_split (tuple) : A tuple containing:
                - left_indices (numpy.ndarray): Indices of the samples in the left split.
                - right_indices (numpy.ndarray): Indices of the samples in the right split.
                - best_split_value (float): The value of the best split.
                - best_feature_index (int): The index of the feature used for the best split.
        """

        #initialize the best mean squared error, the best split value and the best feature index
        best_mean_squared_error = float('inf')
        best_split_value = None
        best_feature_index = None
        
        #iterate over all features 
        for feature_index in feature_indices:
            #get all values of the current feature
            X_column = X[:, feature_index]
            #sort the datapoints by the current feature
            sorted_indices = numpy.argsort(X_column)
            X_column = X_column[sorted_indices]
            y_sorted = y[sorted_indices]
            
            #calculate prefix sums for y and y^2
            prefix_sum_y = numpy.cumsum(y_sorted)
            prefix_sum_y2 = numpy.cumsum(y_sorted ** 2)
            
           
            num_splits = len(numpy.unique(X_column)) - 1
            if self.intervals is None or num_splits <= self.intervals or not isinstance(X_column[0], Number):
                split_indices = range(1, len(X_column))
            else:
                split_indices = numpy.linspace(1, len(X_column) - 1, self.intervals, dtype=int)

            #iterate over all possible split points to find the best split for the current feature
            new_split_value, new_feature_index, new_mse = self._calculate_best_split(y, best_mean_squared_error, feature_index, X_column, prefix_sum_y, prefix_sum_y2, split_indices)
            if new_split_value is not None:
                best_split_value = new_split_value
                best_feature_index = new_feature_index
                best_mean_squared_error = new_mse     
        
        #split the dataset according to the best split
        if best_split_value is None:
            return None, None, None, None
        X_column_best_feature = X[:, best_feature_index]
        left_indices = numpy.where(X_column_best_feature < best_split_value)[0]
        right_indices = numpy.where(X_column_best_feature >= best_split_value)[0]
        
        return left_indices, right_indices, best_split_value, best_feature_index
    
    def predict_single_input(self, single_input):
        """
        Predict the output for a single input using the regression tree.

        Parameters
        ----------
            single_input (list): The input features for which the prediction is to be made.

        Returns
        -------
            prediction (float) : The predicted value for the given input.

        """

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
    def _traverse(self, node, single_input, node_id = 0):
        """
        Recursively traverse the tree to calculate the node id of a leaf.
        
        Parameters
        ----------
            node (Node) : The current node in the tree.
            single_input (list) : A single input data point.
            node_id (int, optional) : The id of the current node. Defaults to 0.
        
        Returns
        -------
            node_id (int) : The node id of the leaf node where the input data point ends up.
        """

        #check if the current node is a leaf and return the node id if it is
        if node.is_leaf(): 
            return node_id
        
        #get the value of the feature of the current node
        feature_value = single_input[node.feature_index]
        
        if (isinstance(feature_value, Number) and feature_value < node.split_value) or \
               (not isinstance(feature_value, Number) and feature_value == node.split_value):
            #go to the left child if the feature value is smaller than the split value
            return self._traverse(node.left_child, single_input, node_id * 2 + 1)
        else:
            #go to the right child if the feature value is greater or equal to the split value
            return self._traverse(node.right_child, single_input, node_id * 2 + 2)
        
    def _calculate_best_split(self, y, best_mean_squared_error, feature_index, X_column, prefix_sum_y, prefix_sum_y2, split_indices):
        """
        Calculate the best split for a given feature in a regression tree.
        Parameters
        ----------
            y (array-like) : The target values.
            best_mean_squared_error (float) : The current best mean squared error.
            feature_index (int) : The index of the feature being evaluated.
            X_column (array-like) : The values of the feature being evaluated.
            prefix_sum_y (array-like) : The prefix sum of the target values.
            prefix_sum_y2 (array-like) : The prefix sum of the squared target values.
            split_indices (array-like) : The indices at which to evaluate potential splits.
        
        Returns
        -------
            best_split (tuple) : A tuple containing the best split value, the best feature index and the best mse.
        """
        #initialize the best split values
        best_split_value = None 
        best_feature_index = None
        
        for i in split_indices:
            if X_column[i] == X_column[i - 1]:
                continue

            #retrieve all necessary values for the mse calculation
            left_count = i
            right_count = len(y) - i
                
            left_sum_y = prefix_sum_y[i - 1]
            right_sum_y = prefix_sum_y[-1] - left_sum_y
                
            left_sum_y2 = prefix_sum_y2[i - 1]
            right_sum_y2 = prefix_sum_y2[-1] - left_sum_y2
                
            left_mean_y = left_sum_y / left_count
            right_mean_y = right_sum_y / right_count
                
            #calculate the mean squared error of the split
            left_mse = left_sum_y2 - 2 * left_mean_y * left_sum_y + left_count * (left_mean_y ** 2)
            right_mse = right_sum_y2 - 2 * right_mean_y * right_sum_y + right_count * (right_mean_y ** 2)
                
            total_mse = (left_mse + right_mse) / len(y)
                
            #update the best split if the current split is better
            if total_mse < best_mean_squared_error:
                best_mean_squared_error = total_mse
                if isinstance(X_column[0], Number):
                    best_split_value = (X_column[i] + X_column[i - 1]) / 2
                else:
                    best_split_value = X_column[i]
                best_feature_index = feature_index
                
        return best_split_value, best_feature_index, best_mean_squared_error
    #endregion
    
    #region Public Methods
    def get_regions(self, X): 
        """
        Retrieves the IDs of the terminal regions of the tree for each input sample.
        
        Parameters
        ----------
            X (numpy.ndarray) : The input data for which to retrieve the terminal region IDs. 

        Returns
        -------
            ids (numpy.ndarray) : An array of integers where each element is the ID of the terminal region 
                        corresponding to the respective input sample.
        """
        '''Retrieves the id's of the terminal regions of the tree'''
        
        #initialize the array to store the region indices
        region_indices = numpy.zeros(X.shape[0], dtype=int)
        
        #traverse the tree for each input
        for i, single_input in enumerate(X): 
            region_indices[i] = self._traverse(self.root, single_input) 
            
        return region_indices
    #endregion
