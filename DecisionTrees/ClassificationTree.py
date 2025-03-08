from DecisionTrees.DecisionTree import DecisionTree
from collections import Counter
import numpy
from numbers import Number
from Node import Node
#Class for classification problems using decision trees

class ClassificationTree(DecisionTree): #inherit from the DecisionTree class
    def __init__(self, root=None, max_depth=13, intervals=256, min_samples_split=20, random_feature_sampling=False, soft_voting=False, prune_alpha=0.0):
        """
        Constructor of the ClassificationTree.

        Parameters
        ----------
        root (Node, optional) : The root node of the tree. Defaults to None.
        max_depth (int, optional) : The maximum depth of the tree. Defaults to 13.
        intervals (list, optional) : List of intervals for discretizing continuous features. Defaults to None.
        min_samples_split (int, optional) : The minimum number of samples required to split an internal node. Defaults to 20.
        random_feature_sampling (bool, optional) : If True, features are randomly sampled for each split. Defaults to False.
        soft_voting (bool, optional) : If True, the tree uses soft voting for the predictions. Defaults to False.
        prune_alpha (float, optional) : The complexity parameter used for Minimal Cost-Complexity Pruning. Defaults to 0.0.
        """
        """Constructor of the ClassificationTree"""
        super().__init__(root=root, 
                         max_depth=max_depth, 
                         intervals=intervals, 
                         min_samples_split=min_samples_split, 
                         random_feature_sampling=random_feature_sampling,
                         prune_alpha=prune_alpha) # call the constructor of the DecisionTree class
        self.soft_voting = soft_voting # If true the tree uses soft voting for the predictions
    
    #region Abstract Methods
    def _contstruct_tree(self, X, y, depth=0, id=0):
        """
        Construct the decision tree recursively.
        
        Parameters
        ----------
            X (numpy.ndarray) : The feature matrix.
            y (numpy.ndarray) : The target vector.
            depth (int) : The current depth of the tree. Default is 0.
            id (int) : The identifier for the current node. Default is 0.
            
        Returns
        -------
            Node (node) : The constructed tree node.
        """
        if self.soft_voting:
            node_value = self._get_probabilites(y)
        else:
            label_counter = Counter(y)
            node_value = label_counter.most_common(1)[0][0]
        samples_amount = len(y)
        
        #use the entropy of the node as the cost complexity pruning value
        #multiplication with the samples is done to later determine the sample weighted purity of the node/subtree
        ocurrences = numpy.bincount(y)
        ccp_node = samples_amount * self._entropy_from_counts(ocurrences)

        #determine the number of unique labels in the dataset
        number_of_labels = len(numpy.unique(y))
        
        #check if the current node is a leaf node
        #by determing if the maximum depth is reached or the minimum amount of samples is reached or the node is pure
        if (depth == self.max_depth) or (samples_amount <= self.min_samples_split) or number_of_labels == 1:
            return self._create_leaf(y, number_of_labels, ccp_node, samples_amount)
        
        #determine the features that are used for the split
        feature_indeces = self._feature_sample(X.shape[1])
        
        #split the dataset
        left_indices, right_indices, split_value, feature_index = self._split(X, y, feature_indeces)
        
        #check if the best split would lead to an empty node 
        if len(left_indices) == 0 or len(right_indices) == 0:
            return self._create_leaf(y, number_of_labels, ccp_node, samples_amount)
        
        #create left and right child
        left_child = self._contstruct_tree(X[left_indices], y[left_indices], depth + 1, id*2+1)
        right_child = self._contstruct_tree(X[right_indices], y[right_indices], depth + 1, id*2+2)
        
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
        This method finds the best feature and value to split the dataset in order to minimize the entropy. It uses prefix sums to efficiently calculate the entropy for each possible split.
        
        Parameters:
        -----------
            X (numpy.ndarray) : The feature matrix of shape.
            y (numpy.ndarray) : The target labels of shape.
            feature_indices (list) : The indices of the features to consider for splitting.
        
        Returns:
        --------
            left_indices (numpy.ndarray) : The indices of the samples that go to the left child.
            right_indices (numpy.ndarray) : The indices of the samples that go to the right child.
            best_split_value (float) : The value of the best split.
            best_feature_index (int) : The index of the feature used for the best split.
        """

        #initialize variables for the best split
        best_entropy = float('inf')
        best_split_value = None 
        best_feature_index = None
        
        #iterate over all features 
        for feature_index in feature_indices:
            #sorting the feature values and the labels
            X_column = X[:, feature_index]
            sorted_indices = numpy.argsort(X_column)
            X_column = X_column[sorted_indices]
            y_sorted = y[sorted_indices]
            
            #calculate prefix sums for entropy calculation
            prefix_counts = numpy.zeros((len(numpy.unique(y)), len(y) + 1))
            #each label gets a row in the prefix counts matrix
            for i, label in enumerate(numpy.unique(y)):
                prefix_counts[i, 1:] = numpy.cumsum(y_sorted == label)
            
            #determine the split values for the current feature
            split_values = self._determine_split_value(X_column)
            
            #calculate the best split for the current feature
            new_split_value, new_feature_index, new_entropy = self._calculate_best_split(y, best_entropy, feature_index, X_column, prefix_counts, split_values)
            if new_split_value is not None: 
                best_split_value = new_split_value
                best_feature_index = new_feature_index
                best_entropy = new_entropy
    
        #split the dataset according to the best split
        X_column_best_feature = X[:, best_feature_index]
        left_indices = numpy.where(X_column_best_feature <= best_split_value)[0]
        right_indices = numpy.where(X_column_best_feature > best_split_value)[0]
        
        #return the indices of the left and right child as well as the best split value and the best feature index
        return left_indices, right_indices, best_split_value, best_feature_index
    
    def predict_single_input(self, single_input, prob=False):
        """
        Predict the output for a single input sample.
        
        Parameters
        ----------
            single_input (list or array-like) : The input sample for which the prediction is to be made.
            prob (bool, optional) : If True, return the probability distribution of the classes. 
                                If False, return the predicted class label. Default is False.
        
        Returns
        -------
            prediction (int or dict) : The predicted class label or the probability distribution of the classes.
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
        
        #check if the tree uses soft voting and the predict method was called with the prob parameter set to True
        if self.soft_voting and not prob:
            #return the label with the highest probability
            return max(node.value, key=node.value.get)
        else:
            #return the value of the leaf as a prediction
            return node.value
    #endregion 
    
    #region Private Methods
    def _determine_split_value(self, X_column):
        """
        Determine the split values for a feature.
        
        Parameters
        ----------
            X_column (array-like) : The feature column for which to determine split values.
            
        Returns
        -------
            split_value (numpy.ndarray) : An array of split values for the feature.

        """
        
        #Feature is numerical and has more unique values than the specified amount of intervals
        if isinstance(X_column[0], Number) and self.intervals is not None and len(numpy.unique(X_column)) > self.intervals:
            #determine the minimum and maximum value of the current feature
            min_value = min(X_column)
            max_value = max(X_column)
            
            #determine the split values for the current feature
            split_values = numpy.linspace(min_value, max_value, self.intervals + 1)
            #remove the minimum and maximum from the split_values   
            split_values = split_values[1:-1] 
        
        #Feature is categorical or has less unique numerical values than the specified amount of intervals
        else: 
            #determine all possible split values for the current feature 
            split_values = numpy.unique(X_column)
        
        return split_values
    
    def _entropy_from_counts(self, counts):
        """
        Calculate entropy from counts of labels.

        Parameters
        ----------
            counts (numpy.ndarray) : An array of counts of each label.

        Returns
        -------
            entropy (float) : The entropy of the label distribution.
        """

        total = numpy.sum(counts)
        probabilities = counts / total if total > 0 else numpy.zeros_like(counts)
        return -numpy.sum([p * numpy.log2(p) for p in probabilities if p > 0])
    
    def _create_leaf(self, y, number_of_labels, ccp_node, samples_amount):
        """
        Create and return a leaf node for the decision tree.

        Parameters
        ----------
            y (array-like) : The target values for the current node.
            number_of_labels (int) : The number of unique labels in the target values.
            ccp_node (float) : The cost complexity pruning value for the node.
            samples_amount (int) : The number of samples in the current node.

        Returns
        -------
            Node (node) : A leaf node with either the probabilities of each label (if soft voting is used),
                        the single unique label (if only one unique label is left), or the most common label.
        """

        #check if the tree uses soft voting
        if self.soft_voting:
            #determine the probabilities of each label with the help of the counter class
            probabilities = self._get_probabilites(y)
            #return a leaf node with the probabilities of each label
            return Node(value = probabilities, ccp_node=ccp_node, samples_amount=samples_amount)
        #check if there is only one unique label left
        elif number_of_labels == 1:
            return Node(value = y[0], ccp_node=ccp_node, samples_amount=samples_amount)
        #else determine the most common label
        else: 
            label_counter = Counter(y)
            most_common_label = label_counter.most_common(1)[0][0]
            return Node(value = most_common_label, ccp_node=ccp_node, samples_amount=samples_amount)
        
    def _get_probabilites(self, y):
        """
        Calculate the probabilities of each unique label in the given array.

        Parameters
        ----------
            y (array-like) : An array of labels.

        Returns
        -------
            probs (dict) : A dictionary where keys are unique labels and values are their corresponding probabilities.
        """
        label_amounts = Counter(y) 
        values_array = numpy.array(list(label_amounts.values())) 
        key_array = numpy.array(list(label_amounts.keys()))  
        relative_occurences_array = values_array / len(y) 
        return dict(zip(key_array, relative_occurences_array))  
    
    def _calculate_best_split(self, y, best_entropy, feature_index, X_column, prefix_counts, split_values):
        """
        Calculate the best split for a given feature based on entropy.
        
        Parameters
        ----------
            y (array-like) : The target values.
            best_entropy (float) : The current best entropy value to compare against.
            feature_index (int) : The index of the feature being evaluated.
            X_column (array-like) : The values of the feature being evaluated.
            prefix_counts (numpy.ndarray) : The prefix sums of class counts for the feature values.
            split_values (array-like) : The potential split values for the feature.
        
        Returns
        -------
            best_split (tuple): A tuple containing the best split value, the best feature index and the best entropy.
        """
        
        best_split_value = None 
        best_feature_index = None
        for value in split_values: 
            #find the split index
            split_index = numpy.searchsorted(X_column, value, side='right')
                
            #calculate the left and right entropy of the current split using prefix sums
            left_counts = prefix_counts[:, split_index]
            right_counts = prefix_counts[:, -1] - left_counts
            left_entropy = self._entropy_from_counts(left_counts)
            right_entropy = self._entropy_from_counts(right_counts)
                
            #calculate the sample weighted entropy of the split
            left_weight = split_index / len(y)
            right_weight = 1 - left_weight
            entropy = left_weight * left_entropy + right_weight * right_entropy
                
            #check if the current feature is the best split so far
            if entropy < best_entropy:
                    #update the best split
                best_entropy = entropy
                best_split_value = value
                best_feature_index = feature_index
        return best_split_value, best_feature_index, best_entropy
    #endregion
    
    #region Public Methods    
    def predict_proba(self, X):
        """
        Return the probability predictions for the dataset X.

        Parameters
        ----------
            X (array-like) : The input samples.

        Returns
        -------
        predictions (list) : The probability predictions for each input sample.

        Raises
        ------
            TypeError : If the tree does not use soft voting.
        """

        if not self.soft_voting:
            raise TypeError("The tree does not use soft voting")
        return [self.predict_single_input(single_input, True) for single_input in X]
    #endregion