from DecisionTrees.DecisionTree import DecisionTree #import the DecisionTree class
from collections import Counter
import numpy
from numbers import Number
from Node import Node #import the Node class

class ClassificationTree(DecisionTree): #inherit from the DecisionTree class
    def __init__(self, root=None, max_depth=13, intervals=5, min_samples_split=20, random_feature_sampling=False, soft_voting=False, prune_alpha=0.0):
        """Constructor of the ClassificationTree"""
        super().__init__(root=root, 
                         max_depth=max_depth, 
                         intervals=intervals, 
                         min_samples_split=min_samples_split, 
                         random_feature_sampling=random_feature_sampling,
                         prune_alpha=prune_alpha) #call the constructor of the DecisionTree class
        self.soft_voting = soft_voting #if true the tree uses soft voting for the predictions
    
    #region Implement Abstract Methods
    def _contrstuct_tree(self, X, y, depth=0, id=0):
        """Construct the tree recursively"""
        label_counter = Counter(y)
        most_common_label = label_counter.most_common(1)[0][0]
        n = len(y)
        #use the entropy of the node as the cost complexity pruning value
        #multiplication with the samples is done to later determine the sample weighted purity of the node/subtree
        ccp_node = n * self._entropy(y)

        #determine the number of unique labels in the dataset
        number_of_labels = len(numpy.unique(y))
        
        #check if the current node is a leaf node
        #by determing if the maximum depth is reached or the minimum amount of samples is reached or the node is pure
        if (depth == self.max_depth) or (len(y) <= self.min_samples_split) or number_of_labels == 1:
            return self._create_leaf(y, number_of_labels, ccp_node, n)
        
        #determine the features that are used for the split
        feature_indeces = self._feature_sample(X.shape[1])
        
        #split the dataset
        left_indices, right_indices, split_value, feature_index = self._split(X, y, feature_indeces)
        
        #create left and right child
        left_child = self._contrstuct_tree(X[left_indices], y[left_indices], depth + 1, id*2+1)
        right_child = self._contrstuct_tree(X[right_indices], y[right_indices], depth + 1, id*2+2)
        
        #create the current node
        return Node(left_child=left_child, right_child=right_child, split_value=split_value, feature_index=feature_index, ccp_node=ccp_node, value=most_common_label, n=n)
    
    def _split(self, X, y, feature_indices): 
        """Return the best split of a dataset"""
        #initialize variables for the best split
        best_information_gain = None
        best_split_value = None 
        best_feature_index = None
        
        #iterate over all features 
        for feature_index in feature_indices:
            #initialize a list to store the information gains of the current feature
            information_gains = []
            #get all values of the current feature
            X_column = X[:,feature_index]
            
            #Feature is numerical and has more  unique values than the specified amount of intervals
            if isinstance(X_column[0], Number) and len(numpy.unique(X_column)) > self.intervals:
                #determine the minimum and maximum value of the current feature
                min_value = min(X_column)
                max_value = max(X_column)
                
                #determine the split values for the current feature
                split_values = numpy.linspace(min_value, max_value, self.intervals+1) #determine the split values including the minimum and maximum
                split_values = split_values[1:-1] #remove the minimum and maximum from the split_values   
            #Feature is categorical or has less unique numerical values than the specified amount of intervals
            else: 
                #determine all possible split values for the current feature 
                split_values = numpy.unique(X_column)
            for value in split_values: 
                #calculate the information gain of the current split
                information_gains.append(self._sample_weighted_entropy(X_column, y,  value))
                
            #determine the best split for the current feature
            best_gain_for_feature = min(information_gains)
            #check if the current feature is the best split so far
            if best_information_gain is None or best_gain_for_feature < best_information_gain:
                #update the best split
                best_information_gain = best_gain_for_feature
                best_split_value = split_values[information_gains.index(best_gain_for_feature)]
                best_feature_index = feature_index
        
        #split the dataset according to the best split
        X_column_best_feature = X[:,best_feature_index]
        left_indices, right_indices = self._determine_indecies(X_column_best_feature, best_split_value)
        
        #return the indices of the left and right child as well as the best split value and the best feature index
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
        
        #check if the tree uses soft voting
        if self.soft_voting:
            #return the label with the highest probability
            return max(node.value, key=node.value.get)
        else: 
            #return the value of the leaf as a prediction
            return node.value
    
    #region Private Methods
    def _sample_weighted_entropy(self, X_column, y, split_value):
        """Return the sample weighted entropy of a split"""
        
        #split the dataset into two parts according to the split value 
        left_indices, right_indices = self._determine_indecies(X_column, split_value)
        
        #determine amount of datapoints of the left and right child as well as the combined amount of datapoints
        left_datapoints_amount = len(left_indices)
        right_datapoints_amount = len(right_indices)
        
        #check if the split is valid
        if left_datapoints_amount <1 or right_datapoints_amount <1:
            return float('inf')
        
        #determine the amount of total datapoints
        total_datapoints = left_datapoints_amount + right_datapoints_amount
        
        #calculate the entropy of the left and right child
        left_child_entropy = self._entropy(y[left_indices])
        right_child_entropy = self._entropy(y[right_indices])
        
        #calculate the total entropy of the children
        children_entropy = (left_datapoints_amount / total_datapoints) * left_child_entropy + (right_datapoints_amount / total_datapoints) * right_child_entropy
        
        #return the information gain
        return children_entropy
            
    def _entropy(self, y):
        """Return the entropy of a dataset"""
        relative_occurences = self._relative_occurences(y)
        return -sum([ro * numpy.log2(ro) for ro in relative_occurences if ro > 0])
            
    def _relative_occurences(self, y): 
        """Return the manifestations of a dataset"""
        ocurrences = numpy.bincount(y)
        # return the relative occurences of each manifestation of y 
        return ocurrences / len(y)
    
    def _create_leaf(self, y, number_of_labels, ccp_node, n):
        """Return a leaf node"""
        #check if the tree uses soft voting
        if self.soft_voting:
            #determine the probabilities of each label with the help of the counter class
            label_amounts = Counter(y) 
            values_array = numpy.array(list(label_amounts.values())) 
            key_array = numpy.array(list(label_amounts.keys()))  
            relative_occurences_array = values_array / len(y) 
            probabilities = dict(zip(key_array, relative_occurences_array)) 
            
            #return a leaf node with the probabilities of each label
            return Node(value = probabilities, ccp_node=ccp_node, n=n) 
        #check if there is only one unique label left
        elif number_of_labels == 1:
                return Node(value = y[0], ccp_node=ccp_node, n=n)
        #else determine the most common label
        else: 
            label_counter = Counter(y)
            most_common_label = label_counter.most_common(1)[0][0]
            return Node(value = most_common_label, ccp_node=ccp_node, n=n)
    #endregion
    
    #region Public Methods
    def predict_proba_single_input(self, single_input):
        """Return the probabilities for a single input"""
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
                
        #return the probabilities of the leaf
        return node.value
    
    def predict_proba(self, X):
        """Return the predictions for the dataset X"""
        if not self.soft_voting:
            raise TypeError("The tree does not use soft voting")
        return [self.predict_proba_single_input(single_input) for single_input in X]
    #endregion