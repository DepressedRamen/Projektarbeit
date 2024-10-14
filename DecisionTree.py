import numpy
import numbers
from collections import Counter
from Node import Node #Node class for the decision tree
#Decision Tree with all necessary functions for a decision tree 

class DecisionTree:
    def __init__(self, root=None, max_depth=13, intervalls=5):
        """Constructor of the decision tree"""
        self.root = root #root node of the tree
        self.max_depth = max_depth #maximum depth of the tree 
        self.intervals = intervalls #number of intervalls for numerical features
        
    def fit(self, X, y): 
        """Fit the decision tree to the dataset"""
        self.root = self._contrstuct_tree(X, y)
        
    #region Private Methods
    def _contrstuct_tree(self, X, y, depth=0):
        """Construct the tree recursively"""
        number_of_labels = len(numpy.unique(y))
        
        #region check if the current node is a leaf node
        #by determing if there is only one label left
        if number_of_labels == 1:
            return Node(value = y[0])
        #by determing the maximum depth is reached
        if depth == self.max_depth:
            label_counter = Counter(y)
            most_common_label = label_counter.most_common(1)[0][0]
            return Node(value = most_common_label)
        #endregion
        
        #split the dataset
        left_indices, right_indices, split_value, feature_index = self._split(X, y)
        
        #create left and right child
        left_child = self._contrstuct_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._contrstuct_tree(X[right_indices], y[right_indices], depth + 1)
        
        #create the current node
        return Node(left_child=left_child, right_child=right_child, split_value=split_value, feature_index=feature_index)
    
    def _split(self, X, y): 
        """Return the best split of a dataset"""
        best_information_gain = None
        best_split_value = None 
        best_feature_index = None
        
        #iterate over all features 
        for feature_index in range(X.shape[1]):
            information_gains = []
            X_column = X[:,feature_index]
            
            if isinstance(X_column[0], numbers.Number) and len(numpy.unique(X_column)) > self.intervals:
                """Feature is numerical and has more than unique values than the specified amount of intervals"""
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
                information_gains.append(self._information_gain(X_column, y,  value))
                
            #determine the best split for the current feature
            best_gain_for_feature = max(information_gains)
            if best_information_gain is None or best_gain_for_feature > best_information_gain:
                best_information_gain = best_gain_for_feature
                best_split_value = split_values[information_gains.index(best_gain_for_feature)]
                best_feature_index = feature_index
        
        #split the dataset according to the best split
        X_column_best_feature = X[:,best_feature_index]
        left_indices = [index for index,element in enumerate(X_column_best_feature) if element < best_split_value]
        right_indices = [index for index,element in enumerate(X_column_best_feature) if element >= best_split_value] 
        
        return left_indices, right_indices, best_split_value, best_feature_index
            
    
    def _information_gain(self, X_column, y, split_value):
        """Return the information gain of a split"""
        parent_entropy = self._entropy(y)
        
        #split the dataset into two parts according to the split value 
        left_indices = [index for index,element in enumerate(X_column) if element < split_value]
        right_indices = [index for index,element in enumerate(X_column) if element >= split_value]
        
        #determine amount of datapoints of the left and right child  as well as the combined amount of datapoints
        left_datapoints_amount = len(left_indices)
        right_datapoints_amount = len(right_indices)
        total_datapoints = left_datapoints_amount + right_datapoints_amount
        
        #calculate the entropy of the left and right child
        left_child_entropy = self._entropy(y[left_indices])
        right_child_entropy = self._entropy(y[right_indices])
        
        #calculate the entropy of the children
        children_entropy = (left_datapoints_amount / total_datapoints) * left_child_entropy + (right_datapoints_amount / total_datapoints) * right_child_entropy
        
        #return the information gain
        return parent_entropy - children_entropy
        
    
    def _entropy(self, y):
        """Return the entropy of a dataset"""
        relative_occurences = self._relative_occurences(y)
        return -sum([ro * numpy.log2(ro) for ro in relative_occurences if ro > 0])
            
    def _relative_occurences(self, y): 
        """Return the manifestations of a dataset"""
        ocurrences = numpy.bincount(y)
        # return the relative occurences of each manifestation of y 
        return ocurrences / len(y)
        
    
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
    #endregion