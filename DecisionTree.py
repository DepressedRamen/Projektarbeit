import numpy
#Decision Tree with all necessary functions for a decision tree 

class DecisionTree:
    def __init__(self, root=None, max_depth=13):
        """Constructor of the decision tree"""
        self.root = root #root node of the tree
        self.max_depth = max_depth #maximum depth of the tree 
        
    #region Private Methods
    def _information_gain(self, y, X_column, split_value):
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
        while not node.isLeaf():
            if single_input[node.feature_count] < node.split_value:
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

    

        
    
    