#Decision Tree with all necessary functions for a decision tree 

class DecisionTree:
    def __init__(self, root=None, max_depth=13):
        """Constructor of the decision tree"""
        self.root = root #root node of the tree
        self.max_depth = max_depth #maximum depth of the tree 
        
    #region Private Methods
    def _predict(self, single_input):
        """Return the prediction for a single input"""
        node = self.root
        while not node.isLeaf():
            if single_input[node.feature_count] < node.split_value:
                node = node.left_child
            else:
                node = node.right_child
        return node.value
    #endregion
        
    #region Public Methods
    def predict(self, X):
        """Return the predictions for the dataset X"""
        return [self._predict(single_input) for single_input in X]
    #endregion

    

        
    
    