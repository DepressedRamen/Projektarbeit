#Node structure of a decision tree 

class Node: 
    def __init__(self, feature_count=None, left_child=None, right_child=None, parent=None, value=None, split_value=None):
        """Constructor of the node"""
        self.feature_count =  feature_count #ID of the feature of the node 
        self.left_child = left_child #left child of the node 
        self.right_child = right_child #right child of the node 
        self.parent = parent #parent of the node 
        self.value = value #value of the node 
        self.splid_value = split_value #value of the split of the node 
        
    def is_leaf(self):
        """Check if the node is a leaf"""
        return self.value is not None