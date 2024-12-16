#Node structure of a decision tree 

class Node: 
    def __init__(self, feature_index=None, left_child=None, right_child=None, value=None, split_value=None, leaf_size=None, ccp_node=None, ccp_subtree=None, effective_alpha=None, n=1):
        """Constructor of the node"""
        self.feature_index =  feature_index #ID of the feature of the node 
        self.left_child = left_child #left child of the node 
        self.right_child = right_child #right child of the node 
        self.value = value #value of the node 
        self.split_value = split_value #value of the split of the node 
        self.leaf_size = leaf_size #number of the leaf nodes of the subtree with this node as root
        self.ccp_node = ccp_node #cost complexity pruning value of the node
        self.ccp_subtree = ccp_subtree #cost complexity pruning value of the subtree with this node as root
        self.effective_alpha = effective_alpha #effective alpha of the node
        self.n = n #number of samples in the node
        
    def is_leaf(self):
        """Check if the node is a leaf"""
        return self.left_child is None and self.right_child is None
    
    def _prune_node(self):
        """Prune the node by deleting the references to the children and setting the values that are necessary for pruning"""
        self.left_child = None
        self.right_child = None
        self.effective_alpha = None
        self.ccp_subtree = 0
        self.leaf_size = 1