#Node structure of a decision tree


class Node:

    def __init__(self,
                 feature_index=None,
                 left_child=None,
                 right_child=None,
                 value=None,
                 split_value=None,
                 leaf_size=None,
                 ccp_node=None,
                 ccp_subtree=None,
                 effective_alpha=None,
                 samples_amount=1):
        """
        Constructor of the node.

        Parameters:
        feature_index (int): ID of the feature of the node.
        left_child (Node): Left child of the node.
        right_child (Node): Right child of the node.
        value (any): Value of the node.
        split_value (float): Value of the split of the node.
        leaf_size (int): Count of the leaf nodes of the subtree with this node as root.
        ccp_node (float): Cost complexity pruning value of the node, used to determine the cost of pruning this node during the pruning process.
        ccp_subtree (float): Sum of the cost complexity pruning values of all nodes in the subtree with this node as root. This value is used to determine the optimal pruning of the subtree during the pruning process.
        effective_alpha (float): Effective alpha of the node.
        samples_amount (int): Number of samples in the node.
        """
        self.feature_index = feature_index
        self.left_child = left_child
        self.right_child = right_child
        self.value = value
        self.split_value = split_value
        self.leaf_size = leaf_size
        self.ccp_node = ccp_node
        self.ccp_subtree = ccp_subtree
        self.effective_alpha = effective_alpha
        self.samples_amount = samples_amount

    def is_leaf(self):
        """
        Check if the node is a leaf.

        A node is considered a leaf if it does not have any children.
        This means both the left_child and right_child attributes are None.
        
        Returns: 
        bool: True if the node is a leaf, False otherwise.
        """
        return self.left_child is None and self.right_child is None

    def _prune_node(self):
        """
        Prune the node by deleting the references to the children and setting the values that are necessary for pruning.

        This involves:
        - Setting `left_child` and `right_child` to None, effectively making this node a leaf.
        - Setting `effective_alpha` to None, as it is no longer relevant for a leaf node.
        - Setting `ccp_subtree` to 0, indicating that there are no subtrees under this node.
        - Setting `leaf_size` to 1, as this node is now a leaf and there is no subtree under it.
        
        Parameters:
        None
        
        Returns:
        None
        """
        self.left_child = None
        self.right_child = None
        self.effective_alpha = None
        self.ccp_subtree = 0
        self.leaf_size = 1
