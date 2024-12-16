from abc import ABC, abstractmethod

class GradientBoostingTrees(ABC):
    def __init__(self,  trees_amount=100, learning_rate=0.1,  max_depth=13, intervals=5, min_samples_split=2):
        """Constructor of the random forest"""
        if trees_amount < 1:
            raise ValueError("The number of trees must be at least 1")
        if learning_rate <= 0:
            raise ValueError("The learning rate must be greater than 0")
        self.learning_rate = learning_rate #learning rate of the model
        self.trees_amount = trees_amount #number of trees in the forest
        self.max_depth = max_depth #maximum depth of the trees
        self.intervals = intervals #number of intervals for numerical features
        self.min_samples_split = min_samples_split #minimum number of samples for a leaf node
        self.trees = [] #list of trees
        
    #region Abstract Methods
    @abstractmethod
    def fit(self, X, y):
        """Fit the random forest to the dataset"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Return the predictions for the dataset X"""
        pass 
   #endregion