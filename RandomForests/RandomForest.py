import random
from abc import ABC, abstractmethod

class RandomForest(ABC):
    def __init__(self, trees_amount=100, bootstrap_size = 100,  max_depth=13, intervals=5, min_samples_split=20):
        """Constructor of the random forest"""
        if trees_amount < 1:
            raise ValueError("The number of trees must be at least 1")
        self.trees_amount = trees_amount #number of trees in the forest
        self.bootstrap_size = bootstrap_size #size of the bootstrap sample in percent
        self.max_depth = max_depth #maximum depth of the trees
        self.intervals = intervals #number of intervals for numerical features
        self.min_samples_split = min_samples_split #minimum number of samples for a leaf node
        self.forest = [] #list of trees in the forest
        
    #region Abstract Methods
    @abstractmethod
    def fit(self, X, y):
        """Fit the random forest to the dataset"""
        pass
    
    @abstractmethod
    def _predict_single_input(self, single_input):
        """Return the prediction for a single input"""
        pass
    #endregion
    
    #region Private Methods
    def _create_bootstrap_sample(self, X, y):
        """Create a bootstrap sample"""
        #create a list of random indices with replacement
        sample_size = round(len(y)*(float(self.bootstrap_size)/100.0))
        indices = [random.randint(0, len(y)-1) for _ in range(sample_size)]
        #return the bootstrap sample
        return X[indices], y[indices]
    #endregion
    
    #region Public Methods
    def predict(self, X):
        """Return the predictions for the dataset X"""
        return [self._predict_single_input(single_input) for single_input in X]
    #endregion