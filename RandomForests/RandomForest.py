import random
from abc import ABC, abstractmethod
import multiprocessing 

class RandomForest(ABC):
    def __init__(self, trees_amount=100, bootstrap_size = 100,  max_depth=13, intervals=5, min_samples_split=20, threads = 1):
        """Constructor of the random forest"""
        #check the input parameters
        if trees_amount < 1:
            raise ValueError("The number of trees must be at least 1")
        if bootstrap_size < 1 or bootstrap_size > 100:
            raise ValueError("The bootstrap size must be between 1 and 100")
        if max_depth < 1:
            raise ValueError("The maximum depth must be at least 1")
        if intervals < 1:
            raise ValueError("The number of intervals must be at least 1")
        if min_samples_split < 1:
            raise ValueError("The minimum number of samples for a leaf node must be at least 1")
        if threads is not None and threads < 1:
            raise ValueError("The number of threads must be at least 1")
        self.trees_amount = trees_amount #number of trees in the forest
        self.bootstrap_size = bootstrap_size #size of the bootstrap sample in percent
        self.max_depth = max_depth #maximum depth of the trees
        self.intervals = intervals #number of intervals for numerical features
        self.min_samples_split = min_samples_split #minimum number of samples for a leaf node
        self.threads = threads #number of threads to use for training and predicting 
        self.forest = [] #list of trees in the forest
        
    #region Abstract Methods
    @abstractmethod
    def fit(self, X, y):
        """Fit the random forest to the dataset"""
        pass
    
    @abstractmethod 
    def _predict_tree(self, args):
        """Predict all labels for a single tree"""
        pass
    
    @abstractmethod
    def _aggregate_predictions(self, predictions):
        """Aggregate predictions from all trees for a single input"""
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
        #predict the labels for each tree in parallel    
        with multiprocessing.Pool(processes=self.threads) as pool:
            tree_predictions = pool.map(self._predict_tree, [(tree, X) for tree in self.forest])
        
        # Transpose the list of lists to get predictions per input
        predictions_per_input = list(zip(*tree_predictions))
        
        # Aggregate predictions per input
        return [self._aggregate_predictions(predictions) for predictions in predictions_per_input]
    #endregion