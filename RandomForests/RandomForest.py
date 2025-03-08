import random
from abc import ABC, abstractmethod
import multiprocessing 
import numpy
#Parent class for the random forest models

class RandomForest(ABC):
    def __init__(self, trees_amount=100, bootstrap_size = 100,  max_depth=13, intervals=256, min_samples_split=20, threads = 1, prune_alpha=0.0):
        """
        Constructor of the RandomForest class.

        Parameters
        ----------
            trees_amount (int) : Number of trees in the forest. Must be at least 1.
            bootstrap_size (int) : Size of the bootstrap sample in percent. Must be between 1 and 100.
            max_depth (int) : Maximum depth of the trees. Must be at least 1.
            intervals (int) : Number of intervals for numerical features. Must be at least 1 if provided.
            min_samples_split (int) : Minimum number of samples for a leaf node. Must be at least 2.
            threads (int) : Number of threads to use for training and predicting. Must be at least 1 if provided.
            prune_alpha (float) : Alpha value for pruning the trees. Must be at least 0.

        Raises
        ------
            ValueError : If any of the input parameters do not meet their respective constraints.
        """

        #check the input parameters
        if trees_amount < 1:
            raise ValueError("The number of trees must be at least 1")
        if bootstrap_size < 1 or bootstrap_size > 100:
            raise ValueError("The bootstrap size must be between 1 and 100")
        if max_depth < 1:
            raise ValueError("The maximum depth must be at least 1")
        if intervals is not None and intervals < 1:
            raise ValueError("The number of intervals must be at least 1")
        if min_samples_split < 2:
            raise ValueError("The minimum number of samples for a leaf node must be at least 1")
        if threads is not None and threads < 1:
            raise ValueError("The number of threads must be at least 1")
        if prune_alpha < 0:
            raise ValueError("The alpha value for pruning must be at least 0")
        self.trees_amount = trees_amount #number of trees in the forest
        self.bootstrap_size = bootstrap_size #size of the bootstrap sample in percent
        self.max_depth = max_depth #maximum depth of the trees
        self.intervals = intervals #number of intervals for numerical features
        self.min_samples_split = min_samples_split #minimum number of samples for a leaf node
        self.threads = threads #number of threads to use for training and predicting 
        self.forest = [] #list of trees in the forest
        self.prune_alpha = prune_alpha #alpha value for pruning the trees
        
    #region Abstract Methods  
    @abstractmethod
    def _create_and_fit_tree(self, args):
        """
        Create and fit a tree to a bootstrap sample.

        Parameters
        ----------
            args : The input arguments required for creating and fitting the tree.

        Returns
        -------
            tree : The trained decision tree.
        """
        pass
      
    @abstractmethod 
    def _predict_tree(self, args):
        """
        Predict all labels for a single tree.

        Parameters
        ----------
            args : The input arguments required for prediction.

        Returns
        -------
            predictions (list) : The list of predictions for the tree.
        """
        pass
    
    @abstractmethod
    def _aggregate_predictions(self, predictions):
        """
        Aggregate predictions from all trees for a single input.

        This method takes the predictions from all the trees in the random forest
        and combines them to produce a final prediction for a single input.

        Parameters
        ----------
            predictions (list) : A list of predictions from each tree.

        Returns
        -------
            prediction (any) : The aggregated prediction.
        """
        pass
    #endregion
    
    #region Private Methods
    def _create_bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample from the given dataset.

        Parameters
        ----------
            X (array-like): Feature matrix.
            y (array-like): Target vector.

        Returns
        -------
        boostrap_sample (tuple) : A tuple containing the bootstrap sample of the feature matrix and the target vector.
        """

        #create a list of random indices with replacement
        sample_size = round(len(y)*(float(self.bootstrap_size)/100.0))
        indices = [random.randint(0, len(y)-1) for _ in range(sample_size)]
        #return the bootstrap sample
        return X[indices], y[indices]
    
    def _convert_to_numpy(self, data):
        """
        Convert the input data to a numpy array.

        Parameters
        ----------
            data (any): The input data to be converted.

        Returns
        -------
            converted_data (numpy.ndarray) : The converted numpy array.

        Raises
        ------
            ValueError : If the input data cannot be converted to a numpy array.
        """

        try:
            return numpy.array(data)
        except:
            raise ValueError("The input data could not be converted to a numpy array")
    #endregion
    
    #region Public Methods
    def fit(self, X, y):
        """
        Fit the random forest to the dataset.
        
        Parameters
        ----------
            X (numpy.ndarray or array-like) : The feature matrix with shape (n_samples, n_features).
            y (numpy.ndarray or array-like) : The target vector with shape (n_samples,).
        
        Raises
        ------
            ValueError : If the input data does not meet the constraints.
        """

        # Convert the input data to numpy arrays
        if not isinstance(X, numpy.ndarray):
            X = self._convert_to_numpy(X)
        if not isinstance(y, numpy.ndarray):
            y = self._convert_to_numpy(y)
            
        # Verify that the input data meets the constraints
        if y.ndim != 1:
            raise ValueError("The target vector y must be one-dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be equal")
        if X.ndim != 2:
            raise ValueError("The feature matrix X must be two-dimensional")
        
        #reset the forest
        self.forest = []
        
        #create the trees in parallel
        with multiprocessing.Pool(processes=self.threads) as pool:
            self.forest = pool.map(self._create_and_fit_tree, [(X, y)] * self.trees_amount)
    
    def predict(self, X):
        """
        Predict the labels for the given dataset X using the trained random forest.
        
        Parameters
        ----------
            X (array-like) : The input dataset for which predictions are to be made.
        Returns
        -------
            predictions (list) : A list of predicted labels for each input sample in X.
        """
 
        #predict the labels for each tree in parallel    
        with multiprocessing.Pool(processes=self.threads) as pool:
            tree_predictions = pool.map(self._predict_tree, [(tree, X) for tree in self.forest])
        
        # Transpose the list of lists to get predictions per input
        predictions_per_input = list(zip(*tree_predictions))
        
        # Aggregate predictions per input
        return [self._aggregate_predictions(predictions) for predictions in predictions_per_input]
    #endregion