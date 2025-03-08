from abc import ABC, abstractmethod
import numpy
#Parent class for Gradient Boosting Trees

class GradientBoostingTrees(ABC):
    def __init__(self,  trees_amount=100, learning_rate=0.1,  max_depth=13, intervals=256, min_samples_split=2, prune_alpha=0.0):
        """
        Initializes the Gradient Boosting Trees model with the specified parameters.

        Parameters
        ----------
            trees_amount (int) : The number of trees in the forest. Must be at least 1. Default is 100.
            learning_rate (float) : The learning rate of the model. Must be greater than 0 and less than or equal to 1. Default is 0.1.
            max_depth (int) : The maximum depth of the trees. Must be at least 1. Default is 13.
            intervals (int) : The number of intervals for numerical features. Must be at least 1 if specified. Default is 256.
            min_samples_split (int) : The minimum number of samples required to split an internal node. Must be at least 2. Default is 2.
            prune_alpha (float) : The alpha value for pruning the trees. Must be at least 0. Default is 0.0.

        Raises
        ------
            ValueError : If any of the input parameters do not meet their respective constraints.
        """

        #check the input parameters
        if trees_amount < 1:
            raise ValueError("The number of trees must be at least 1")
        if learning_rate <= 0 or learning_rate > 1:
            raise ValueError("The learning rate must be greater than 0")
        if max_depth < 1:
            raise ValueError("The maximum depth must be at least 1")
        if intervals is not None and intervals < 1:
            raise ValueError("The number of intervals must be at least 1")
        if min_samples_split < 2:
            raise ValueError("The minimum number of samples for a leaf node must be at least 1")
        if prune_alpha < 0:
            raise ValueError("The alpha value for pruning must be at least 0")
        
        #set the parameters
        self.learning_rate = learning_rate #learning rate of the model
        self.trees_amount = trees_amount #number of trees in the forest
        self.max_depth = max_depth #maximum depth of the trees
        self.intervals = intervals #number of intervals for numerical features
        self.min_samples_split = min_samples_split #minimum number of samples for a leaf node
        self.trees = [] #list of trees
        self.prune_alpha = prune_alpha #alpha value for pruning the trees
        
    #region Abstract Methods
    @abstractmethod
    def fit(self, X, y):
        """
        Fit the Gradient Boosting Trees model to the dataset.

        Parameters
        ----------
            X (array-like) : The input samples with shape (n_samples, n_features).
            y (array-like) : The target values with shape (n_samples,).
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Predict the target values for the given input data.

        Parameters
        ----------
            X (array-like) : The input samples for which to predict the target values.

        Returns
        -------
            predictions (list) : The predicted target values for the input samples.
        """

        pass 
    #endregion
    
    #region Private Methods
    def _validate_data(self, X, y=None):
        """
        Validate and preprocess the input data.
        
        Parameters
        -----------
        X (array-like) : The feature matrix to be validated. It should be convertible to a 2D numpy array.
        y (array-like) : The target values to be validated. If provided, it will be validated against the feature matrix X.
        
        Returns
        --------
        X (numpy.ndarray) : The validated and possibly converted feature matrix.
        y (numpy.ndarray) : The validated target values, if provided.
        
        Raises
        -------
            ValueError : If the feature matrix X is not two-dimensional.
        """
        
        if not isinstance(X, numpy.ndarray):
            X = self._convert_to_numpy(X)
        if X.ndim != 2:
            raise ValueError("The feature matrix X must be two-dimensional")

        #if the target values are provided, validate them
        if y is not None:
            y = self._validate_y(y, X)
            return X, y
        
        return X
    
    def _validate_y(self, y, X):
        """
        Validate the target vector y.

        Parameters
        ----------
        y (array-like) : The target vector to be validated. It can be a list, pandas Series, or numpy array.
        X (numpy.ndarray) : The feature matrix. It is used to check if the number of samples matches with y.

        Returns
        -------
        y (numpy.ndarray) : The validated target vector y as a one-dimensional numpy array.

        Raises
        ------
        ValueError : If y is not one-dimensional or if the number of samples in X and y do not match.
        """
        
        if not isinstance(y, numpy.ndarray):
            y = self._convert_to_numpy(y)
        if y.ndim != 1:
            raise ValueError("The target vector y must be one-dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be equal")
        return y
        
    def _convert_to_numpy(self, data):
        """
        Convert the input data to a numpy array.

        Parameters
        ----------
        data (any) : The input data to be converted.

        Returns
        -------
        converted_data (numpy.ndarray) : The converted numpy array.

        Raises:
        ValueError: If the input data could not be converted to a numpy array.
        """
        """Convert the input data to a numpy array"""
        try:
            return numpy.array(data)
        except:
            raise ValueError("The input data could not be converted to a numpy array")
   #endregion