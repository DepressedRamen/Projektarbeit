from GBT.GradientBoostingTrees import GradientBoostingTrees
from DecisionTrees.RegressionTree import RegressionTree
import numpy 
#Class for regression problems using Gradient Boosting Trees

class GBTRegressor(GradientBoostingTrees): 
    #region Abstract methods    
    def fit(self, X, y):     
        """
        Fit the Gradient Boosting Trees model to the data.

        Parameters
        ----------
        X (array-like) : The input samples.
        y (array-like) : The target values.
        """

        X, y = self._validate_data(X, y)
        #initialize the prediction with 0
        pred = 0 
        for _ in range(self.trees_amount):
            #calculate the residual error
            residual_error = y - pred 
            #create a new regression tree
            tree = RegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, intervals=self.intervals, prune_alpha=self.prune_alpha)
            tree.fit(X, residual_error)
            #store the new tree in the list of trees
            self.trees.append(tree)
            #update the prediction with the new tree and the learning rate
            pred += numpy.multiply(tree.predict(X), self.learning_rate)
  
    def predict(self, X):
        """
        Predict the target values for the given dataset X.
        
        Parameters
        ----------
        X (array-like) : The input samples.
        
        Returns
        -------
        y_prediction (ndarray) : The predicted target values.
        """

        X = self._validate_data(X)
        #initialize the prediction with 0 
        y_prediction = numpy.full(X.shape[0], 0.0)
        
        #sum up the predictions of all trees
        for tree in self.trees:
            #multiply the prediction of the tree with the learning rate
            tree_prediction = numpy.multiply(tree.predict(X), self.learning_rate)
            y_prediction += tree_prediction 
            
        return y_prediction
    #endregion 
    