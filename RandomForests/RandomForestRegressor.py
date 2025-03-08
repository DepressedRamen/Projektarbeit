from RandomForests.RandomForest import RandomForest
from DecisionTrees.RegressionTree import RegressionTree
import numpy
#Class for solving regression problems using a Random Forest.

class RandomForestRegressor(RandomForest):
    #region Abstract Methods    
    def _create_and_fit_tree(self, args):
        """
        Create and fit a regression tree to a bootstrap sample.

        Parameters:
        ----------
            args (tuple): A tuple containing the feature matrix X and the target vector y.

        Returns
        -------
            tree (RegressionTree) : A fitted regression tree.
        """

        X, y = args
        #create a bootsrap sample
        X_bootstrap, y_bootstrap = self._create_bootstrap_sample(X, y) 
        tree = RegressionTree(max_depth=self.max_depth, 
                              intervals=self.intervals, 
                              min_samples_split=self.min_samples_split, 
                              random_feature_sampling=True,
                              prune_alpha=self.prune_alpha)
        #fit the tree to the bootstrap sample
        tree.fit(X_bootstrap, y_bootstrap) 
        return tree
    
    def _predict_tree(self, args):
        """
        Helper function to predict all inputs for a single tree.

        Parameters
        ----------
            args (tuple) : A tuple containing a tree object and the input data (X).
                        - tree: The decision tree used for prediction.
                        - X: A list or array of input data to be predicted.

        Returns
        -------
            predictions (list): A list of predictions for each input in X.
        """
        
        tree, X = args
        #predict the labels for each input
        return [tree.predict_single_input(single_input) for single_input in X]  
        
    
    def _aggregate_predictions(self, predictions):
        """
        Aggregate predictions from all trees for a single input.
        This method aggregates the predictions from all the trees in the forest
        to produce a final prediction for a single input.
        
        Parameters
        ----------
            predictions (list of dict) : A list where each element is a dictionary
                                         representing the prediction probabilities
                                         from a single tree.
        
        Returns
        -------
            prediction (int) : The final predicted class label. If soft voting is used, it
                        returns the class with the highest averaged probability. If
                        hard voting is used, it returns the class with the most votes.
        """
        
        #return the mean of the predictions
        return numpy.mean(predictions)
    #endregion  