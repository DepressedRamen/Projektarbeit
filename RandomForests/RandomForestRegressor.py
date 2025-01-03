from RandomForests.RandomForest import RandomForest
from DecisionTrees.RegressionTree import RegressionTree
import numpy
import multiprocessing


class RandomForestRegressor(RandomForest):
    #region Implemented Abstract Methods    
    def fit(self, X, y):
        """Fit the random forest to the dataset"""
        self.forest = [] #reset the forest
        
        #create the trees in parallel
        with multiprocessing.Pool(processes=self.threads) as pool:
            self.forest = pool.map(self._create_and_fit_tree, [(X, y)] * self.trees_amount)
    
    def _create_and_fit_tree(self, args):
        """Create and fit a tree to a bootstrap sample"""	
        X, y = args
        #create a bootsrap sample
        X_bootstrap, y_bootstrap = self._create_bootstrap_sample(X, y) 
        tree = RegressionTree(max_depth=self.max_depth, 
                              intervals=self.intervals, 
                              min_samples_split=self.min_samples_split, 
                              random_feature_sampling=True)
        #fit the tree to the bootstrap sample
        tree.fit(X_bootstrap, y_bootstrap) 
        return tree
    
    def _predict_tree(self, args):
        """Helper function to predict all inputs for a single tree"""
        tree, X = args
        #predict the labels for each input
        return [tree.predict_single_input(single_input) for single_input in X]  
        
    
    def _aggregate_predictions(self, predictions):
        """Aggregate predictions from all trees for a single input"""
        #return the mean of the predictions
        return numpy.mean(predictions)
    #endregion  