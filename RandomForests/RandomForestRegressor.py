from RandomForests.RandomForest import RandomForest
from DecisionTrees.RegressionTree import RegressionTree
import numpy


class RandomForestRegressor(RandomForest):
    #region Implemented Abstract Methods
    def fit(self, X, y):
        """Fit the random forest to the dataset"""
        self.forest = [] #reset the forest
        for _ in range(self.trees_amount):
            X_bootstrap, y_bootstrap = self._create_bootstrap_sample(X, y) #create a bootstrap sample 
            #create a new tree
            tree = RegressionTree(max_depth=self.max_depth, intervals=self.intervals, min_samples_split=self.min_samples_split, random_feature_sampling=True)
            tree.fit(X_bootstrap, y_bootstrap) #fit the tree to the bootstrap sample
            self.forest.append(tree) #add the tree to the forest
    
    def _predict_single_input(self, single_input):
        """Predict the label for a single input"""
        predictions = [] #list to store the predictions of the trees
        for tree in self.forest:
            predictions.append(tree.predict_single_input(single_input)) #predict the label for the input
        #return the average of the predictions
        return numpy.mean(predictions)
    #endregion 
    
    