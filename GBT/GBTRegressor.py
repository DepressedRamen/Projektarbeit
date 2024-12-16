from GBT.GradientBoostingTrees import GradientBoostingTrees
from DecisionTrees.RegressionTree import RegressionTree
import numpy 

class GBTRegressor(GradientBoostingTrees): 
    def __init__(self,  trees_amount=100, learning_rate=0.1,  max_depth=13, intervals=5, min_samples_split=2):
        super().__init__(trees_amount, learning_rate, max_depth, intervals, min_samples_split)
        self.initial_prediction = None #initial prediction of the model
    
    #region implement abstract methods
    def fit(self, X, y):
        #initialize the prediction with the mean of the target values
        self.initial_prediction = numpy.mean(y)
        
        
        #create the first regression tree and fit it to the data
        tree = RegressionTree(max_depth=self.max_depth, intervals=self.intervals, min_samples_split=self.min_samples_split)
        tree.fit(X, y)
        
        #predict the target values and store the tree in the list of trees
        y_pred = tree.predict(X)
        self.trees.append(tree)         
        
        #fit the remaining trees to the residual error
        for _ in range(self.trees_amount - 1): 
            #calculate the residual error
            residual_error = y - y_pred
            
            #create a new regression tree
            tree = RegressionTree(max_depth=self.max_depth, intervals=self.intervals, min_samples_split=self.min_samples_split)
            # fit the tree to the residual error
            tree.fit(X, residual_error) 
            
            #predict the target values and update the prediction
            tree_prediction =  numpy.multiply(tree.predict(X), self.learning_rate)
            y_pred += tree_prediction
            
            #store the new tree in the list of trees
            self.trees.append(tree)
            
    def predict(self, X):
        """Return the predictions for the dataset X"""
        #initialize the prediction with the prediction of the first tree
        y_prediction = self.trees[0].predict(X)
        self.trees.pop(0)
        
        #sum up the predictions of all trees
        for tree in self.trees:
            tree_prediction = numpy.multiply(tree.predict(X), self.learning_rate)
            y_prediction += tree_prediction 
            
        return y_prediction
    #endregion 
    