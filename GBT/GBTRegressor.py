from GBT.GradientBoostingTrees import GradientBoostingTrees
from DecisionTrees.RegressionTree import RegressionTree
from numpy import zeros

class GBTRegressor(GradientBoostingTrees): 
    #region implement abstract methods
    def fit(self, X, y):
        #TODO: implement a contraint for the tree amounts
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
            y_pred += self.learning_rate * tree.predict(X)
            
            #store the new tree in the list of trees
            self.trees.append(tree)
            
    def predict(self, X):
        """Return the predictions for the dataset X"""
        #initialize the prediction with zeros
        y_prediction = zeros(X.shape[0])
        
        #sum up the predictions of all trees
        for tree in self.trees:
            y_prediction += self.learning_rate * tree.predict(X) 
            
        return y_prediction
    #endregion 
    