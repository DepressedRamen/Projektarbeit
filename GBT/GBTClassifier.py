from GBT.GradientBoostingTrees import GradientBoostingTrees
from DecisionTrees.RegressionTree import RegressionTree
import numpy 
from scipy.special import softmax

class GBTClassifier(GradientBoostingTrees): 
    def __init__(self, trees_amount=100, learning_rate=0.1,  max_depth=13, intervals=5, min_samples_split=20):
        """Constructor of the GBTClassifier"""
        super().__init__(trees_amount, learning_rate, max_depth, intervals, min_samples_split)
        self.classes_amount = None #amount of unique labels
    
    #region implement abstract methods
    def fit(self, X, y):
        '''Fit the model to the data X and the target values y with the help of the softmax function'''
        #initialize epsilon used in the calculation of the gamma values to avoid division by zero
        epsilon = 1e-15
        
        #determine the amount of unique labels
        self.classes_amount = len(set(y))
        
        #initialize logits to zero
        self.logits = numpy.zeros((y.shape[0], self.classes_amount))
        
        #create one hot encoding for the labels
        y_one_hot = self._one_hot_encoding(y)
        
        #initialize the list of trees for each label
        self.trees = [[] for _ in range(self.classes_amount)]
        
        for _ in range(self.trees_amount):
            #calculate the probabilities
            prob = softmax(self.logits, axis=1)
            
            #calculate the residuals
            residuals = y_one_hot - prob
            
            for label in range(self.classes_amount): 
                #create a new classification tree
                tree = RegressionTree(max_depth=self.max_depth, intervals=self.intervals, min_samples_split=self.min_samples_split)
                # fit the tree to the residual error
                #fit the tree to the residuals of the current label
                tree.fit(X, residuals[:, label])
                #store the tree in the list of trees for the current label
                self.trees[label].append(tree)
                
                #calculate the regions of the tree and determine the unique regions
                regions = tree.get_regions(X)
                unique_regions = numpy.unique(regions)
                
                gamma = {}
                #calculate the gamma values for each region
                for region in unique_regions: 
                    #mask the residuals with the current region
                    region_mask = (regions == region)
                    #calculate the sum and the absolute sum of the masked residuals
                    masked_residuals = residuals[region_mask]
                    residual_sum = numpy.sum(masked_residuals)
                    abs_res_sum = numpy.sum(numpy.abs(masked_residuals))  
                    #calculate the gamma value  
                    gamma[region] = ((self.classes_amount - 1)/self.classes_amount) * (residual_sum/(abs_res_sum*(1-abs_res_sum) + epsilon))
                    
                #update the logits
                for region, gamma_value in gamma.items(): 
                    #update the logits of the current label for the current region
                    self.logits[regions==region, label] += self.learning_rate * gamma_value 
                
    def predict(self, X):
        """Return the predictions for the dataset X"""
        #initialize the logits with zeros
        logits = numpy.zeros((X.shape[0], self.classes_amount))
        
        #sum up the predictions of all trees
        for label in range(self.classes_amount):
            for tree in self.trees[label]:
                logits[:, label] += numpy.multiply(tree.predict(X), self.learning_rate)
        
        #return the label with the highest probability
        return numpy.argmax(softmax(logits, axis=1), axis=1)
    #endregion 
    
    def _one_hot_encoding(self, y):
        """One hot encoding of the labels"""
        y_one_hot = numpy.zeros((len(y), self.classes_amount))
        y_one_hot[numpy.arange(len(y)), y] = 1
        return y_one_hot