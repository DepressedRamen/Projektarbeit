from RandomForests.RandomForest import RandomForest
from DecisionTrees.ClassificationTree import ClassificationTree
from collections import Counter
import random

class RandomForestClassifier(RandomForest): 
    #region Implemented Abstract Methods
    def fit(self, X, y):
        """Fit the random forest to the dataset"""
        self.forest = [] #reset the forest
        for _ in range(self.trees_amount):
            X_bootstrap, y_bootstrap = self._create_bootstrap_sample(X, y) #create a bootstrap sample 
            #create a new tree
            tree = ClassificationTree(max_depth=self.max_depth, intervals=self.intervals, min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap) #fit the tree to the bootstrap sample
            self.forest.append(tree) #add the tree to the forest
    
    def _predict_single_input(self, single_input):
        """Predict the label for a single input"""
        predictions = [] #list to store the predictions of the trees
        for tree in self.forest:
            predictions.append(tree.predict_single_input(single_input)) #predict the label for the input
        predictions_tuple = tuple(predictions) #convert the list to a tuple to make it hashable for the Counter

        #return the most common label of the predictions
        label_counter = Counter(predictions_tuple)
        return label_counter.most_common(1)[0][0]
    #endregion
