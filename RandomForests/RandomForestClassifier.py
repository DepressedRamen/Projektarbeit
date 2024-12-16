from RandomForests.RandomForest import RandomForest
from DecisionTrees.ClassificationTree import ClassificationTree
from collections import Counter
import random

class RandomForestClassifier(RandomForest): 
    #region Implemented Abstract Methods
    def __init__(self, trees_amount=100, bootstrap_size = 100,  max_depth=13, intervals=5, min_samples_split=20, soft_voting=True):
        """Constructor of the random forest"""
        super().__init__(trees_amount=trees_amount,
                         bootstrap_size=bootstrap_size, 
                         max_depth=max_depth, 
                         intervals=intervals, 
                         min_samples_split=min_samples_split)
        self.soft_voting = soft_voting #if true the forest uses soft voting
    
    def fit(self, X, y):
        """Fit the random forest to the dataset"""
        self.forest = [] #reset the forest
        for _ in range(self.trees_amount):
            X_bootstrap, y_bootstrap = self._create_bootstrap_sample(X, y) #create a bootstrap sample 
            #create a new tree
            tree = ClassificationTree(max_depth=self.max_depth, intervals=self.intervals, min_samples_split=self.min_samples_split, random_feature_sampling=True, soft_voting=self.soft_voting)
            tree.fit(X_bootstrap, y_bootstrap) #fit the tree to the bootstrap sample
            self.forest.append(tree) #add the tree to the forest
    
    def _predict_single_input(self, single_input):
        """Predict the label for a single input"""
        if self.soft_voting:
            '''Forest uses Soft Voting'''
            #iterate over all trees in the forest
            probabilites = {}
            for tree in self.forest:
                #predict the label for the input
                tree_dict = tree.predict_proba_single_input(single_input) 
                
                #iterate over all classes in the dictionary
                for key in tree_dict:
                    probabilites[key] = probabilites.get(key, 0) + tree_dict[key]
                    
            #normalize the probabilites
            number_of_trees = len(self.forest)
            averaged_probabilites = {key: value/number_of_trees for key, value in probabilites.items()}
            
            #return the label with the highest probability    
            return max(averaged_probabilites, key=averaged_probabilites.get)              
                
        else: 
            '''Forest uses Hard Voting'''
            predictions = [] #list to store the predictions of the trees
            #iterate over all trees in the forest
            for tree in self.forest:
                predictions.append(tree.predict_single_input(single_input)) #predict the label for the input
            predictions_tuple = tuple(predictions) #convert the list to a tuple to make it hashable for the Counter

            #return the most common label of the predictions
            label_counter = Counter(predictions_tuple)
            return label_counter.most_common(1)[0][0]
    #endregion
    
    #region Private Methods
    def _predict_proba_single_input(self, single_input):
        """Return the probabilites for a single input"""
        probabilites = {}
        for tree in self.forest:
            tree_dict = tree.predict_proba_single_input(single_input) 
            for key in tree_dict:
                probabilites[key] = probabilites.get(key, 0) + tree_dict[key]
        number_of_trees = len(self.forest)
        return {key: value/number_of_trees for key, value in probabilites.items()}
    #endregion
    
    #region Public Methods
    def predict_proba(self, X):
        """Return the probabilites for the dataset X"""
        if not self.soft_voting:
            raise TypeError("The forest does not use soft voting")
        return [self._predict_proba_single_input(single_input) for single_input in X]
    #endregion