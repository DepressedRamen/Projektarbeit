from RandomForests.RandomForest import RandomForest
from DecisionTrees.ClassificationTree import ClassificationTree
from collections import Counter
import random
import multiprocessing

class RandomForestClassifier(RandomForest): 
    #region Implemented Abstract Methods
    def __init__(self, trees_amount=100, bootstrap_size = 100,  max_depth=13, intervals=5, min_samples_split=20, soft_voting=True, threads = 1):
        """Constructor of the random forest"""
        super().__init__(trees_amount=trees_amount,
                         bootstrap_size=bootstrap_size, 
                         max_depth=max_depth, 
                         intervals=intervals, 
                         min_samples_split=min_samples_split,
                         threads=threads)
        self.soft_voting = soft_voting #if true the forest uses soft voting
    
    def fit(self, X, y):
        """Fit the random forest to the dataset"""
        #reset the forest
        self.forest = []
        
        #create the trees in parallel
        with multiprocessing.Pool(processes=self.threads) as pool:
            self.forest = pool.map(self._create_and_fit_tree, [(X, y)] * self.trees_amount)
    
    def _create_and_fit_tree(self, args):
        """Create and fit a tree to a bootstrap sample"""	
        X, y = args
        #create a bootstrap sample
        X_bootstrap, y_bootstrap = self._create_bootstrap_sample(X, y) 
        #create a tree
        tree = ClassificationTree(max_depth=self.max_depth, 
                                  intervals=self.intervals, 
                                  min_samples_split=self.min_samples_split, 
                                  random_feature_sampling=True, 
                                  soft_voting=self.soft_voting)
        #fit the tree to the bootstrap sample
        tree.fit(X_bootstrap, y_bootstrap) 
        return tree
        
    def _predict_tree(self, args):
        """Helper function to predict all inputs for a single tree"""
        tree, X = args
        if(self.soft_voting):
            '''Forest uses Soft Voting'''
            return [tree.predict_proba_single_input(single_input) for single_input in X]
        else: 
            '''Forest uses Hard Voting'''
            return [tree.predict_single_input(single_input) for single_input in X]    
    
    def _aggregate_predictions(self, predictions):
        """Aggregate predictions from all trees for a single input"""
        if self.soft_voting:
            '''Forest uses Soft Voting'''
            probabilites = {}
            #sum up the probabilites for each class
            for tree_dict in predictions:
                for key in tree_dict:
                    probabilites[key] = probabilites.get(key, 0) + tree_dict[key]
                    
            #calculate the averaged probabilites
            number_of_trees = len(predictions)
            averaged_probabilites = {key: value/number_of_trees for key, value in probabilites.items()}
            
            #return the class with the highest probability
            return max(averaged_probabilites, key=averaged_probabilites.get)
        else:
            '''Forest uses Hard Voting'''
            label_counter = Counter(predictions)
            #return the label with the most votes
            return label_counter.most_common(1)[0][0]
    
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