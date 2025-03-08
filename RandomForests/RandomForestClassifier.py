from RandomForests.RandomForest import RandomForest
from DecisionTrees.ClassificationTree import ClassificationTree
from collections import Counter
#Class for solving classification problems using a Random Forest.

class RandomForestClassifier(RandomForest): 
    def __init__(self, trees_amount=100, bootstrap_size = 100,  max_depth=13, intervals=256, min_samples_split=20, soft_voting=True, threads = 1, prune_alpha=0.0):
        """
        Initializes the RandomForestClassifier with the specified parameters.

        Parameters
        ----------
            trees_amount (int) : The number of trees in the forest. Default is 100.
            bootstrap_size (int) : The size of the bootstrap sample. Default is 100.
            max_depth (int) : The maximum depth of the trees. Default is 13.
            intervals (int) : The number of intervals for discretization. Default is 256.
            min_samples_split (int) : The minimum number of samples required to split an internal node. Default is 20.
            soft_voting (bool) : If True, the forest uses soft voting. Default is True.
            threads (int) : The number of threads to use for parallel processing. Default is 1.
            prune_alpha (float) : The alpha parameter for pruning. Default is 0.0.
        """

        super().__init__(trees_amount=trees_amount,
                         bootstrap_size=bootstrap_size, 
                         max_depth=max_depth, 
                         intervals=intervals, 
                         min_samples_split=min_samples_split,
                         threads=threads,
                         prune_alpha=prune_alpha) #call the constructor of the RandomForest class
        self.soft_voting = soft_voting #if true the forest uses soft voting
    
    #region Abstract Methods    
    def _create_and_fit_tree(self, args):
        """
        Create and fit a classification tree to a bootstrap sample.

        Parameters
        ----------
            args (tuple) : A tuple containing the feature matrix X and the target vector y.

        Returns
        -------
            tree (ClassificationTree) : A fitted classification tree.
        """

        X, y = args
        #create a bootstrap sample
        X_bootstrap, y_bootstrap = self._create_bootstrap_sample(X, y) 
        #create a tree
        tree = ClassificationTree(max_depth=self.max_depth, 
                                  intervals=self.intervals, 
                                  min_samples_split=self.min_samples_split, 
                                  random_feature_sampling=True, 
                                  soft_voting=self.soft_voting,
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
            predictions (list) : A list of predictions for each input in X.
        """
        
        tree, X = args
        return [tree.predict_single_input(single_input, self.soft_voting) for single_input in X]   
    
    def _aggregate_predictions(self, predictions):
        """
        Aggregate predictions from all trees for a single input.
        This method aggregates the predictions from all the trees in the forest
        to produce a final prediction for a single input. It supports both soft
        voting and hard voting.
        
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
    #endregion
    
    #region Private Methods    
    def _predict_proba_single_input(self, single_input):
        """
        Return the probabilities for a single input.

        This method calculates the probabilities for each class label for a given single input
        by aggregating the predictions from all the trees in the forest. Each tree's prediction
        is a dictionary where the keys are class labels and the values are the predicted probabilities
        for those labels. The method sums these probabilities for each class label across all trees
        and then normalizes them by dividing by the number of trees.

        Parameters
        ----------
            single_input (array-like) : The input data for which to predict probabilities.

        Returns
        -------
            avg_probs (dict) : A dictionary where the keys are class labels and the values are the normalized
                  probabilities for those labels.
        """
        
        probabilites = {}
        for tree in self.forest:
            tree_dict = tree.predict_single_input(single_input, True) 
            for key in tree_dict:
                probabilites[key] = probabilites.get(key, 0) + tree_dict[key]
        number_of_trees = len(self.forest)
        return {key: value/number_of_trees for key, value in probabilites.items()}
    #endregion
    
    #region Public Methods
    def predict_proba(self, X):
        """
        Return the probabilities for the dataset X.

        Parameters
        ----------
            X (array-like) : The input samples.

        Returns
        -------
            probabilities (list) : The class probabilities of the input samples. 

        Raises
        ------
            TypeError : If the forest does not use soft voting.
        """

        if not self.soft_voting:
            raise TypeError("The forest does not use soft voting")
        return [self._predict_proba_single_input(single_input) for single_input in X]
    #endregion