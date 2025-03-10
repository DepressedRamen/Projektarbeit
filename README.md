# Project Work

A self made implementation of the machine learning models [random forest](https://en.wikipedia.org/wiki/Random_forest) and [gradient boosting trees](https://en.wikipedia.org/wiki/Gradient_boosting), created as part of a project at the Hochschule Karlsruhe. The models are implemented in Python and can be used for both regression and classification tasks.

## Table of Contents
- [1. Implementation](#1-implementation)
  - [1.1 Decision Trees](#11-decision-trees)
  - [1.2 Random Forest](#12-random-forest)
  - [1.3 Gradient Boosting Trees](#13-gradient-boosting-trees)
- [2. Limitations](#2-limitations)
- [3. Integration and Execution](#3-integration-and-execution)
  - [3.1 Decision Trees](#31-decision-trees)
  - [3.2 Random Forest Classification](#32-random-forest-classification)

## 1. Implementation
### 1.1 Decision Trees
The implementation of [decision trees](https://en.wikipedia.org/wiki/Decision_tree) can be found in the "DecisionTrees" module. It contains the abstract base class `DecisionTree`, from which the classes `ClassificationTree` and `RegressionTree` inherit.

The `ClassificationTree` is a decision tree that makes predictions for classification tasks. The model is created by calculating the [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)).

The `RegressionTree` is a decision tree that makes predictions for regression tasks. This model is trained by calculating the [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error). The trees also support [pruning](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning).

### 1.2 Random Forest
The implementation of the random forest models can be found in the "RandomForests" module. It contains the abstract base class `RandomForest`, from which the classes `RandomForestClassifier` and `RandomForestRegressor` inherit.

For classification problems, both soft and hard voting are supported for aggregating predictions. In regression problems, the average of the predictions is used. Using Python's [_multiprocessing_](https://docs.python.org/3/library/multiprocessing.html) library, it is possible to train the trees of the random forest in parallel.

### 1.3 Gradient Boosting Trees
The implementation of the models can be found in the "GBT" module. The classes `GBTRegressor` and `GBTClassifier` inherit from the abstract base class `GradientBoostingTrees`.

## 2. Limitations
The models have the following limitations:
* classes for classification problems must be numeric.
* feature data `X` must be a two-dimensional array.
* label data `y` must be a one-dimensional array.

## 3. Integration and Execution
1. Integrate the required models into the wanted directory.
2. Import the classes into the desired Python program.
3. Create a model using the constructor.
4. Train the model using the `fit` method.
5. Make predictions using the `predict` method.

The parameters of the individual functions are described in more detail in the docstrings. Individual models have additional functions, which are explained below.

### 3.1 Decision Trees
Decision trees also have the `get_pruning_path` method, which returns possible alpha values for pruning the tree.

### 3.2 Random Forest Classification
The `RandomForestClassifier` class also has the `predict_proba` function, which predicts class probabilities for a random forest. This is only possible if the random forest aggregates its predictions using soft voting.

