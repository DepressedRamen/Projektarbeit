# Projektarbeit

Eine eigenständige Implementierung der ML-Modelle [Random Forest](https://en.wikipedia.org/wiki/Random_forest) und [Gradient Boosting Trees](https://en.wikipedia.org/wiki/Gradient_boosting), welche im Rahmen einer Projektarbeit erstellt werden. Die Modelle werden in Python implementiert und sollen sowohl für Regression als auch für Klassifikation genutzt werden können. 

## 1. Implementierung 
### 1.1 Entscheidungsbäume 
Die Implementierung der Entscheidungsbäume befindet sich im Modul "DecisionTrees". In diesem existiert die abstrakte Basisklasse DecisionTree von der die Klassen ClassficationTree und RegressionTree erben. Der ClassificationTree ist ein Entscheidungsbaum, welcher Vorhersagen für Klassfikationsprobleme vornimmt. Das Modell wird dabei über die Berechnung von [Entropie](https://en.wikipedia.org/wiki/Entropy_(information_theory)) und [Information Gain](https://en.wikipedia.org/wiki/Information_gain_(decision_tree)) erstellt. RegressionTree ist ein Entscheidungsbaum, welcher Vorhersagen für Regressionsprobleme vornimmt. Dieses Modell wird über die Berechnung des [Mean Squared Errors](https://en.wikipedia.org/wiki/Mean_squared_error) trainiert. 

