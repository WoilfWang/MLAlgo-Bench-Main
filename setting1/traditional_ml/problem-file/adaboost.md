Implement the Adaboost classifier with python, numpy and scipy. It can handle multi-class classification problems. 

The Adaboost (Adaptive Boosting) classifier is a machine learning ensemble technique that is used to boost the accuracy of weak classifiers by combining them into a single strong classifier. The fundamental idea behind Adaboost is to fit a sequence of weak learners (typically simple decision trees, also called decision stumps) on repeatedly modified versions of the data. The predictions from all of them are then combined through a weighted majority vote (or sum) to produce the final prediction. The data modifications at each iteration consist of applying weights to each of the training samples. Initially, all weights are equal, but on each subsequent round, the weights of incorrectly classified instances are increased so that the weak learners focus more on the difficult cases.


Algorithmic Flow

1.	Initialize Weights: Start by assigning equal weights to each of the training samples. If there are  $N$  samples, each sample  $i$  receives an initial weight of  $w_i = \frac{1}{N}$.
2.	For each iteration  $t = 1$  to  $T$ :
	•	Fit a Classifier: Train a weak learner  $h_t$  using the weighted samples. The learner’s goal is to minimize the weighted error  $\epsilon_t$ :

$$\epsilon_t = \frac{\sum_{i=1}^N w_i \cdot \mathbf{1}(y_i \neq h_t(x_i))}{\sum_{i=1}^N w_i}$$

where  $\mathbf{1}(condition)$  is an indicator function that is 1 if the condition is true and 0 otherwise.
•	Calculate the Learner’s Weight  $\alpha_t$ : This weight is calculated based on the error  $\epsilon_t$:

$$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$

Here,  $\alpha_t$  represents the amount of say the classifier will have in the final decision. A smaller  $\epsilon_t$  leads to a larger  $\alpha_t$ .
•	Update Weights: Increase the weights of incorrectly classified instances and decrease the weights of correctly classified instances, then normalize:

$$w_i \leftarrow w_i \cdot \exp\left(-\alpha_t \cdot y_i \cdot h_t(x_i)\right)$$

Afterwards, normalize the weights so that they sum to 1:

$$w_i \leftarrow \frac{w_i}{\sum_{j=1}^N w_j}$$

3.	Combine the Weak Learners: After  $T$  iterations, the final model is a weighted sum of the weak learners:

$$H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t \cdot h_t(x)\right)$$

where $ \text{sign}(z) $ is a function that returns 1 if  $z \geq 0$ and -1 otherwise.

The base classifier is decision tree, and the max depth is set to 1. You should first implement the decision tree from scratch.  

The module should be named GPTAdaboostClassifier.  

The init function should include the following parameters:

	n_estimators: The maximum number of estimators at which boosting is terminated;
	learning_rate: Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier.
The module must contain a fit function and a predict function.  

The fit function accepts X_train, y_train as input and return None where  

	X_train: the features of the train data, which is a numpy array, and the shape of X_train is [N, d]. N is the number of the train data and d is the dimension.  
	y_train: the labels pf the train data,which is a numpy array.  
	
The predict function acceptze s w ds X_test as input and return predictions where  
	X_test: the features of the test data, which is a numpy array, and the shape of X_train is [N, d]. N is the number of the test data and d is the dimension.  
	predctions: the predicted classes for X_test, which is a numpy arrary.  
You should just return the code for the module, don't return anything else.