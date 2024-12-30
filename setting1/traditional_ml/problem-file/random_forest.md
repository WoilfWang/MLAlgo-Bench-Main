Implement the random forest classifier with python, numpy and scipy. It can handle multi-class classification problems.

A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 

### Algorithmic Flow

1. **Bootstrap Sampling**: 
   - From the original dataset with $ N $ samples, create $ M $ bootstrap samples. Each bootstrap sample is created by randomly selecting $ N $ samples with replacement from the original dataset.

2. **Building Decision Trees**:
   - For each bootstrap sample, grow a decision tree. However, instead of considering all features for splitting at each node, only a random subset of features is considered. This is known as feature bagging and helps in decorrelating the trees.
   - The decision tree is built using the Gini impurity as the criterion for splitting nodes.

3. **Gini Impurity Calculation**:
   - For a node $ t $, the Gini impurity is calculated as:
     $$
     Gini(t) = 1 - \sum_{i=1}^{C} p(i|t)^2
     $$
     where $ C $ is the number of classes and $ p(i|t) $ is the proportion of class $ i $ instances among the training instances in node $ t $.

4. **Splitting Criterion**:
   - For each candidate split, calculate the Gini impurity for the left and right child nodes. The Gini gain is computed as:
     $$
     \Delta Gini = Gini(parent) - \left( \frac{N_{left}}{N_{total}} \times Gini(left) + \frac{N_{right}}{N_{total}} \times Gini(right) \right)
     $$
     where $ N_{left} $ and $ N_{right} $ are the number of samples in the left and right nodes, respectively, and $ N_{total} $ is the total number of samples in the parent node.
   - Choose the split that maximizes the Gini gain.

5. **Tree Growth**:
   - Continue splitting nodes until a stopping criterion is met (e.g., maximum depth, minimum number of samples per leaf, or no further Gini gain).

6. **Aggregation**:
   - Once all trees are built, the Random Forest classifier aggregates the predictions from each tree. For classification, it uses majority voting to determine the final class label for each instance.


The base classifier is decision tree, which you should implement from scratch. You should use the Gini coefficient as the criterion.

The module should be named GPTRandomForestClassifier.  

The init function should include the following parameters:

      n_estimators: The number of trees in the forest;
      max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples;
      min_samples_split: The minimum number of samples required to split an internal node;
      min_samples_leaf: The minimum number of samples required to be at a leaf node.
The module must contain a fit function and a predict function.  

The fit function accepts X_train, y_train as input and return None where  

      X_train: the features of the train data, which is a numpy array, and the shape of X_train is [N, d]. N is the number of the train data and d is the dimension.  
      y_train: the labels pf the train data,which is a numpy array.  
The predict function accepts X_test as input and return predictions where  

      X_test: the features of the test data, which is a numpy array, and the shape of X_train is [N, d]. N is the number of the test data and d is the dimension.  
      predctions: the predicted classes for X_test, which is a numpy arrary.  
You should just return the code for the module, don't return anything else.