Implement a gradient boosting regression tree for regression with python, numpy and scipy.  
This estimator builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative gradient of the given loss function.

### Algorithmic Flow

1. **Initialization**:
   - Start with an initial model, typically a constant value. For regression, this is often the mean of the target values:
     $$
     F_0(x) = \arg\min_{\gamma} \sum_{i=1}^{n} L(y_i, \gamma)
     $$
     where $ L $ is the loss function, $ y_i $ are the true values, and $ \gamma $ is a constant.

2. **Iterative Boosting Process**:
   - For each iteration $ m = 1, 2, \ldots, M $:
     1. **Compute the Pseudo-Residuals**:
        - Calculate the negative gradient of the loss function with respect to the current predictions:
          $$
          r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F(x) = F_{m-1}(x)}
          $$
        - For squared error loss, this simplifies to:
          $$
          r_{im} = y_i - F_{m-1}(x_i)
          $$

     2. **Fit a CART Regressor**:
        - Fit a regression tree $ h_m(x) $ to the pseudo-residuals $ r_{im} $.

     3. **Compute the Step Size**:
        - Determine the optimal step size $ \gamma_m $ by minimizing the loss function:
          $$
          \gamma_m = \arg\min_{\gamma} \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))
          $$

     4. **Update the Model**:
        - Update the model by adding the scaled tree to the current model:
          $$
          F_m(x) = F_{m-1}(x) + \nu \gamma_m h_m(x)
          $$
        - Here, $ \nu $ is the learning rate, a small positive number that controls the contribution of each tree.

3. **Final Prediction**:
   - After $ M $ iterations, the final model is:
     $$
     F_M(x) = F_0(x) + \sum_{m=1}^{M} \nu \gamma_m h_m(x)
     $$

The base regressor is cart, which you should first implement. 

The module should be named GPTGradientBoostRegression.  

The init function should include the following parameters:

    learning_rate: Learning rate shrinks the contribution of each tree by learning_rate;
    max_depth: Maximum depth of the individual regression estimators;
    n_estimators: The number of boosting stages to perform;
    min_samples_split: The minimum number of samples required to split an internal node;
    min_samples_leaf: The minimum number of samples required to be at a leaf node;
    subsample: The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting.
The module must contain a fit function and a predict function.  

The fit function accepts X_train and y_train and return None where  

    X_train: the features of the train data, which is a numpy array, and the shape of X_train is [N, d]. N is the number of the train data and d is the dimension.  
    y_train: the labels pf the train data,which is a numpy array.  
The predict function accepts X_test as input and return predictions where  

    X_test: the features of the test data, which is a numpy array, and the shape of X_train is [N, d]. N is the number of the test data and d is the dimension.  
    predctions: the predicted results for X_test, which is a numpy arrary.  
You should just return the code for the module, don't return anything else