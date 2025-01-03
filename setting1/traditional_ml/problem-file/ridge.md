Implement a ridge regression model with python, numpy and scipy.  
Linear least squares with l2 regularization. Minimizes the objective function: ||y - Xw||^2_2 + alpha * ||w||^2_2. This model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm.

The principle behind ridge regression is to introduce a penalty term to the ordinary least squares (OLS) loss function. This penalty term is proportional to the square of the magnitude of the coefficients. The goal is to minimize the following cost function:

$$ 
J(\beta) = \sum_{i=1}^{n} (y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij})^2 + \lambda \sum_{j=1}^{p} \beta_j^2 
$$

Here:
- $ y_i $ is the dependent variable.
- $ x_{ij} $ are the independent variables.
- $ \beta_0 $ is the intercept.
- $ \beta_j $ are the coefficients.
- $ \lambda $ is the regularization parameter (also known as the ridge parameter or shrinkage parameter).

### Algorithmic Flow

1. **Standardize the Data**: Before applying ridge regression, it is common to standardize the predictors to have zero mean and unit variance. This ensures that the penalty term is applied uniformly across all coefficients.

2. **Set Up the Ridge Regression Problem**: The ridge regression problem can be expressed in matrix form. Let $ \mathbf{X} $ be the matrix of input features, $ \mathbf{y} $ be the vector of outputs, and $ \mathbf{\beta} $ be the vector of coefficients. The cost function becomes:

   $$
   J(\mathbf{\beta}) = (\mathbf{y} - \mathbf{X}\mathbf{\beta})^T(\mathbf{y} - \mathbf{X}\mathbf{\beta}) + \lambda \mathbf{\beta}^T\mathbf{\beta}
   $$

3. **Derive the Ridge Regression Coefficients**: To find the coefficients that minimize the cost function, take the derivative of $ J(\mathbf{\beta}) $ with respect to $ \mathbf{\beta} $, set it to zero, and solve for $ \mathbf{\beta} $:

   $$
   \mathbf{\beta} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
   $$

   Here, $ \mathbf{I} $ is the identity matrix, and $ \lambda \mathbf{I} $ is the regularization term that shrinks the coefficients.

4. **Choose the Regularization Parameter $ \lambda $**: The choice of $ \lambda $ is crucial. A larger $ \lambda $ increases the amount of shrinkage, leading to coefficients that are more biased but have lower variance. Cross-validation is often used to select an optimal $ \lambda $.

The module should be named GPTRidgeRegression. 

The init function's parameter should contain alpha, which is is the parameter to balance the loss and L2 loss.

The module must contain a fit function and a predict function.  

The fit function accepts X_train, y_train and alpha as input and return None where  

    X_train: the features of the train data, which is a numpy array, and the shape of X_train is [N, d]. N is the number of the train data and d is the dimension.  
    y_train: the labels pf the train data,which is a numpy array.  
The predict function accepts X_test as input and return predictions where  

    X_test: the features of the test data, which is a numpy array, and the shape of X_train is [N, d]. N is the number of the test data and d is the dimension.  
    predctions: the predicted results for X_test, which is a numpy arrary.  
You should just return the code for the module, don't return anything else.