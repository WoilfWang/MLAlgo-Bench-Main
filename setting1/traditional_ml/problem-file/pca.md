Implement the principal component analysis algorithm for dimensionality reduction with python, numpy and scipy.

The main idea behind PCA is to identify the directions (principal components) along which the variation in the data is maximized. These directions are orthogonal to each other and are determined by the eigenvectors of the covariance matrix of the data.

### Algorithmic Flow of PCA

1. **Standardization**: 
   - Center the data by subtracting the mean of each variable from the dataset to ensure that the PCA results are not biased by the scale of the variables.
   - Optionally, scale the data to have unit variance if the variables are measured in different units.

2. **Covariance Matrix Computation**:
   - Compute the covariance matrix $ \mathbf{C} $ of the standardized data. If $ \mathbf{X} $ is the data matrix with $ n $ samples and $ p $ features, the covariance matrix is given by:
     $$
     \mathbf{C} = \frac{1}{n-1} \mathbf{X}^T \mathbf{X}
     $$

3. **Eigenvalue Decomposition**:
   - Perform eigenvalue decomposition on the covariance matrix $ \mathbf{C} $ to find its eigenvalues and eigenvectors. The eigenvectors represent the directions of maximum variance (principal components), and the eigenvalues indicate the magnitude of variance in these directions.

4. **Sort Eigenvectors**:
   - Sort the eigenvectors by their corresponding eigenvalues in descending order. The eigenvector with the highest eigenvalue is the first principal component, the one with the second highest is the second principal component, and so on.

5. **Select Principal Components**:
   - Choose the top $ k $ eigenvectors to form a matrix $ \mathbf{W} $ that will be used to transform the data. The choice of $ k $ depends on the desired level of variance to be retained (often a cumulative variance threshold like 95%).

6. **Transform the Data**:
   - Project the original data onto the new feature space using the matrix $ \mathbf{W} $:
     $$
     \mathbf{Z} = \mathbf{X} \mathbf{W}
     $$
   - Here, $ \mathbf{Z} $ is the transformed dataset with reduced dimensions.

The module should be named GPTPCA.
The init function should include the following parameters:

      n_components: Number of components to keep.
The module must contain a fit_transform function, which is used for fitting data and performing dimensionality reduction transformations.

The fit_transform function accepts X as input and return reduced_X where

      X: X is the features of the data, which is a numpy array and it's shape is [N, d]. N is the number of the train data and d is the dimension.
      reduced_X: reduced_X is the reduced features after dimensionality reduction. The shape should be [N, low_d], where N is the num of the data and low_d is the reduced dimension.
You should just return the code for the module, don't return anything else.