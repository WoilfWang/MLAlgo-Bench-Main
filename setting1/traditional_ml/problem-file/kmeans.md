Implement a Kmeans algorithm for clustering with python, numpy and scipy.  

The K-means algorithm aims to partition a set of n data points into K clusters, where each data point belongs to the cluster with the nearest mean. The "mean" here refers to the centroid of the cluster, which is the average of all the points in that cluster.

### Algorithmic Flow

1. **Initialization**: 
   - Choose K initial centroids randomly from the dataset. These centroids can be selected randomly or by using methods like K-means++ to improve convergence.

2. **Assignment Step**:
   - Assign each data point $ x_i $ to the nearest centroid based on the Euclidean distance. Mathematically, for each data point $ x_i $, find the cluster $ C_j $ such that:
     $$
     C_j = \arg\min_{k} \| x_i - \mu_k \|^2
     $$
     where $ \mu_k $ is the centroid of cluster $ k $.

3. **Update Step**:
   - Recalculate the centroids of the clusters by taking the mean of all data points assigned to each cluster. For each cluster $ C_j $, update the centroid $ \mu_j $ as:
     $$
     \mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
     $$
     where $ |C_j| $ is the number of points in cluster $ C_j $.

4. **Convergence Check**:
   - Repeat the assignment and update steps until the centroids no longer change significantly, or a maximum number of iterations is reached. Convergence can be determined by checking if the centroids' positions have stabilized or if the change in the cost function (sum of squared distances) is below a threshold.

The module should be named GPTKmeans.  

The init function should include the following parameters:

      n_clusters: The number of clusters to form as well as the number of centroids to generate.
The module must contain a fit_predict function.  
The fit_predict function accepts X as input and return labels  where  

      X: X is the features of the data, which is a numpy array and it's shape is [N, d]. N is the number of the train data and d is the dimension.  
      labels: A numpy array of shape (n_samples,) containing the index of the cluster each sample belongs to.
You should just return the code for the module, don't return anything else.