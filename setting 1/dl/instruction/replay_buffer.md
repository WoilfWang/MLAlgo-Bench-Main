You should implement Prioritized Experience Replay Buffer using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

The transitions are prioritized by the Temporal Difference error (td error), $\delta$.

We sample transition $i$ with probability,
    $$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$
    where $\alpha$ is a hyper-parameter that determines how much
    prioritization is used, with $\alpha = 0$ corresponding to uniform case.
    $p_i$ is the priority.

We use proportional prioritization $p_i = |\delta_i| + \epsilon$ where
    $\delta_i$ is the temporal difference for transition $i$.

We correct the bias introduced by prioritized replay using
     importance-sampling (IS) weights
    $$w_i = \bigg(\frac{1}{N} \frac{1}{P(i)}\bigg)^\beta$$ in the loss function.
    This fully compensates when $\beta = 1$.
    We normalize weights by $\frac{1}{\max_i w_i}$ for stability.
    Unbiased nature is most important towards the convergence at end of training.
    Therefore we increase $\beta$ towards end of training.

### Binary Segment Tree
We use a binary segment tree to efficiently calculate
    $\sum_k^i p_k^\alpha$, the cumulative probability,
    which is needed to sample.
    We also use a binary segment tree to find $\min p_i^\alpha$,
    which is needed for $\frac{1}{\max_i w_i}$.
    We can also use a min-heap for this.
    Binary Segment Tree lets us calculate these in $\mathcal{O}(\log n)$
    time, which is way more efficient that the naive $\mathcal{O}(n)$
    approach.

This is how a binary segment tree works for sum;
    it is similar for minimum.
    Let $x_i$ be the list of $N$ values we want to represent.
    Let $b_{i,j}$ be the $j^{\mathop{th}}$ node of the $i^{\mathop{th}}$ row
     in the binary tree.
    That is two children of node $b_{i,j}$ are $b_{i+1,2j}$ and $b_{i+1,2j + 1}$.

The leaf nodes on row $D = \left\lceil {1 + \log_2 N} \right\rceil$
     will have values of $x$.
    Every node keeps the sum of the two child nodes.
    That is, the root node keeps the sum of the entire array of values.
    The left and right children of the root node keep
     the sum of the first half of the array and
     the sum of the second half of the array, respectively.
    And so on...

$$b_{i,j} = \sum_{k = (j -1) * 2^{D - i} + 1}^{j * 2^{D - i}} x_k$$

Number of nodes in row $i$,
    $$N_i = \left\lceil{\frac{N}{D - i + 1}} \right\rceil$$
    This is equal to the sum of nodes in all rows above $i$.
    So we can use a single array $a$ to store the tree, where,
    $$b_{i,j} \rightarrow a_{N_i + j}$$

Then child nodes of $a_i$ are $a_{2i}$ and $a_{2i + 1}$.
    That is,
    $$a_i = a_{2i} + a_{2i + 1}$$

This way of maintaining binary trees is very easy to program.
    *Note that we are indexing starting from 1*.

We use the same structure to compute the minimum.

The module should be named ReplayBuffer.
The init function needs to include the following parameters:
capacity: The maximum number of experiences (transitions) that the buffer can store;
alpha: A hyperparameter that controls the degree of prioritization in experience replay. 

The module needs to include at least the following functions:
1. add: Add sample to queue.
It should include the following parameters:
obs: The current observation (state) of the agent;
action: The action taken by the agent in the current state;
reward: the reward received after taking the action in the given state;
next_obs: The next observation (state) the agent will observe after taking the action. This is typically the environment’s state after the action is executed;
done: A boolean flag indicating whether the episode has ended.
It return nothing.

2. find_prefix_sum_idx: Find largest $i$ such that $\sum_{k=1}^{i} p_k^\alpha  \le P$.
It should include the following parameters:
prefix_sum: A floating-point value representing the cumulative probability up to which we want to find the corresponding index. 
It should return:
idx: The index of the transition in the buffer that corresponds to the given cumulative probability (prefix sum). 

3. sample: Sample from buffer.
It should include the following parameters:
batch_size: An integer representing the number of experiences to sample from the buffer in one batch. It controls the size of the batch used for training;
beta: A floating-point value representing the importance-sampling exponent.
It should return: 
samples: samples(dictionary) is the samples data.

4. update_priorities: Update priorities.
It should include the following parameters:
indexes: These are the indices of the experiences in the buffer whose priorities need to be updated;
priorities: These are the new priority values for the experiences at the specified indices. 
It should return nothing. 

5. is_full: Whether the buffer is full.
is_full function has no input parameters.
It should return:
result: Whether the buffer is full.


You just need to implement the algorithm module; no need to provide corresponding examples