You should implement the Relative Multi-Headed Attention using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


### Get relative attention scores

   With absolute attention

   \begin{align}
        A^{abs}_{j} &= lin_q(X^q_i + P_i)^\top lin_k(X^k_j + P_j) \\
                      &= \underset{\textcolor{lightgreen}{A}}{Q_i^\top K_j} +
                         \underset{\textcolor{lightgreen}{B}}{Q_i^\top U^K_j} +
                         \underset{\textcolor{lightgreen}{C}}{{U^Q_i}^\top K_j} +
                         \underset{\textcolor{lightgreen}{D}}{{U^Q_i}^\top U^K_j}
        \end{align}

   where $Q_i, K_j$, are linear transformations of
         original embeddings $X^q_i, X^k_j$
         and $U^Q_i, U^K_j$ are linear transformations of
         absolute positional encodings $P_i, P_j$.

   They reason out that the attention to a given key should be the same regardless of
        the position of query.
        Hence replace $\underset{\textcolor{lightgreen}{C}}{{U^Q_i}^\top K_j}$
        with a constant $\underset{\textcolor{lightgreen}{C}}{\textcolor{orange}{v^\top} K_j}$.

   For the second and third terms relative positional encodings are introduced.
        So $\underset{\textcolor{lightgreen}{B}}{Q_i^\top U^K_j}$ is
        replaced with $\underset{\textcolor{lightgreen}{B}}{Q_i^\top \textcolor{orange}{R_{i - j}}}$
        and $\underset{\textcolor{lightgreen}{D}}{{U^Q_i}^\top U^K_j}$
        with $\underset{\textcolor{lightgreen}{D}}{\textcolor{orange}{S_{i-j}}}$.

   \begin{align}
        A^{rel}_{i,j} &= \underset{\mathbf{\textcolor{lightgreen}{A}}}{Q_i^\top K_j} +
                         \underset{\mathbf{\textcolor{lightgreen}{B}}}{Q_i^\top \textcolor{orange}{R_{i - j}}} +
                         \underset{\mathbf{\textcolor{lightgreen}{C}}}{\textcolor{orange}{v^\top} K_j} +
                         \underset{\mathbf{\textcolor{lightgreen}{D}}}{\textcolor{orange}{S_{i-j}}}
        \end{align}

The module should be named RelativeMultiHeadAttention.

The init function needs to include the following parameters:
heads: the nums of the attention head;
d_model: d_model is the number of features in the query , key and value vectors;
dropout_prob: the proportion of neuron dropout. The default value is 0.1;

The model needs to include at least the following functions:
1. forward: forward propagation function. It should include the following parameters:
   query: the tensors that store collection of *query*. It has the shape `[seq_len, batch_size, d_model]`; 
   value: the tensors that store collection of *value.* It has the shape `[seq_len, batch_size, d_model]`;
   key: the tensors that store collection of *key.* It has the shape `[seq_len, batch_size, d_model]`;
   mask: `mask` has shape `[seq_len, seq_len, batch_size]` and `mask[i, j, b]` indicates whether for batch `b` , query at position `i` has access to key-value at position `j`.
   The return of forward function includes:
   x: the results of the relative multi-head attention

You just need to implement the algorithm module; no need to provide corresponding examples