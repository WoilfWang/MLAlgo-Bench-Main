You should implement multi-headed attention using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


 ## Multi-Head Attention Module

This computes scaled multi-headed attention for given `query`, `key` and `value` vectors.

$$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$

In simple terms, it finds keys that matches the query, and gets the values of
     those keys.

It uses dot-product of query and key as the indicator of how matching they are.
    Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
    This is done to avoid large dot-product values causing softmax to
    give very small gradients when $d_k$ is large.

Softmax is calculated along the axis of of the sequence (or time).

The module should be named MultiHeadAttention.

The init function needs to include the following parameters:
heads: the nums of the attention head;
d_model: d_model is the number of features in the query , key and value vectors;
dropout_prob: the proportion of neuron dropout. The default value is 0.1.

The model needs to include at least the following functions:
1. forward: forward propagation function. It should include the following parameters:
   query: the tensors that store collection of query. It has the shape `[seq_len, batch_size, d_model]`;
   value: the tensors that store collection of value. It has the shape `[seq_len, batch_size, d_model]`;
   key: the tensors that store collection of key. It has the shape `[seq_len, batch_size, d_model]`;
   mask: `mask` has shape `[seq_len, seq_len, batch_size]` and `mask[i, j, b]` indicates whether for batch `b` , query at position `i` has access to key-value at position `j`.
   The return of forward function includes:
   x: the results of the multi-head attention

You just need to implement the algorithm module; no need to provide corresponding examples