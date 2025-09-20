You should implement the AFT Local Operation of the Attention Free Transformer using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 
The module replaces the self-attention layer with a new efficient operation, that has memory complexity of $\mathcal{O}(Td)$, where $T$ is the sequence length and $d$ is the dimensionality of embeddings. 
The paper introduces AFT along with AFT-local and AFT-conv.
Here we have implemented AFT-local which pays attention to closeby tokens in an autoregressive model.

## Attention Free Transformer

AFT (similar to [MHA](../mha.html)) first transforms the embeddings $X$ into query $Q = XW^Q$, key $K = XW^K$ and value $V = XW^V$ tensors with learned weights. The output for each position $t \in [1, T]$ is calculated with the following operation.

$$Y_t = \sigma(Q_t) \odot
 \frac{\sum_{t'=1}^T \exp(K_{t'} + w_{t,t'}) \odot V_{t'}}
 {\sum_{t'=1}^T \exp(K_{t'} + w_{t,t'})}$$
, where $\odot$ is element-wise product, $\sigma$ is a non-linearity (sigmoid) and $w \in \mathbb{R}^{T \times T}$ is a learned matrix of pair-wise position biases.

This means that we take the weighted average of values and multiply them by the query. This eliminates the need to calculate the $T \times T$ attention matrix that [MHA](../mha.html) requires, and therefore reduce the memory requirement.

## AFT Local
AFT Local only apply learned pair-wise position biases locally:
\begin{align}
w'_{t,t'} =
\begin{cases}
w_{t,t'},  & {\text{for } \lvert t-t' \rvert \lt s} \\
0, & \text{otherwise}
\end{cases}
\end{align}
, where $s \le T$ is the local window size.

Although $w'_{t,t'}$ is $0$ outside the local window the AFT operation still uses key-value pairs from other areas. This is different from local transformers where embeddings outside the local window are completely not visible.

### AFT Local Operation

$$Y_t = \sigma(Q_t) \odot
     \frac{\sum_{t'=1}^T \exp(K_{t'} + w_{t,t'}) \odot V_{t'}}
     {\sum_{t'=1}^T \exp(K_{t'} + w_{t,t'})}$$
where, 
\begin{align}
    w'_{t,t'} =
    \begin{cases}
    w_{t,t'},  & {\text{for } \lvert t-t' \rvert \lt s} \\
    0, & \text{otherwise}
    \end{cases}
\end{align}

The module should be named AFTLocal.
The init function needs to include the following parameters:
d_model: d_model is the number of features in the query , key and value vectors;
seq_len: seq_len is T;
local_window_size: local_window_size is the local window size s.

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
query: the tensors that store collection of query. It has the shape `[seq_len, batch_size, d_model]`;
value: the tensors that store collection of value. It has the shape `[seq_len, batch_size, d_model]`;
key: the tensors that store collection of key. It has the shape `[seq_len, batch_size, d_model]`;
mask: `mask` has shape `[seq_len, seq_len, batch_size]` and `mask[i, j, b]` indicates whether for batch `b` , query at position `i` has access to key-value at position `j`. The default value is None.

It should return:
result: the result of AFT Local Operation.

You just need to implement the algorithm module; no need to provide corresponding examples