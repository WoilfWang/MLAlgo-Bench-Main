You should implement the Feedback Attention of the Feedback Transformer using python, numpy, and pytorch from scratch.
Normal transformers process tokens in parallel. Each transformer layer pays attention
to the outputs of the previous layer.
Feedback transformer pays attention to the output of all layers in previous steps.
So this adds recurrence, and we need to process token-by-token.
This slows down the training significantly (about 5X - 10X depending on the sequence length).
However, when predicting Feedback Transformer is faster because you can predict the next token
if you cache the memory vectors.

In order to speed up the training, the paper discusses starting with a short sequence length and gradually increasing it. They also discuss using a pretrained parallel transformer as the starting point.

The original feedback transformer doesn't keep the outputs of all layers. Instead it keeps weighted sum of the output of all layers. This reduces the memory used for caching during prediction. The first half of this file implements this.

The updated feedback transformer shares weights $W^l_k$ and $W^l_v$ used to calculate keys and values among the layers. We then calculate the keys and values for each step only once and keep
them cached. When geting attention scores, 
### Get attention scores
We use relative positional encodings for attention, similar to [relative multi-head attention form Transformer-XL paper](../relative_mha.html).

Attention from current step's query to key in step $j$ (relative to current step) is,
$$
\begin{align}
        A_{j} &= Q^\top K_j \\
            &= lin_q(X^q + P_q)^\top lin_k(X^k_j + P_j) \\
            &= (Q + U^Q)^\top(K_j + U^K_j) \\
            &= \underset{\textcolor{lightgreen}{A}}{Q^\top K_j} +
               \underset{\textcolor{lightgreen}{B}}{Q^\top U^K_j} +
               \underset{\textcolor{lightgreen}{C}}{{U^Q}^\top K_j} +
               \underset{\textcolor{lightgreen}{D}}{{U^Q}^\top U^K_j}
\end{align}
$$

where $Q, K_j$, are linear transformations of original embeddings $X^q, X^k_j$ and $U^Q, U^K_j$ are linear transformations of positional encodings $P_q, P_j$. We replace term $\textcolor{lightgreen}{D}$ with $S_j$.

The module should be named FeedbackAttention.

The init function needs to include the following parameters:

    heads: heads is the number of attention heads;
    d_model: d_model is the number of features in the transformer;
    dropout_prob: dropout_prob is the attention dropout probability;
    is_kv_precomputed: iss_kv_precomputed is whether key, value tensors are already calculated. The default value is False.

The model needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:

    query: the tensors that store collection of query. query has shape [batch_size, d_model]；
    value: the tensors that store collection of value. It has the shape `[seq_len, batch_size, d_model]`;
    key: the tensors that store collection of key. It has the shape `[seq_len, batch_size, d_model]`.
The return of the function should include:

    result: The result of the feedback attention.

You just need to implement the algorithm module; no need to provide corresponding examples, and no need to output any other content