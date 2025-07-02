You should implement FNet using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

FNet replaces the self-attention layer with two Fourier transforms to mix tokens. This is a 7× more efficient than self-attention.
## Mixing tokens with two Fourier transforms

We apply Fourier transform along the hidden dimension (embedding dimension) and then along the sequence dimension.
$$
\mathcal{R}\big(\mathcal{F}_\text{seq} \big(\mathcal{F}_\text{hidden} (x) \big) \big)
$$

where $x$ is the embedding input, $\mathcal{F}$ stands for the fourier transform and
$\mathcal{R}$ stands for the real component in complex numbers.

This is very simple to implement on PyTorch - just 1 line of code. The paper suggests using a precomputed DFT matrix and doing matrix multiplication to get the Fourier transformation.

The module should be named FNetMix. 
This module simply implements
    $$
    \mathcal{R}\big(\mathcal{F}_\text{seq} \big(\mathcal{F}_\text{hidden} (x) \big) \big)
    $$
The init function don't include any parameters.

The module needs to include at least the following functions:
1. forward: forward propagation function. The [normal attention module](../mha.html) can be fed with different token embeddings for $\text{query}$,$\text{key}$, and $\text{value}$ and a mask.  We follow the same function signature so that we can replace it directly.  For FNet mixing, $$x = \text{query} = \text{key} = \text{value}$$ and masking is not possible. Shape of `query` (and `key` and `value`) is `[seq_len, batch_size, d_model]`.
It should include the following parameters:
 query: the tensors that store collection of query. It has the shape `[seq_len, batch_size, d_model]`;
   value: the tensors that store collection of value. It has the shape `[seq_len, batch_size, d_model]`;
   key: the tensors that store collection of key. It has the shape `[seq_len, batch_size, d_model]`.
It should return:
result: the result of the FNet.

You just need to implement the algorithm module; no need to provide corresponding examples