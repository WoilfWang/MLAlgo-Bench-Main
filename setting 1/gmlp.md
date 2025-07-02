You should implement gMLP block using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

## gMLP Block 
Each block does the following transformations to input embeddings $X \in \mathbb{R}^{n \times d}$ where $n$ is the sequence length and $d$ is the dimensionality of the embeddings:
    \begin{align}
    Z &= \sigma(XU) \\
    \tilde{Z} &= s(Z) \\
    Y &= \tilde{Z}V \\
    \end{align}

where $V$ and $U$ are learnable projection weights. $s(\cdot)$ is the Spacial Gating Unit defined below. Output dimensionality of $s(\cdot)$ will be half of $Z$. $\sigma$ is an activation function such as [GeLU]

## Spatial Gating Unit
$$s(Z) = Z_1 \odot f_{W,b}(Z_2)$$
where $f_{W,b}(Z) = W Z + b$ is a linear transformation along the sequence dimension, and $\odot$ is element-wise multiplication. $Z$ is split into to parts of equal size $Z_1$ and $Z_2$ along the channel dimension (embedding dimension).

The module should be named GMLPBlock.
The init function needs to include the following parameters:
d_model: d_model is the dimensionality (d) of X;
d_ffn: d_ffn is the dimensionality of Z;
seq_len: seq_len is the length of the token sequence (n).

It has a class variable size, and self.size is assigned the value of d_model.

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
x: x is the input embedding tensor X of shape [seq_len, batch_size, d_model];
mask: mask is a boolean mask of shape [seq_len, seq_len, 1] that controls the visibility of tokens among each other.

It should return:
x: the results of the gMLP block.

You just need to implement the algorithm module; no need to provide corresponding examples.