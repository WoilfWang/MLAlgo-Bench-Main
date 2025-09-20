You should implement the Low-Rank Adaptation using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 

Low-Rank Adaptation (LoRA) freezes pre-trained model weights and injects trainable rank decomposition matrices into each layer of the transformer. This makes it possible to efficiently fine-tune large langauge models by reducing trainable parameters by a large factor.

You should implement two module: LoRA Linear Layer and LoRA Embedding Layer.

## LoRA Linear Layer

LoRA linear layer adds a low-rank decomposition to the pre-trained
    weight matrix ($W_0 \in \mathbb{R}^{d \times k}$)
    of the linear layer.

$$W_0 + \Delta W = W_0 + BA$$

, where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$,
     and the rank $r \ll min(d, k)$.

All parameters are frozen except $A$ and $B$.

$\Delta W$ is initialized to be zero at the beginning of the training.

They multiple $x \Delta W^T$ by $\frac{\alpha}{r}$ where $\alpha$ is a hyper-parameter.
    Once $\alpha$ is tuned it can be kept the same when varying $r$.

As for LoRA Linear Layer, the module should be named Linear.
The init function need to include the following parameters:
in_features: in_features is the number of input features of the linear layer.
in_features: out_features is the number of output features of the linear layer.
bias: bias is a flag indicating if there is a bias parameter.
r: r is the rank of the decomposition r.
alpha: alpha is the scaling factor α. The default value of alpha is None. If alpha is not provided, should set alpha=r. 

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
x: x is the input tensor.

It should return:
result: the result of LoRA Linear Layer Operation.

## LoRA Embedding Layer

Similar to LoRA linear layer this adds a low-rank decomposition to the pre-trained
    embedding weights matrix ($W_0 \in \mathbb{R}^{d \times k}$).

$$W_0 + \Delta W = W_0 + BA$$

As for LoRA Embedding Layer, the module should be named Embedding.
The init function need to include the following parameters:
num_embeddings: num_embeddings is the number of embeddings.
embedding_dim: embedding_dim is the number embedding dimensions.
r: r is the rank of the decomposition r.
alpha: alpha is the scaling factor α. The default value of alpha is None. If alpha is not provided, should set alpha=r. 

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
x: x is the input tensor.

It should return:
result: the result of LoRA Embedding Layer Operation.


You just need to implement the algorithm module; no need to provide corresponding examples.