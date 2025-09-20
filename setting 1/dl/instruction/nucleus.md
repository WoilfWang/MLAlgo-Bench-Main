You should implement Nucleus Sampling using python and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


Nucleus sampling practically performs better than other sampling methods such as pure sampling, temperature sampling or top-k sampling for text generation.

Nucleus sampling first picks a subset of the vocabulary $V^{(p)} \subset V$, where $V^{(p)}$ is smallest set of tokens such that $\sum_{x_i \in V^{(p)}} P(x_i | x_{1:i-1}) \ge p$. That is, we pick the highest probable tokens until the sum of their probabilities is less that $p$.

Then we sample from the selected tokens.
The module should be named NucleusSampler.

This module should inherit from Sampler module. You can code with "from labml_nn.sampling import Sampler" to import that module.

The init function needs to include the following parameters:
p: the sum of probabilities of tokens to pick p;
sampler: the sampler to use for the selected tokens. It can be any sampler that takes a logits tensor as input and returns a token tensor;
The module needs to include at least the following functions:

1. `__call__`: function to sample. It should include the following parameters:
   logits: the tensors that store the distribution of logits.
   The return of `__call__` function includes:
   res: the results of nucleus sampling.
You just need to implement the algorithm module; no need to provide corresponding examples.