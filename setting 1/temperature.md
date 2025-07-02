You should implement temperature sampling using python and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


Temperature sampling is to sample from language models with temperature. Here we sample from the following probability distribution:
$$
P(x_i=V_l | x_{1:i-1}) = \frac{\exp(\frac{u_l}{T})}{\sum_j \exp(\frac{u_j}{T})}
$$
where $V$ is the vocabulary, $u_{1:|V|}$ are the logits of the distribution and T is the temperature. 

When $T = 1$, it degenerates into normal random sampling.
The module should be named TemperatureSampler.

This module should inherit from Sampler module. You can code with "from labml_nn.sampling import Sampler" to import that module.

The init function needs to include the following parameters:
temperature: the temperature to sample with. The default value is 1.0;

The module needs to include at least the following functions:

1. `__call__`: function to sample. It should include the following parameters:
   logits: the tensors that store the distribution of logits.
   The return of `__call__` function includes:
   res: the results of temperature sampling.

You just need to implement the algorithm module; no need to provide corresponding examples.