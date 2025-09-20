You should implement the Fast weights attention using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 
 
The module should be named FastWeightsAttention.
The paper Linear Transformers Are Secretly Fast Weight Memory Systems in PyTorch finds similarities between linear self-attention and fast weight systems and makes modifications to self-attention update rule based on that. It also introduces a simpler, yet effective kernel function.
## Linear self-attention

Original transformer self-attention is, (omitting $\frac{1}{d_k}$ for clarity)

\begin{align}
y^{(i)} &= \Big[v^{(1)}, v^{(2)}, ..., v^{(i)}\Big] \text{softmax}
 \bigg(
    \Big[k^{(1)}, k^{(2)}, ..., k^{(i)}\Big] ^\top
    q^{(i)}
 \bigg) \\
 &= \sum^i_{j=1} \frac
 { v^{(j)} \kappa(k^{(j)}, q^{(i)}) }
 { \sum^i_{j'=1} \kappa(k^{(j')}, q^{(i)}) } \\
\end{align}

where $\kappa(k, q) = \text{exp}(k \cdot q)$

The idea behind linearizing self attention is to replace softmax kernel $\kappa$ with a different kernel $\kappa '$ so that we can calculate the denominator of the self attention function faster:
$$\kappa '(k, q) = \textcolor{lightgreen}{\phi(k)}^\top \textcolor{lightgreen}{\phi(q)}$$

This gives

\begin{align}
y^{(i)} &= \frac
 {\Big( \sum^i_{j=1} v^{(j)} \otimes \textcolor{lightgreen}{\phi(k^{(j)})} \Big)
  \textcolor{lightgreen}{\phi(q^{(i)})} }
 { \Big( \sum^i_{j'=1}
   \textcolor{lightgreen}{\phi(k^{(j')})} \Big)
    \textcolor{lightgreen}{\phi(q^{(i)})} }
\end{align}

With $\textcolor{cyan}{W^{(i)}} = \sum^i_{j=1} v^{(j)} \otimes \phi(k^{(j)})$ and $z^{(i)} = \sum^i_{j=1} \textcolor{lightgreen}{\phi(k^{(j)})}$, we can calculate them efficiently:

\begin{align}
\textcolor{cyan}{W^{(i)}} &= \textcolor{cyan}{W^{(i-1)}} + v^{(i)} \otimes \textcolor{lightgreen}{\phi(k^{(i)})} \\
z^{(i)} &= z{(i)} + \textcolor{lightgreen}{\phi(k^{(i)})} \\
y^{(i)} &= \frac{1}{z^{(i)} \cdot \textcolor{lightgreen}{\phi(q^{(i)})}}
    W^{(i)} \textcolor{lightgreen}{\phi(q^{(i)})}
\end{align}

This is quite similar to fast weights.

The paper introduces a new linear attention projection function $\textcolor{lightgreen}{\phi}$ a new update rule for $\textcolor{cyan}{W^{(i)}} = f(\textcolor{cyan}{W^{(i-1)}})$ and change the normalization $\frac{1}{z^{(i)} \cdot \textcolor{lightgreen}{\phi(q^{(i)})}}$

You should first the module of Deterministic Parameter Free Project (DPFP). The module should be named DPFP(nn.Module).
## Deterministic Parameter Free Project (DPFP)

This is the new projection function $\textcolor{lightgreen}{\phi}$ introduced in the paper.  DPFP projects $k$ of dimensionality $d_{key}$ to dimensionality $d_{dot} = 2 d_{key} \nu$, where $\nu \in \\{1, 2, ..., 2 d_{key} - 1 \\}$ is a hyper-parameter.
    $$\textcolor{lightgreen}{\phi_{2 d_{key} (i - 1)  + j}(k)}
     = \text{ReLU}\Big(\big[k, -k\big]\Big)_{j}\text{ReLU}\Big(\big[k, -k\big]\Big)_{i + j}$$

where $\big[k, -k\big]$ is the concatenation of $k$ and $-k$ to give a vector of size $2 d_{key}$, $i \in \\{1, 2, ..., \nu \\}$, and $j \in \\{1, 2, ..., 2 d_{key}\\}$. $x_i$ is the $i$-th element of vector $x$ and is rolled around if $i$ is larger than the number of elements in $x$.

Basically, it creates a new vector by multiplying elements of $[k, -k]$ shifted by $i$.

This produces projections that are sparse (only a few elements of $phi$ are non-zero) and orthogonal ($\textcolor{lightgreen}{\phi(k^{(i)})} \cdot \textcolor{lightgreen}{\phi(k^{(j)})}  \approx 0$ for most $i, j$  unless $k^{(i)}$ and $k^{(j)}$ are very similar.

### Normalization

Paper introduces a simple normalization for $\textcolor{lightgreen}{\phi}$,
    $$\textcolor{lightgreen}{\phi '(k)} =
     \frac{\textcolor{lightgreen}{\phi(k)}}{\sum^{d_{dot}}_{j=1} \textcolor{lightgreen}{\phi(k)_j}}$$
The init function should include the following parameters:
nu: nu is the hyper-parameter $\nu$. The deflault value is 1;
eps: eps is the small value used to make sure there is no division-by-zero when normalizing. The deflault is 1e-6.
The module needs to include at least the following functions:
1. forward. It should include the following parameters:
k: the input tensor.
The return of the function are:
result: the result of the deterministic parameter free project.

Then implement the module of FastWeightsAttention.
## Fast Weights Attention

The paper introduces a new update rule for calculating $\textcolor{cyan}{W^{(i)}}$.
    The model first retrieves the current value
    $\bar{v}^{(i)}$ paired with the key $k^{(i)}$.
    Then stores a combination $v^{(i)}_{new}$
    of the retrieved value $\bar{v}^{(i)}$ and the input $v^{(i)}$.

\begin{align}
    k^{(i)}, v^{(i)}, q^{(i)} &=
     \textcolor{orange}{W_k} x^{(i)}, \textcolor{orange}{W_v} x^{(i)}, \textcolor{orange}{W_q} x^{(i)} \\
    \bar{v}^{(i)} &= \textcolor{cyan}{W^{(i-1)}} \textcolor{lightgreen}{\phi'(k^{(i)})} \\
    \beta^{(i)} &= \sigma \Big(\textcolor{orange}{W_\beta} x^{(i)} \Big) \\
    v^{(i)}_{new} &= \beta^{(i)} v^{(i)} + \Big(1 - \beta^{(i)} \Big) \bar{v}^{(i)} \\
    \textcolor{cyan}{W^{(i)}}
     &= \textcolor{cyan}{W^{(i-1)}} + v^{(i)}_{new} \otimes \textcolor{lightgreen}{\phi'(k^{(i)})} \\
     &= \textcolor{cyan}{W^{(i-1)}} +
     \beta^{(i)} \Big( v^{(i)} - \bar{v}^{(i)} \Big ) \otimes \textcolor{lightgreen}{\phi'(k^{(i)})} \\
    y^{(i)} &= \textcolor{cyan}{W^{(i)}} \textcolor{lightgreen}{\phi'(q^{(i)})}
    \end{align}

where $\textcolor{orange}{W_\beta}$ is a trainable parameter and $\sigma$ is the sigmoid function.

Note that we don't need the normalization term $z$ because $\textcolor{lightgreen}{\phi'}$ is normalized.
The init function needs to include the following parameters:
heads: the nums of the attention head;
d_model: d_model is the number of features in the query , key and value vectors; 
phi: the object of the module DPFP.
dropout_prob: the proportion of neuron dropout. The default value is 0.1;

The module needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
x: the input tensor.
It should return:
result: the result of the module.

You just need to implement the algorithm module; no need to provide corresponding examples