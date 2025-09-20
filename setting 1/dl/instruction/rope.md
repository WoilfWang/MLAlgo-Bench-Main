You should implement the Rotary Positional Embeddings using python, numpy, and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


Rotary Positional Embeddings (RoPE) encode position information of tokens with a rotation matrix that naturally incorporates explicit relative position dependency.
RoPE module
    Rotary encoding transforms pairs of features by rotating in the 2D plane. That is, it organizes the $d$ features as $\frac{d}{2}$ pairs. Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it by an angle depending on the position of the token.
    ### For a pair of features
    Let $x^{(1)}_m$ and $x^{(2)}_m$ be two features of the  key or query of any head at position $m$. Or for simplicity assume $x$ has only two features. Then the transformation is,
    \begin{align}
    RoPE\big(x^{(1)}_m, x^{(2)}_m, m\big) &=
    \begin{pmatrix}
    \cos m \theta & - \sin m \theta \\
    \sin m \theta & \cos m \theta
    \end{pmatrix}
    \begin{pmatrix}
    x^{(1)}_m \\
    x^{(2)}_m \\
    \end{pmatrix} \\
    &=
    \begin{pmatrix}
    x^{(1)}_m \cos m\theta - x^{(2)}_m \sin m \theta \\
    x^{(2)}_m \cos m\theta + x^{(1)}_m \sin m \theta \\
    \end{pmatrix} \\
    \end{align}
    where $\theta$ is a constant angle. The other pairs of features are transformed similarly.

### Attention is relative
For a pair of features, dot-product attention score between two positions $m$ and $n$ would be

\begin{align}
    \Big \langle RoPE\big(x^{(1)}_m, x^{(2)}_m, m\big),  RoPE\big(x^{(1)}_n, x^{(2)}_n, n\big) \Big \rangle &= \\
    (x^{(1)}_m \cos m\theta - x^{(2)}_m \sin m \theta)(x^{(1)}_n \cos n\theta - x^{(2)}_n \sin n \theta) &+ \\
    (x^{(2)}_m \cos m\theta + x^{(1)}_m \sin m \theta)(x^{(2)}_n \cos n\theta + x^{(1)}_n \sin n \theta) &= \\
    x^{(1)}_m x^{(1)}_n (\cos m\theta \cos n\theta + \sin m \theta \sin n \theta) &+ \\
    x^{(1)}_m x^{(2)}_n (-\cos m\theta \sin n\theta + \sin m \theta \cos n \theta) &+ \\
    x^{(2)}_m x^{(1)}_n (-\sin m\theta \cos n\theta + \cos m \theta \sin n \theta) &+ \\
    x^{(2)}_m x^{(2)}_n (\sin m\theta \sin n\theta + \cos m \theta \cos n \theta) &= \\
    x^{(1)}_m x^{(1)}_n \cos (m - n) \theta +
    x^{(1)}_m x^{(2)}_n \sin(m - n) \theta &+ \\
    - x^{(2)}_m x^{(1)}_n \sin (m - n) \theta +
    x^{(2)}_m x^{(2)}_n \cos (m - n) \theta &= \\
    \big(x^{(1)}_m \cos (m - n)\theta - x^{(2)}_m \sin (m - n) \theta\big) x^{(1)}_n &+ \\
    \big(x^{(2)}_m \cos (m - n)m\theta + x^{(1)}_m \sin (m - n) \theta\big) x^{(2)}_n  &= \\

\Big \langle RoPE\big(x^{(1)}_m, x^{(2)}_m, m - n\big),  RoPE\big(x^{(1)}_n, x^{(2)}_n, 0\big) \Big \rangle
    \end{align}
This shows that for dot-production attention the rotary encodings gives relative attention.

### For all features

The features are grouped into pairs and handled as above. They use a different $\theta$ for each pair.
    The paper suggests using $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$ for the $\frac{d}{2}$ pairs of features.
    We pair feature $i$ with feature $i + \frac{d}{2}$. So for position $m$ we transform

\begin{align}
    \begin{pmatrix}
    x^{(i)}_m \\
    x^{(i + \frac{d}{2})}_m
    \end{pmatrix}
    \end{align}

to

\begin{align}
    \begin{pmatrix}
    x^{(i)}_m \cos m \theta_i - x^{(i + \frac{d}{2})}_m \sin m \theta_i \\
    x^{(i + \frac{d}{2})}_m \cos m\theta_i + x^{(i)}_m \sin m \theta_i \\
    \end{pmatrix} \\
    \end{align}

The module should be named RotaryPositionalEmbeddings.

The init function needs to include the following parameters:
d: is the number of features $d$;
base: is the constant used for calculating $\Theta$.

The model needs to include at least the following functions:
1. forward: forward propagation function. 
It should include the following parameters:
x: x is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`.
The return of the function should include:
result: The result of the Rotary Positional Embeddings.

You just need to implement the algorithm module; no need to provide corresponding examples