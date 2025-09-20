You should implement the Wasserstein GAN (WGAN) using Python and PyTorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


The original GAN loss is based on Jensen-Shannon (JS) divergence
between the real distribution $\mathbb{P}_r$ and generated distribution $\mathbb{P}_g$.
The Wasserstein GAN is based on Earth Mover distance between these distributions.

$$
W(\mathbb{P}_r, \mathbb{P}_g) =
 \underset{\gamma \in \Pi(\mathbb{P}_r, \mathbb{P}_g)} {\mathrm{inf}}
 \mathbb{E}_{(x,y) \sim \gamma}
 \Vert x - y \Vert
$$

$\Pi(\mathbb{P}_r, \mathbb{P}_g)$ is the set of all joint distributions, whose
marginal probabilities are $\gamma(x, y)$.

$\mathbb{E}_{(x,y) \sim \gamma} \Vert x - y \Vert$ is the earth mover distance for
a given joint distribution ($x$ and $y$ are probabilities).

So $W(\mathbb{P}_r, \mathbb{P}_g)$ is equal to the least earth mover distance for
any joint distribution between the real distribution $\mathbb{P}_r$ and generated distribution $\mathbb{P}_g$.

The paper shows that Jensen-Shannon (JS) divergence and other measures for the difference between two probability
distributions are not smooth. And therefore if we are doing gradient descent on one of the probability
distributions (parameterized) it will not converge.

Based on Kantorovich-Rubinstein duality,
$$
W(\mathbb{P}_r, \mathbb{P}_g) =
 \underset{\Vert f \Vert_L \le 1} {\mathrm{sup}}
 \mathbb{E}_{x \sim \mathbb{P}_r} [f(x)]- \mathbb{E}_{x \sim \mathbb{P}_g} [f(x)]
$$

where $\Vert f \Vert_L \le 1$ are all 1-Lipschitz functions.

That is, it is equal to the greatest difference
$$\mathbb{E}_{x \sim \mathbb{P}_r} [f(x)] - \mathbb{E}_{x \sim \mathbb{P}_g} [f(x)]$$
among all 1-Lipschitz functions.

For $K$-Lipschitz functions,
$$
W(\mathbb{P}_r, \mathbb{P}_g) =
 \underset{\Vert f \Vert_L \le K} {\mathrm{sup}}
 \mathbb{E}_{x \sim \mathbb{P}_r} \Bigg[\frac{1}{K} f(x) \Bigg]
  - \mathbb{E}_{x \sim \mathbb{P}_g} \Bigg[\frac{1}{K} f(x) \Bigg]
$$

If all $K$-Lipschitz functions can be represented as $f_w$ where $f$ is parameterized by
$w \in \mathcal{W}$,

$$
K \cdot W(\mathbb{P}_r, \mathbb{P}_g) =
 \max_{w \in \mathcal{W}}
 \mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{x \sim \mathbb{P}_g} [f_w(x)]
$$

If $(\mathbb{P}_{g})$ is represented by a generator $$g_\theta (z)$$ and $z$ is from a known
distribution $z \sim p(z)$,

$$
K \cdot W(\mathbb{P}_r, \mathbb{P}_\theta) =
 \max_{w \in \mathcal{W}}
 \mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]
$$

Now to converge $g_\theta$ with $\mathbb{P}_{r}$ we can gradient descent on $\theta$
to minimize above formula.

Similarly we can find $\max_{w \in \mathcal{W}}$ by ascending on $w$,
while keeping $K$ bounded. *One way to keep $K$ bounded is to clip all weights in the neural
network that defines $f$ clipped within a range.*

The model needs to include at least the following classes:
1.Discriminator Loss.
The module should be named DiscriminatorLoss.
We want to find w to maximize $\mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]$,  so we minimize, $-\frac{1}{m} \sum_{i=1}^m f_w \big(x^{(i)} \big) +\frac{1}{m} \sum_{i=1}^m f_w \big( g_\theta(z^{(i)}) \big)$
The forward function needs to include the following parameters:
f_real: tensors, is $f_w(x)$. The shape of [batch_size, 1];
f_fake: tensors, is $f_w(g_\theta(z))$. The shape of [batch_size, 1].
This returns the a tuple with losses for  $f_w(x)$  and $f_w(g_\theta(z)) $, which are later added. They are kept separate for logging.
2.GeneratorLoss
We want to find  $\theta$  to minimize $\mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))] $ , The first component is independent of  $\theta $,
 so we minimize,$ -\frac{1}{m} \sum_{i=1}^m f_w \big( g_\theta(z^{(i)}) \big)  $
The forward function needs to include the following parameter:
f_fake: is $f_w(g_\theta(z))$. The shape of [batch_size, 1].
The return of forward function includes:
A single tensor representing the loss for the generator:-f_fake.mean()
You just need to implement the algorithm module; no need to provide corresponding examples.