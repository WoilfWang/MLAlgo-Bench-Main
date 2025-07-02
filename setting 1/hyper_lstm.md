You should implement HyperNetworks - HyperLSTM using Python and PyTorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


HyperNetworks use a smaller network to generate weights of a larger network.
There are two variants: static hyper-networks and dynamic hyper-networks.
Static HyperNetworks have smaller networks that generate weights (kernels)
of a convolutional network. Dynamic HyperNetworks generate parameters of a
recurrent neural network
for each step. This is an implementation of the latter.

## Dynamic HyperNetworks
In a RNN the parameters stay constant for each step.
Dynamic HyperNetworks generate different parameters for each step.
HyperLSTM has the structure of a LSTM but the parameters of
each step are changed by a smaller LSTM network.

In the basic form, a Dynamic HyperNetwork has a smaller recurrent network that generates
a feature vector corresponding to each parameter tensor of the larger recurrent network.
Let's say the larger network has some parameter $\textcolor{cyan}{W_h}$ the smaller network generates a feature
vector $z_h$ and we dynamically compute $\textcolor{cyan}{W_h}$ as a linear transformation of $z_h$.
For instance $\textcolor{cyan}{W_h} =  \langle W_{hz}, z_h \rangle$ where
$W_{hz}$ is a 3-d tensor parameter and $\langle . \rangle$ is a tensor-vector multiplication.
$z_h$ is usually a linear transformation of the output of the smaller recurrent network.

### Weight scaling instead of computing

Large recurrent networks have large dynamically computed parameters.
These are calculated using linear transformation of feature vector $z$.
And this transformation requires an even larger weight tensor.
That is, when $\textcolor{cyan}{W_h}$ has shape $N_h \times N_h$,
$W_{hz}$ will be $N_h \times N_h \times N_z$.

To overcome this, we compute the weight parameters of the recurrent network by
dynamically scaling each row of a matrix of same size.

\begin{align}
d(z) = W_{hz} z_h \\
\\
\textcolor{cyan}{W_h} =
\begin{pmatrix}
d_0(z) W_{hd_0} \\
d_1(z) W_{hd_1} \\
... \\
d_{N_h}(z) W_{hd_{N_h}} \\
\end{pmatrix}
\end{align}

where $W_{hd}$ is a $N_h \times N_h$ parameter matrix.

We can further optimize this when we compute $\textcolor{cyan}{W_h} h$,
as
$$\textcolor{lightgreen}{d(z) \odot (W_{hd} h)}$$
where $\odot$ stands for element-wise multiplication.

The model needs to include at least the following classes:
1.class HyperLSTM.
The module should be named HyperLSTM
The init function needs to include the following parameters:
input_size (int): Size of the input feature vector;
hidden_size (int): Size of the hidden state of the main LSTM in each layer;
hyper_size (int): is the size of the smaller LSTM that alters the weights of the larger outer LSTM;
n_z (int): is the size of the feature vectors used to alter the LSTM weights;
n_layers (int): Number of HyperLSTM layers.

The forward function(processes the input sequence through the multi-layer HyperLSTM network.)needs to include the following parameters:
x (torch.Tensor): Input tensor with shape [n_steps, batch_size, input_size];
state (Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], optional): Tuple  containing initial hidden, cell, and hypernetwork states. If None, states are initialized to  zeros.
h and c: Hidden and cell states of the main LSTM with shape [batch_size, hidden_size].
h_hat and c_hat: Hidden and cell states of the hypernetwork LSTM with shape [batch_size,  hyper_size].

The return of forward function includes:
out (torch.Tensor): Output tensor with shape [n_steps, batch_size, hidden_size].
state (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing the  final hidden, cell, and hypernetwork states.

You just need to implement the algorithm module; no need to provide corresponding examples.