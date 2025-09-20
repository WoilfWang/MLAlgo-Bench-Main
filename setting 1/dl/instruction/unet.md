You should implement U-Net model using python and pytorch from scratch. We provide the GPU that can be used for training or inference.
I will provide you with a description of the algorithm module, and you should implement it strictly according to this description. 


U-Net consists of a contracting path and an expansive path. The contracting path contains encoder layers that capture contextual information and reduce the spatial resolution of the input, while the expansive path contains decoder layers that decode the encoded data and use the information from the contracting path via skip connections to generate a segmentation map.The contracting path in U-Net is responsible for identifying the relevant features in the input image. The encoder layers perform convolutional operations that reduce the spatial resolution of the feature maps while increasing their depth, thereby capturing increasingly abstract representations of the input. On the other hand, the expansive path works on decoding the encoded data and locating the features while maintaining the spatial resolution of the input. The decoder layers in the expansive path perform convolutional operations that upsample the feature maps. The skip connections from the contracting path help to preserve the spatial information lost in the contracting path, which helps the decoder layers to locate the features more accurately.

The module should be named UNet.

The init function needs to include the following parameters:
n_channels: number of channels in the input image;
out_channels: number of channels in the result feature map.
The module needs to include at least the following functions:
1. forward: forward: forward propagation function. It should include the following parameters:
x: the tensors that store the input image. It has the shape [bs, 3, width, height]
The return of forward function includes:
x: the results of the U-Net model. It has the shape [bs, 1, width, height].

You just need to implement the algorithm module; no need to provide corresponding examples.