"""
---
title: Train ConvMixer on CIFAR 10
summary: >
  Train ConvMixer on CIFAR 10
---

#  Train a [ConvMixer](index.html) on CIFAR 10

This script trains a ConvMixer on CIFAR 10 dataset.

This is not an attempt to reproduce the results of the paper.
The paper uses  image augmentations
present in [PyTorch Image Models (timm)](https://github.com/rwightman/pytorch-image-models)
for training. We haven't done this for simplicity - which causes our validation accuracy to drop.
"""

from labml import experiment
from labml.configs import option
from labml_nn.experiments.cifar10 import CIFAR10Configs


class Configs(CIFAR10Configs):
    """
    ## Configurations

    We use [`CIFAR10Configs`](../experiments/cifar10.html) which defines all the
    dataset related configurations, optimizer, and a training loop.
    """

    # Size of a patch, $p$
    patch_size: int = 2
    # Number of channels in patch embeddings, $h$
    d_model: int = 256
    # Number of [ConvMixer layers](#ConvMixerLayer) or depth, $d$
    n_layers: int = 8
    # Kernel size of the depth-wise convolution, $k$
    kernel_size: int = 7
    # Number of classes in the task
    n_classes: int = 10


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--solution', default='golden')
args = parser.parse_args()

import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

if args.solution == 'golden':
    from labml_nn.conv_mixer import ConvMixerLayer, ConvMixer, ClassificationHead, PatchEmbeddings
else:
    exec(f'from llm_module.{args.solution} import ConvMixerLayer, ConvMixer')
    exec(f'from llm_module.{args.solution} import conv_ClassificationHead as ClassificationHead')
    exec(f'from llm_module.{args.solution} import conv_PatchEmbeddings as PatchEmbeddings')

@option(Configs.model)
def _conv_mixer(c: Configs):
    """
    ### Create model
    """

    # Create ConvMixer
    return ConvMixer(ConvMixerLayer(c.d_model, c.kernel_size), c.n_layers,
                     PatchEmbeddings(c.d_model, c.patch_size, 3),
                     ClassificationHead(c.d_model, c.n_classes)).to(c.device)


def main():
    # Create experiment
    experiment.create(name='ConvMixer', comment='cifar10')
    # Create configurations
    conf = Configs()
    # Load configurations
    experiment.configs(conf, {
        # Optimizer
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,

        # Training epochs and batch size
        'epochs': 10,
        'train_batch_size': 128,

        # Simple image augmentations
        'train_dataset': 'cifar10_train_augmented',
        # Do not augment images for validation
        'valid_dataset': 'cifar10_valid_no_augment',
    })
    # Set model for saving/loading
    experiment.add_pytorch_models({'model': conf.model})
    # Start the experiment and run the training loop
    with experiment.start():
        import time
        t1 = time.time()
        conf.run()
        t2 = time.time()
        print(t2 - t1)
    
    valid_acc = conf.validator.state_modules[0].data.correct / conf.validator.state_modules[0].data.samples
    print('final valid acc: ', valid_acc)
#
if __name__ == '__main__':
    main()
