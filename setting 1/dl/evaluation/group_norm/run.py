"""
---
title: CIFAR10 Experiment to try Group Normalization
summary: >
  This trains is a simple convolutional neural network that uses group normalization
  to classify CIFAR10 images.
---

# CIFAR10 Experiment for Group Normalization
"""

import torch.nn as nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.cifar10 import CIFAR10Configs, CIFAR10VGGModel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--solution', default='golden')
args = parser.parse_args()

import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

if args.solution == 'golden':
    from labml_nn.normalization.group_norm import GroupNorm
else:
    exec(f'from llm_module.{args.solution} import GroupNorm')


class Model(CIFAR10VGGModel):
    """
    ### VGG model for CIFAR-10 classification

    This derives from the [generic VGG style architecture](../../experiments/cifar10.html).
    """

    def conv_block(self, in_channels, out_channels) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            GroupNorm(self.groups, out_channels),#new
            nn.ReLU(inplace=True),
        )

    def __init__(self, groups: int = 32):
        self.groups = groups#input param:groups to conv_block
        super().__init__([[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]])


class Configs(CIFAR10Configs):
    # Number of groups
    groups: int = 16


@option(Configs.model)
def model(c: Configs):
    """
    ### Create model
    """
    return Model(c.groups).to(c.device)


def main():
    # Create experiment
    experiment.create(name='cifar10', comment='group norm')
    # Create configurations
    conf = Configs()
    conf.epochs = 5
    # Load configurations
    experiment.configs(conf, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,
    })
    # Start the experiment and run the training loop
    with experiment.start():
        import time
        t1 = time.time()
        # Run training
        conf.run()
        t2 = time.time()
        print('time overhead: ', t2 - t1)
    
    valid_acc = conf.validator.state_modules[0].data.correct / conf.validator.state_modules[0].data.samples
    print('final valid acc: ', valid_acc)


#
if __name__ == '__main__':
    main()