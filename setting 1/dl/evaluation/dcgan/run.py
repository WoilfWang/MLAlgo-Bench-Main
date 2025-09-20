"""
---
title: Deep Convolutional Generative Adversarial Networks (DCGAN)
summary: A simple PyTorch implementation/tutorial of Deep Convolutional Generative Adversarial Networks (DCGAN).
---

# Deep Convolutional Generative Adversarial Networks (DCGAN)

This is a [PyTorch](https://pytorch.org) implementation of paper
[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434).

This implementation is based on the [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
"""

import torch.nn as nn

from labml import experiment
from labml.configs import calculate
from labml_helpers.module import Module
from config import Configs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--solution', default='golden')
args = parser.parse_args()

import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

if args.solution == 'golden':
    from labml_nn.gan.dcgan import Generator, Discriminator
else:
    exec(f'from llm_module.{args.solution} import Generator, Discriminator')


# We import the [simple gan experiment](../original/experiment.html) and change the
# generator and discriminator networks

calculate(Configs.generator, 'cnn', lambda c: Generator().to(c.device))
calculate(Configs.discriminator, 'cnn', lambda c: Discriminator().to(c.device))


def main():
    conf = Configs()
    conf.epochs = 10
    experiment.create(name='mnist_dcgan')
    experiment.configs(conf,
                       {'discriminator': 'cnn',
                        'generator': 'cnn',
                        'label_smoothing': 0.01})
    with experiment.start():
        import time
        t1 = time.time()
        conf.run()
        t2 = time.time()
        print('time overhead', t2 - t1)
        
    print('loss: ', conf.valid_loss[-1])


if __name__ == '__main__':
    main()
