"""
---
title: WGAN experiment with MNIST
summary: This experiment generates MNIST images using convolutional neural network.
---

# WGAN experiment with MNIST
"""
from labml import experiment

from labml.configs import calculate
# Import configurations from [DCGAN experiment](../dcgan/index.html)
from labml_nn.gan.dcgan import Configs

# Import [Wasserstein GAN losses](./index.html)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--solution', default='golden')
args = parser.parse_args()

import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

if args.solution == 'golden':
    from labml_nn.gan.wasserstein import GeneratorLoss, DiscriminatorLoss
else:
    exec(f'from llm_module.{args.solution} import GeneratorLoss, DiscriminatorLoss')

# Set configurations options for Wasserstein GAN losses
calculate(Configs.generator_loss, 'wasserstein', lambda c: GeneratorLoss())
calculate(Configs.discriminator_loss, 'wasserstein', lambda c: DiscriminatorLoss())


def main():
    # Create configs object
    conf = Configs()
    # Create experiment
    experiment.create(name='mnist_wassertein_dcgan', comment='test')
    # Override configurations
    experiment.configs(conf,
                       {
                           'discriminator': 'cnn',
                           'generator': 'cnn',
                           'label_smoothing': 0.01,
                           'generator_loss': 'wasserstein',
                           'discriminator_loss': 'wasserstein',
                       })

    # Start the experiment and run training loop
    with experiment.start():
        import time
        t1 = time.time()
        conf.run()
        t2 = time.time()
        print('time overhead: ', t2 - t1)
    


if __name__ == '__main__':
    main()