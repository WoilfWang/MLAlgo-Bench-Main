"""
---
title: Rotary Positional Embeddings (RoPE) Experiment
summary: This experiment trains a transformer model with Rotary Positional Embeddings (RoPE) on tiny Shakespeare dataset.
---

# Rotary Positional Embeddings (RoPE) Experiment

This is an annotated PyTorch experiment to train a transformer model with Rotary Positional Embeddings (RoPE).
"""

from labml import experiment
from labml.configs import option, calculate
from labml_nn.transformers import TransformerConfigs
from labml_nn.transformers.basic.autoregressive_experiment import AutoregressiveTransformer, Configs


# ### Rotary PE attention
def _rotary_pe_mha(c: TransformerConfigs):
    from rope import RotaryPEMultiHeadAttention
    return RotaryPEMultiHeadAttention(c.n_heads, c.d_model, 1.)

# Configuration options
calculate(TransformerConfigs.encoder_attn, 'rotary', _rotary_pe_mha)
calculate(TransformerConfigs.decoder_attn, 'rotary', _rotary_pe_mha)
calculate(TransformerConfigs.decoder_mem_attn, 'rotary', _rotary_pe_mha)


@option(Configs.model, 'rotary_pe_transformer')
def _model(c: Configs):
    """
    Create an autoregressive model and initialize weights
    """
    m = AutoregressiveTransformer(c.transformer.encoder,
                                  c.transformer.src_embed,
                                  c.transformer.generator).to(c.device)

    return m


def main():
    # Create experiment
    experiment.create(name="rotary_pe_transformer", writers={'screen'})
    # Create configs
    conf = Configs()
    # Override configurations
    experiment.configs(conf, {
        # No fixed positional embeddings
        'transformer.src_embed': 'no_pos',
        'transformer.tgt_embed': 'no_pos',

        # Encoder with RoPE
        'transformer.encoder_attn': 'rotary',

        #
        'model': 'rotary_pe_transformer',

        # Use character level tokenizer
        'tokenizer': 'character',
        # Prompt separator is blank
        'prompt_separator': '',
        # Starting prompt for sampling
        'prompt': 'It is ',
        # Use Tiny Shakespeare dataset
        'text': 'tiny_shakespeare',

        # Use a context size of $256$
        'seq_len': 512,
        # Train for 32 epochs
        'epochs': 10,
        # Batch size $4$
        'batch_size': 4,
        # Switch between training and validation for $10$ times
        # per epoch
        'inner_iterations': 1,

        # Model size
        'd_model': 128,
        'transformer.ffn.d_ff': 512,
        'transformer.n_heads': 16,
        'transformer.dropout': 0.0,

        # Use [Noam optimizer](../../optimizers/noam.html)
        'optimizer.optimizer': 'Noam',
        'optimizer.learning_rate': 1.,

        'dataloader_shuffle_with_replacement': True
    })

    # Set models for saving and loading
    experiment.add_pytorch_models({'model': conf.model})

    # Start the experiment
    with experiment.start():
        # Run training
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
