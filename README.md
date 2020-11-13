# torch-bandpass

This is an implementation of the [Prism layer](https://arxiv.org/abs/2011.04823), a DCT-based bandpass filter suitable for transformer sequence models.

# Usage

See [example.ipynb](example.ipynb) for full usage. The basic usage is as follows:

```python
seq_len = 512  # number of timesteps per sequence
d_model = 768  # number of feature channels in the transformer

# Create a Prism layer, which only needs to know about the
# total sequence length and how you want to split up features.
layer = Prism(seq_len, mid_periods=(2, 8, 32, 256))

# random [N x T x C] tensor.
input_sequence = torch.randn(BATCH_SIZE, seq_len, d_model)

# output is the same shape as input_sequence
output_sequence = layer(input_sequence)
```
