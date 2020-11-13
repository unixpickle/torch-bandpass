import torch
import torch.nn as nn

from .dct import create_dct_matrix, create_dct_inverse, dct_period_to_bin


class Prism(nn.Module):
    """
    A Prism module performs different bandpass filters on different subsets of
    the features in a batch of activations.

    :param num_samples: the number of timesteps to expect in the inputs.
    :param mid_periods: the periods separating the bands. Periods of 1 and
                        infinity are implied at the two extremes.
    """

    def __init__(self, num_samples, mid_periods=(2, 8, 32, 256)):
        super().__init__()
        self.num_samples = num_samples
        self.mid_periods = mid_periods
        bins = (
            [0]
            + [dct_period_to_bin(num_samples, p) for p in mid_periods[::-1]]
            + [num_samples]
        )
        self.bands = nn.ModuleList([])
        for min_index, max_index in zip(bins, bins[1:]):
            self.bands.append(Bandpass(num_samples, min_index, max_index))

    def forward(self, x):
        """
        Apply the Prism layer to a batch of sequences.

        :param x: an [N x T x C] Tensor, where C is the number of features,
                  T is the number of timesteps, and N is the batch size.
        """
        n, t, _c = x.shape
        assert t == self.num_samples

        x = x.permute(2, 0, 1).contiguous()  # put C on the outer dimension

        chunks_in = _split_up_chunks(x, len(self.bands))
        chunks_out = []
        for bandpass, chunk in zip(self.bands, chunks_in):
            chunk = chunk.reshape(-1, t)
            chunk = bandpass(chunk)
            chunk = chunk.reshape(-1, n, t)
            chunks_out.append(chunk)
        joined_out = torch.cat(chunks_out, dim=0)
        joined_out = joined_out.permute(
            1, 2, 0
        ).contiguous()  # put C back as the inner dimension
        return joined_out


class Bandpass(nn.Module):
    def __init__(self, num_samples, min_index, max_index):
        super().__init__()
        dct = create_dct_matrix(num_samples)[min_index:max_index]
        inv = create_dct_inverse(num_samples)[:, min_index:max_index]
        self.register_buffer("forward_dct", torch.from_numpy(dct.T))
        self.register_buffer("backward_dct", torch.from_numpy(inv.T))

    def forward(self, x):
        return (x @ self.forward_dct) @ self.backward_dct


def _split_up_chunks(batch, num_chunks):
    chunk_size = batch.shape[0] // num_chunks
    num_larger_chunks = batch.shape[0] % num_chunks
    start_idx = 0
    result = []
    for i in range(num_chunks):
        this_chunk_size = chunk_size
        if i < num_larger_chunks:
            this_chunk_size += 1
        result.append(batch[start_idx : start_idx + this_chunk_size])
        start_idx += this_chunk_size
    return result
