import numpy as np


def create_dct_matrix(num_samples):
    """
    Create a matrix that can be left-multiplied to perform the discrete cosine
    transform.
    """
    result = np.zeros([num_samples, num_samples], dtype=np.float32)
    result += np.pi / num_samples
    result *= np.arange(num_samples, dtype=np.float32) + 0.5
    result *= np.arange(num_samples, dtype=np.float32)[:, None]
    result = np.cos(result)
    return result


def create_dct_inverse(num_samples):
    """
    Create the inverse of create_dct_matrix().
    """
    return np.linalg.inv(create_dct_matrix(num_samples))


def dct_period_to_bin(num_samples, period):
    """
    Get the DCT bin index closest to the given period.
    """
    return round(num_samples / (2 * period))
