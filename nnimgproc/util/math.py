import math
import numpy as np
import scipy.ndimage.filters as fi


def mse(noisy: np.ndarray, clean: np.ndarray) -> float:
    """
    Calculate the MSE value between noise and clean image.

    :param noisy: ndarray, noisy image
    :param clean: ndarray, clean image
    :return: float, mean-square-error value
    """
    assert noisy.shape == clean.shape, "Shape mismatch when computing MSE."
    res = np.square(noisy.astype(float) - clean.astype(float)).mean()
    return res


def psnr(noisy: np.ndarray, clean: np.ndarray, dynamic: float=1.0) -> float:
    """
    Calculate the PSNR value between noise and clean image.

    :param noisy: ndarray, noisy image
    :param clean: ndarray, clean image
    :param dynamic: float, the scale (max value) of the image
    :return: float, PSNR value
    """
    assert noisy.shape == clean.shape, "Shape mismatch when computing PSNR."
    peak = dynamic * dynamic
    return 10 * math.log10(peak / mse(noisy, clean))


def gaussian_kernel(size: int, sigma: float=1, scale: float=1) -> np.ndarray:
    """
    Create a 2-dimensional Gaussian kernel

    :param size: integer, size of the kernel / width and height
    :param sigma: float, standard variance of the kernel
    :param scale: float, sum of the output
    :return: ndarray, Gaussian kernel array
    """
    inp = np.zeros((size, size))
    # set element at the middle to one, a dirac delta
    inp[size // 2, size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, sigma) * scale
