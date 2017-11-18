#!/usr/bin/python3

# Evaluate the PSNR between noisy and clean images

import argparse
import logging
import math
import numpy as np

from nnimgproc.util.image import read


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=str, required=True,
                        help="Clean image file")
    parser.add_argument("--noisy", type=str, required=True,
                        help="Noisy image file")
    args = parser.parse_args()

    # Choose different noise to use
    clean = read(path=args.clean)
    noise = read(path=args.noisy)
    logging.info("PSNR: {} || MSE: {}".format(psnr(noise, clean),
                                              mse(noise, clean)))


if __name__ == "__main__":
    main()
