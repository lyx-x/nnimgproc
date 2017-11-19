#!/usr/bin/python3

# Evaluate the PSNR between noisy and clean images

import argparse
import logging

from nnimgproc.util.image import read
from nnimgproc.util.math import mse, psnr


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
