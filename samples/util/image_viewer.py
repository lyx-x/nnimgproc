#!/usr/bin/python3

# A sample script displaying an image

import argparse
import logging

from nnimgproc.dataset import ImageSingleton
from nnimgproc.util.image import show


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to the image.')
    args = parser.parse_args()

    # Load the image
    dataset = ImageSingleton(path=args.input, shape=(128, 128), as_grey=False)
    image = dataset.get()
    logging.info("Load image %s" % args.input)
    logging.info("Shape %s" % str(image.shape))

    # Display the image
    show(image)


if __name__ == '__main__':
    main()
