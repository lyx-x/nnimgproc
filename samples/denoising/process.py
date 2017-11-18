#!/usr/bin/python3

# Noise demonstration: test noise generator

import argparse
import os

from nnimgproc.target_processor.denoising import DenoisingTargetProcessor
from nnimgproc.util.image import read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the image.')
    parser.add_argument('--output_folder', type=str, default='results',
                        help='Path to the output file')
    parser.add_argument('--output_image', type=str, default='noisy.jpg',
                        help='File name of x output by TargetProcessor')
    parser.add_argument('--output_meta', type=str, default='meta.pkl',
                        help='Filename for meta')
    parser.add_argument('--noise', nargs='*', required=False,
                        default=['gaussian', 0.1], help='Noise type')
    args = parser.parse_args()

    # Define processors
    noise_type = args.noise[0]
    noise_params = [] if len(args.noise) == 1 else args.noise[1:]
    target_processor = DenoisingTargetProcessor(noise_type, noise_params)

    image = read(path=args.input, shape=(128, 128), as_grey=True)
    x, y, meta = target_processor(image)

    write(image=x, path=os.path.join(args.output_folder, args.output_image))
    meta.save(path=os.path.join(args.output_folder, args.output_meta))


if __name__ == '__main__':
    main()
