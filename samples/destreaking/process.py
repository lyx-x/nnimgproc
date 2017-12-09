#!/usr/bin/python3

# Radon reconstruction demonstration: test streak generator

import argparse
import logging
import os

from nnimgproc.target_processor.desteaking import DestreakingTargetProcessor
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
    parser.add_argument('--streak', nargs='*', required=False,
                        default=['periodic', 10], help='Radon parameters')
    args = parser.parse_args()

    # Define processors
    streak_type = args.streak[0]
    streak_params = [] if len(args.streak) == 1 else args.streak[1:]
    target_processor = DestreakingTargetProcessor(streak_type, streak_params)

    image = read(path=args.input, shape=(128, 128), as_grey=True)
    x, y, meta = target_processor(image)

    write(image=x, path=os.path.join(args.output_folder, args.output_image))
    meta.save(path=os.path.join(args.output_folder, args.output_meta))
    logging.info("Finish processing %s" % args.input)


if __name__ == '__main__':
    main()
