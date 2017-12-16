#!/usr/bin/python3

# A sample script for training de-streaker with convolutional neural network
# implemented with Chainer

import argparse
import concurrent.futures
import logging
import numpy as np
from typing import Dict, List, Tuple, Union

from chainer import Chain
from chainer.functions import relu, mean_squared_error
from chainer.links import Convolution2D

from nnimgproc import build_model, build_trainer, load_model
from nnimgproc.dataset import ImageFolder
from nnimgproc.processor import BatchProcessor
from nnimgproc.target_processor.desteaking import DestreakingTargetProcessor
from nnimgproc.util.parameters import Parameters


# Define a training batch generator in a BatchProcessor
class DestreakingBatchProcessor(BatchProcessor):
    def __init__(self, x_shape: Tuple[int, int], y_shape: Tuple[int, int],
                 batch_size: int, workers: int):
        super(DestreakingBatchProcessor, self).__init__()
        self._x_shape = x_shape
        self._y_shape = y_shape
        self._batch_size = batch_size
        self._workers = workers

    def __call__(self, xs: List[np.ndarray], ys: List[np.ndarray],
                 metas: List[Parameters]) \
            -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]],
                     Union[np.ndarray, Dict[str, np.ndarray]]]:
        # Assume all elements in x and y have the same shape
        # Image shape
        shape = xs[0].shape

        # Generate a list of indexes: select images for each batch position
        indices = np.random.randint(0, len(xs), self._batch_size)

        # Generate for each index a patch position / a tuple of integers
        # Here the range(2) represents (horizontal, vertical) coordinates
        margin = tuple(map(lambda i: min(self._x_shape[i],
                                         self._y_shape[i]), range(2)))

        x_offset = tuple(map(lambda i: (self._x_shape[i] - margin[i]) // 2,
                             range(2)))
        y_offset = tuple(map(lambda i: (self._y_shape[i] - margin[i]) // 2,
                             range(2)))

        corner_positions = tuple(map(lambda i:
                                     np.random.randint(0, shape[i] - margin[i],
                                                       self._batch_size),
                                     range(2)))

        # Output the training pair
        x_batch = np.zeros((self._batch_size,) + self._x_shape, np.float32)
        y_batch = np.zeros((self._batch_size,) + self._y_shape, np.float32)

        def fill(i):
            x_batch[i] = xs[indices[i]][
                         corner_positions[0][i] + x_offset[0]:
                         corner_positions[0][i] + x_offset[0]
                         + self._x_shape[0],
                         corner_positions[1][i] + x_offset[1]:
                         corner_positions[1][i] + x_offset[1]
                         + self._x_shape[0],
                         :
                         ]

            y_batch[i] = ys[indices[i]][
                         corner_positions[0][i] + y_offset[0]:
                         corner_positions[0][i] + y_offset[0]
                         + self._y_shape[0],
                         corner_positions[1][i] + y_offset[1]:
                         corner_positions[1][i] + y_offset[1]
                         + self._y_shape[0],
                         :
                         ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._workers) \
                as executor:
            executor.map(fill, range(self._batch_size))

        # chainer requires a input shape like (N, W, H, C)
        x_batch = np.moveaxis(x_batch, 3, 1)
        y_batch = np.moveaxis(y_batch, 3, 1)

        return {'x': x_batch}, {'y': y_batch}


class CNN(Chain):

    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self._conv1 = Convolution2D(1, 16, 5, stride=1, pad=2)
            self._conv2 = Convolution2D(16, 16, 3, stride=1, pad=1)
            self._conv3 = Convolution2D(16, 1, 5, stride=1, pad=2)

    # We incorporate the loss function inside the model definition.
    # The input should match what is returned by the BatchProcessor.
    def __call__(self, x, y):
        h = relu(self._conv1(x['x']))
        h = relu(self._conv2(h))
        h = self._conv3(h)
        h = mean_squared_error(h, y['y'])
        return h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Prefix for output files')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Clean image folder')
    parser.add_argument('--max_image_count', type=int, default=10000,
                        help='Maximum number of images in the RAM')
    parser.add_argument('--device', type=int, default=-1,
                        help='Number of GPU device, -1 for CPU')
    parser.add_argument('--streak', nargs='*', required=False,
                        default=['periodic', 10], help='Radon parameters')
    parser.add_argument('--input_patch', type=int, default=65,
                        help='Patch size for the input')
    parser.add_argument('--output_patch', type=int, default=65,
                        help='Patch size for the output')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--minibatch', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--training', type=int, default=64000,
                        help='Number of training samples per epoch')
    parser.add_argument('--validation', type=int, default=1280,
                        help='Number of validation samples per epoch')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of threads for image generation')
    parser.add_argument('--resume', dest='resume', action='store_true')
    args = parser.parse_args()

    # Set parameters
    params = Parameters()
    params.set('output_dir', args.output_dir)
    params.set('input_shape', (args.input_patch, args.input_patch, 1))
    params.set('output_shape', (args.output_patch, args.output_patch, 1))
    params.set('device', args.device)
    params.set('learning_rate', args.learning_rate)
    params.set('image_minibatch', args.minibatch // 8)
    params.set('training_minibatch', args.minibatch)
    params.set('epochs', args.epochs)
    params.set('training_batches', args.training // args.minibatch)
    params.set('validation_batches', args.validation // args.minibatch)
    params.set('workers', args.workers)

    logging.info("Parameters loaded.")

    # Prepare the raw image dataset
    folder = args.image_dir
    dataset = ImageFolder(folder, shape=(128, 128), as_grey=True,
                          max_size=args.max_image_count)

    logging.info("Image folder loaded.")

    chain = CNN()

    # Create or load a pre-trained model
    if args.resume:
        model = load_model(model=chain, path=params.get('output_dir'),
                           backend='chainer')
    else:
        model = build_model(model=chain, backend='chainer')

    logging.info("Model constructed.")

    # Define processors
    streak_type = args.streak[0]
    streak_params = [] if len(args.streak) == 1 else args.streak[1:]
    target_processor = DestreakingTargetProcessor(streak_type, streak_params)

    x_shape = params.get('input_shape')
    y_shape = params.get('output_shape')
    batch_size = params.get('training_minibatch')
    batch_processor = DestreakingBatchProcessor(x_shape, y_shape, batch_size,
                                                params.get('workers'))

    logging.info("Processors defined.")

    # Create a trainer
    trainer = build_trainer(model, params, dataset, target_processor,
                            batch_processor)
    trainer.train()
    trainer.save(path=args.output_dir)

    logging.info("Exit training script.")


if __name__ == '__main__':
    main()
