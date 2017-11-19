#!/usr/bin/python3

# A sample script for training denoiser with multilayer perceptron using keras

import argparse
import concurrent.futures
import logging
import numpy as np
from typing import Dict, List, Tuple, Union

from keras.layers import Input, Dense, Reshape, Flatten
from keras.models import Model
from keras.optimizers import Adam

from nnimgproc import build_model, build_trainer, load_model
from nnimgproc.dataset import ImageFolder
from nnimgproc.processor import BatchProcessor
from nnimgproc.target_processor.denoising import DenoisingTargetProcessor
from nnimgproc.util.parameters import Parameters


# Define a training batch generator in a BatchProcessor
class DenoisingBatchProcessor(BatchProcessor):
    def __init__(self, x_shape: Tuple[int, int], y_shape: Tuple[int, int],
                 batch_size: int, workers: int):
        super(DenoisingBatchProcessor, self).__init__()
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
        x_batch = np.zeros((self._batch_size,) + self._x_shape)
        y_batch = np.zeros((self._batch_size,) + self._y_shape)

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

        return {'x': x_batch}, {'y': y_batch}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Prefix for output files')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Clean image folder')
    parser.add_argument('--max_image_count', type=int, default=10000,
                        help='Maximum number of images in the RAM')
    parser.add_argument('--noise', nargs='*', default=['gaussian', 0.1],
                        help='Noise type')
    parser.add_argument('--input_patch', type=int, default=17,
                        help='Patch size for the input')
    parser.add_argument('--output_patch', type=int, default=17,
                        help='Patch size for the output')
    parser.add_argument('--layers', nargs='*', default=[2047, 2047, 2047, 2047],
                        help='Size of hidden layers')
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

    # Create or load a pre-trained model
    if args.resume:
        model = load_model(path=params.get('output_dir'), backend='keras')
    else:
        # Declare the model
        input = Input(shape=params.get('input_shape'), name='x')
        x = Flatten()(input)
        for h in args.layers:
            x = Dense(int(h), activation='relu')(x)
        x = Dense(args.output_patch * args.output_patch)(x)
        output = Reshape(target_shape=params.get('output_shape'), name='y')(x)

        # This creates a model that includes
        # the Input layer and three Dense layers
        model = Model(inputs=[input], outputs=[output])
        model.compile(optimizer=Adam(lr=params.get('learning_rate')),
                      loss='mse')

        model = build_model(model=model, backend='keras')

    logging.info("Model constructed.")

    # Define processors
    noise_type = args.noise[0]
    noise_params = [] if len(args.noise) == 1 else args.noise[1:]
    target_processor = DenoisingTargetProcessor(noise_type, noise_params)

    x_shape = params.get('input_shape')
    y_shape = params.get('output_shape')
    batch_size = params.get('training_minibatch')
    batch_processor = DenoisingBatchProcessor(x_shape, y_shape, batch_size,
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
