#!/usr/bin/python3

import argparse
import logging

from keras.layers import Input, Dense
from keras.models import Model

from nnimgproc import build_model, build_trainer, load_model
from nnimgproc.dataset import ImageFolder
from nnimgproc.util.parameters import default_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results", help="Prefix for output files")
    parser.add_argument("--image_dir", type=str, default="/home/lyx/Storage/Data/ILSVRC2010_images_val/val_256", help="Clean image folder")
    parser.add_argument('--noise', nargs='*', required=False, default=['cte_gaussian', 25], help="Noise type")
    parser.add_argument("--patch", type=int, default=17, help="Patch size for the input")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--minibatch", type=int, default=32, help="Batch size")
    parser.add_argument("--epoch", type=int, default=20, help="Number of epoch")
    parser.add_argument("--training", type=int, default=100000, help="Number of training samples")
    parser.add_argument("--validation", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--worker", type=int, default=1, help="Number of threads for image generation")
    parser.add_argument('--resume', dest='resume', action='store_true')
    args = parser.parse_args()

    # Prepare the raw image dataset
    folder = args.output_dir
    dataset = ImageFolder(folder, max_size=1000, shape=(128, 128), as_grey=False)

    # Create or load a pre-trained model
    if args.resume:
        model = load_model(path=args.output_dir, backend='keras')
    else:
        # Declare the model
        input = Input(shape=(256,))
        x = Dense(64, activation='relu')(input)
        x = Dense(64, activation='relu')(x)
        output = Dense(256)(x)

        # This creates a model that includes
        # the Input layer and three Dense layers
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

        model = build_model(model=model, backend='keras')

    # Create a trainer
    trainer = build_trainer(model=model,
                            training_parameters=default_parameters,
                            dataset=dataset,
                            target_processing=None,
                            batch_processing=None,
                            post_processing=None
                            )
    trainer.train()
    trainer.save(path=args.output_dir)


if __name__ == '__main__':
    main()
