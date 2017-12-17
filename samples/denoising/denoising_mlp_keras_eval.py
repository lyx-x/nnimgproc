#!/usr/bin/python3

# A sample script for evaluating pretrained denoiser in keras

import argparse
import concurrent.futures
import numpy as np
from typing import Tuple

from nnimgproc import load_model, build_tester
from nnimgproc.backend.keras import Model
from nnimgproc.dataset import ImageSingleton
from nnimgproc.processor import ModelProcessor
from nnimgproc.util.image import write
from nnimgproc.util.math import gaussian_kernel
from nnimgproc.util.parameters import Parameters


# Define a image to patch to image pipeline in a ModelProcessor
class DenoisingModelProcessor(ModelProcessor):
    """
    Divide the image into patches, denoise each patch and assemble them
    with a gaussian window of SD sigma.
    """
    def __init__(self, patch: int, stride: int, sigma: float,
                 batch_size: int=32, workers: int=1):
        super(DenoisingModelProcessor, self).__init__()
        self._patch = patch
        self._stride = stride
        self._sigma = sigma
        self._batch_size = batch_size
        self._workers = workers

    def __call__(self, img: np.ndarray, meta: Parameters, model: Model) \
            -> np.ndarray:
        shape = img.shape
        channels = shape[2]
        self._weight_window = np.zeros((self._patch, self._patch, channels))
        for i in range(channels):
            self._weight_window[:, :, i] = gaussian_kernel(self._patch,
                                                           self._sigma)

        # Count the number of patches on the image.
        # If the patches as cannot cover the whole image, the model will go
        # through all corner cases, e.g. bottom and right border, hence extra +1
        hor_count = (shape[0] - self._patch) // self._stride + 1 + 1
        ver_count = (shape[1] - self._patch) // self._stride + 1 + 1
        patch_count = hor_count * ver_count

        def get_coordinates(idx: int) -> Tuple[int, int]:
            hor = min(idx % hor_count * self._stride, shape[0] - self._patch)
            ver = min(idx // hor_count * self._stride, shape[1] - self._patch)
            return hor, ver

        # Allocate a size that is a multiple of batch size
        alloc_count = (patch_count + self._batch_size - 1) \
            // self._batch_size * self._batch_size
        batches = np.zeros((alloc_count, self._patch, self._patch, channels))
        positions = [get_coordinates(i) for i in range(patch_count)]

        y = np.zeros_like(img)
        weight = np.zeros_like(img)

        def divide(idx):
            hor, ver = positions[idx]
            batches[idx] = img[hor: hor + self._patch,
                               ver: ver + self._patch, :]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._workers) \
                as executor:
            executor.map(divide, range(patch_count))

        for i in range(alloc_count // self._batch_size):
            x_batch = batches[self._batch_size * i: self._batch_size * (i + 1)]
            y_batch = model.model.predict({'x': x_batch})

            for j in range(self._batch_size):
                idx = i * self._batch_size + j
                if idx >= patch_count:
                    break
                hor, ver = positions[idx]
                y[hor: hor + self._patch, ver: ver + self._patch, :] += \
                    y_batch[j] * self._weight_window
                weight[hor: hor + self._patch, ver: ver + self._patch, :] += \
                    self._weight_window

        assert weight.min() > 0, 'Divided by 0 is not accepted, use smoother ' \
                                 'Gaussian window, e.g. increase sigma'
        return (y / weight).clip(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='results', required=True,
                        help='Folder containing the pretrained model.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the noisy image.')
    parser.add_argument('--meta', type=str, help='Path to the meta file.')
    parser.add_argument('--output', type=str, default='results/result.png',
                        help='Path to the output file.')
    parser.add_argument('--patch', type=int, default=17,
                        help='Patch size for the input.')
    parser.add_argument('--minibatch', type=int, default=32,
                        help='Size of minibatch.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel batch generation.')
    args = parser.parse_args()

    # Load the image
    dataset = ImageSingleton(path=args.input, shape=(128, 128), as_grey=True)

    # Load the pretrained model
    model = load_model(model=None, path=args.model, backend='keras')

    # Define ModelProcessor
    patch = args.patch
    stride = 3
    sigma = 5
    model_processor = DenoisingModelProcessor(patch, stride, sigma,
                                              batch_size=args.minibatch,
                                              workers=args.workers)

    # Create a trainer
    tester = build_tester(model, model_processor)

    meta = Parameters()
    meta.load(args.meta)
    output = tester.eval(dataset.get(), meta)

    write(output, args.output)


if __name__ == '__main__':
    main()
