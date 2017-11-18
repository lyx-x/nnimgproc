import logging
from typing import Optional

import numpy as np
import time

from nnimgproc.dataset import Dataset
from nnimgproc.model import BaseModel
from nnimgproc.processor import TargetProcessor, BatchProcessor, ModelProcessor
from nnimgproc.util.parameters import Parameters


class BaseTrainer(object):
    """
    Meta class for training and evaluating a neural network
    """
    def __init__(self, model: BaseModel, params: Optional[Parameters],
                 dataset: Dataset, target_processor: Optional[TargetProcessor],
                 batch_processor: Optional[BatchProcessor],
                 model_processor: Optional[ModelProcessor]):
        """
        Initialize a neural network trainer/optimizer

        :param model: Model (inherited from nnimgproc.model.BaseModel)
        :param params: Parameters (from nnimgproc.util.parameters),
                       training parameter set such as learning rate
        :param dataset: Dataset (from nnimgproc.dataset), image provider
        :param target_processor: TargetProcessor (from nnimgproc.processor)
        :param batch_processor: BatchProcessor (from nnimgproc.processor)
        :param model_processor: ModelProcessor (from nnimgproc.processor)
        """
        self._logger = logging.getLogger(__name__)

        # For training, params, target_processor and batch_processor are needed
        # For evaluation, only model_processor is needed

        self._model = model
        self._params = params
        self._dataset = dataset
        self._target_processor = target_processor
        self._batch_processor = batch_processor
        self._model_processor = model_processor

        self._output_dir = self._params.get('output_dir')

        self._logger.info('Trainer (base) created.')

    def train(self):
        """
        Training loop, from data generation to the actual training

        :return:
        """
        raise NotImplementedError

    def eval(self, image: np.ndarray, meta: Parameters):
        """
        Process one image. By default, this function doesn't know whether
        there is a ground truth. All extra information is provided by the
        meta variable which in theory corresponds to the meta output by the
        TargetProcessor when generating the input/output pair. In practice,
        if the dataset is not simulated, one can use external procedures to
        approximate this meta variable.

        For example, a noise level estimator can output a noise map into
        the meta variable which can be used by the denoiser if the denoiser
        is trained with such extra information.

        :param: image: ndarray of shape (w, h, 1) or (w, h, 3),
                       image to be processed
        :param: meta: Parameters
        :return: ndarray, processed image
        """
        # Timed operation
        start = time.time()

        assert image.ndim == 3, "image should be a ndarray of 3 dimensions."
        self._logger.info("Start individual processing.")

        output = self._model_processor(image, meta, self._model)
        end = time.time()
        elapsed = end - start

        self._logger.info("Processing completed in %.3f seconds." % elapsed)
        # The output can be anything, usually it is an image
        return output

    def save(self, path: str):
        """
        Save the model to the file system

        :param path: string, root folder for the model file
        :return:
        """
        self._model.save(path=path)
