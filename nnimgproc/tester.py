import logging
import numpy as np
import time
from typing import Optional

from nnimgproc.model import BaseModel
from nnimgproc.processor import ModelProcessor
from nnimgproc.util.parameters import Parameters


class Tester(object):
    """
    Wrapper around Model and ModelProcessor for evaluating a neural network
    """
    def __init__(self, model: BaseModel,
                 model_processor: Optional[ModelProcessor]):
        """
        Initialize a neural network tester

        :param model: Model (inherited from nnimgproc.model.BaseModel)
        :param model_processor: ModelProcessor (from nnimgproc.processor)
        """
        self._logger = logging.getLogger(__name__)

        self._model = model
        self._model_processor = model_processor

        self._logger.info('Trainer (base) created.')

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

    def eval_all(self, images, metas):
        pass
