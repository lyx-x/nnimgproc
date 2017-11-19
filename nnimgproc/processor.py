import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from nnimgproc.model import BaseModel
from nnimgproc.util.parameters import Parameters


class TargetProcessor(object):
    """
    This class encapsulates a lambda function images -> (input, output, meta),
    image processing pipeline to imitate. The meta contains some parameters
    used in the processing pipeline which can then be used to help learning
    or evaluating the result.
    """
    def __init__(self):
        pass

    def __call__(self, img: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, Optional[Parameters]]:
        """

        :param img: ndarray of shape (w, h, 1) for grey image or (w, h, 3)
        :return: tuple of (input, output, meta) / (ndarray, ndarray, Parameters)
        """
        raise NotImplementedError


class BatchProcessor(object):
    """
    This class encapsulates a lambda function (xs, ys, metas) -> (batch_x,
    batch_y) that converts a list of images to training batches. The batch_x
    and batch_y can be dictionary if needed.
    """
    def __init__(self):
        pass

    def __call__(self, xs: List[np.ndarray], ys: List[np.ndarray],
                 metas: List[Parameters]) \
            -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]],
                     Union[np.ndarray, Dict[str, np.ndarray]]]:
        """

        :param xs: ndarrays, inputs output by TargetProcessor
        :param ys: ndarrays, outputs output by TargetProcessor
        :param metas: a list of Parameters (from nnimgproc.util.parameters),
                      meta output by TargetProcessor
        :return: 2 ndarrays or 2 dictionary that can be read by a neural network
        """
        raise NotImplementedError


class ModelProcessor(object):
    """
    This class encapsulates a lambda function (img, meta, model) -> output,
    image processing pipeline to imitate. The meta contains some parameters
    used in the processing pipeline which can then be used to help generating
    the result. The __call__ function takes model as input because it
    contains a preprocessor and postprocessor that act before and after the
    model is called. The preprocessor is usually similar to what is defined
    in the BatchProcessor.
    """
    def __init__(self):
        pass

    def __call__(self, img: np.ndarray, meta: Parameters, model: BaseModel) \
            -> np.ndarray:
        """

        :param img: ndarray of shape (w, h, 1) for grey image or (w, h, 3) for
                    RBG images
        :param meta: Parameters (from nnimgproc.util.parameters), meta output
                     by TargetProcessor or produced by external procedures
        :return: ndarray representing the processed image
        """
        raise NotImplementedError
