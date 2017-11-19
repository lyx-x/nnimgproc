import logging

from nnimgproc.dataset import Dataset
from nnimgproc.model import BaseModel
from nnimgproc.processor import TargetProcessor, BatchProcessor
from nnimgproc.util.parameters import Parameters


class BaseTrainer(object):
    """
    Meta class for training and evaluating a neural network
    """
    def __init__(self, model: BaseModel, params: Parameters, dataset: Dataset,
                 target_processor: TargetProcessor,
                 batch_processor: BatchProcessor):
        """
        Initialize a neural network trainer/optimizer

        :param model: Model (inherited from nnimgproc.model.BaseModel)
        :param params: Parameters (from nnimgproc.util.parameters),
                       training parameter set such as learning rate
        :param dataset: Dataset (from nnimgproc.dataset), image provider
        :param target_processor: TargetProcessor (from nnimgproc.processor)
        :param batch_processor: BatchProcessor (from nnimgproc.processor)
        """
        self._logger = logging.getLogger(__name__)

        self._model = model
        self._params = params
        self._dataset = dataset
        self._target_processor = target_processor
        self._batch_processor = batch_processor

        self._output_dir = self._params.get('output_dir')

        self._logger.info('Trainer (base) created.')

    def train(self):
        """
        Training loop, from data generation to the actual training

        :return:
        """
        raise NotImplementedError

    def save(self, path: str):
        """
        Save the model to the file system

        :param path: string, root folder for the model file
        :return:
        """
        self._model.save(path=path)
