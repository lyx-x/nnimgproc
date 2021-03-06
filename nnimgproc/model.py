import logging
from typing import Any


class BaseModel(object):
    """
    Base class to define a image processing model based on neural network
    """
    def __init__(self, model: Any, backend: str):
        """
        Create a model wrapper for different backend

        :param model: definition of the neural network
        :param backend: name of the backend among "keras" and "chainer" (to be extended)
        """
        self._logger = logging.getLogger(__name__)

        self._model = model
        self._backend = backend
        self._logger.info('Model (base) created.')

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def model(self):
        return self._model

    def save(self, path: str):
        """
        Save the model to the file system

        :param path: path to the model file
        :return:
        """
        pass
