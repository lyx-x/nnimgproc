import logging


class BaseModel(object):
    """
    Base class to define a image processing model based on neural network
    """
    def __init__(self, model, backend):
        """
        Create a model wrapper for different backend

        :param model: definition of the neural network
        :param backend: name of the backend among "keras" and "chainer" (to be extended)
        """
        self._logger = logging.getLogger(__name__)

        self._model = model
        self._backend = backend
        self._logger.info('Model created using %s as backend]' % self._backend)

    @property
    def backend(self):
        return self._backend

    @property
    def model(self):
        return self._model

    def save(self, path):
        """
        Save the model to the file system

        :param path: path to the model file
        :return:
        """
        pass

    def load(self, path):
        """
        Load a model from the file system

        :param path: path to the model file
        :return:
        """
        pass
