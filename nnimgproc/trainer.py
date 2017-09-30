import logging

from nnimgproc.util.parameters import Parameters


class BaseTrainer(object):
    """
    Meta class for training and evaluating a neural network
    """
    def __init__(self, model, learning_parameters, dataset, target_processing, reshaper):
        """
        Initialize a neural network trainer/optimizer

        :param model: neural network model
        :param learning_parameters: training parameters in form of a dictionary
        :param dataset: image minibatch provider
        :param target_processing: image processing pipeline to imitate
        :param reshaper: a tuple of pre/post-processing methods that are used to create minibatches
        """
        self._logger = logging.getLogger(__name__)

        self._model = model
        self._learning_parameters = learning_parameters
        self._dataset = dataset
        self._target_processing = target_processing
        self._pre_processing = reshaper[0]
        self._post_processing = reshaper[1]

        assert isinstance(self._learning_parameters, Parameters), 'learning_parameters should use the Parameters module'
        self._logger.info('Trainer created using %s as backend]' % self._model.backend)

    def train(self):
        pass

    def test(self):
        pass
