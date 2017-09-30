import logging

from nnimgproc.util.parameters import Parameters


class BaseTrainer(object):
    """
    Meta class for training and evaluating a neural network
    """
    def __init__(self, model, training_parameters, dataset, target_processing, batch_processing, post_processing):
        """
        Initialize a neural network trainer/optimizer

        :param model: Model (from nnimgproc.backend.keras)
        :param training_parameters: Parameters (from nnimgproc.util.parameters), training parameter set
        :param dataset: Dataset (from nnimgproc.dataset), image minibatch provider
        :param target_processing: lambda function img -> (input, output, meta), image processing pipeline to imitate.
                                 The meta contains some parameters used in the processing pipeline which
                                 can then be used to help learning or evaluating the result
        :param batch_processing: lambda function (x, y, meta, shape, batch_size, is_random) -> (x, y, meta),
                                 convert images to patches. If is_random is False, the processor will act in a
                                 deterministic way which (together with meta) will be used to rebuild the image
        :param post_processing: (batch_x, meta) -> img, rebuild one image from batches which may be patches of it
        """
        self._logger = logging.getLogger(__name__)

        self._model = model
        self._training_parameters = training_parameters
        self._dataset = dataset
        self._target_processing = target_processing
        self._batch_processing = batch_processing
        self._post_processing = post_processing

        assert isinstance(self._training_parameters, Parameters), 'training_parameters should use the Parameters module'
        self._logger.info('Trainer created using %s as backend]' % self._model.backend)

    def train(self):
        """
        Training loop

        :return:
        """
        pass

    def eval(self, images):
        """
        Process all images in the dataset

        :param: images: ndarray of shape (size, w, h, 1) or (size, w, h, 3), batch images to be processed
        :return: ndarray, processed images
        """
        pass
