"""
Keras backend 

The backend exposes:
    - Class: Trainer
    - Class: Model
    - Function: load
"""
import os
import time

from nnimgproc.model import BaseModel
from nnimgproc.trainer import BaseTrainer

# Name of the backend depends on the file name
BACKEND = os.path.basename(__file__)
# Name of the saved model file
MODEL_FILENAME = 'model.h5'


class Trainer(BaseTrainer):
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
        super(Trainer, self).__init__(model, training_parameters, dataset, target_processing, batch_processing,
                                      post_processing)
        assert model.backend == BACKEND, 'The model backend is not keras: %s' % model.backend

        # Read out data-related training parameters
        self.image_batch_size = self._training_parameters.get('image_minibatch', 32)
        self.minibatch_size = self._training_parameters.get('training_minibatch', 32)
        self.epochs = self._training_parameters.get('epochs', 20)
        self.training_batches = self._training_parameters.get('training_batches', 50)
        self.validation_batches = self._training_parameters.get('validation_batches', 5)
        self.workers = self._training_parameters.get('workers', 1)
        self.input_shape = self._training_parameters.get('input_shape', (17, 17))

    def train(self):
        """
        Training loop
        
        :return: 
        """
        def minibatch_generator(is_validation):
            while True:
                # Get some images from dataset
                images = self._dataset.get_minibatch(self.image_batch_size, is_validation)
                # Do target processing on those images
                x, y, meta = self._target_processing(images)
                # Convert each pair of (x, y) to some training points (usually sub-sampling/patch)
                batch_x, batch_y, meta = self._batch_processing(x, y, meta,
                                                                self.input_shape, self.minibatch_size,
                                                                is_random=True)
                yield batch_x, batch_y

        # Start training and timer
        start = time.time()
        self._logger.info("Start training the keras model.")
        self._model.model.fit_generator(self, minibatch_generator(False), self.training_batches, epochs=self.epochs,
                                        validation_data=minibatch_generator(False),
                                        validation_steps=self.validation_batches,
                                        workers=self.workers)
        end = time.time()
        elapsed = end - start
        self._logger.info("End of training after %.3f seconds." % elapsed)

    def eval(self, image):
        """
        Process all images in the dataset

        :param: image: ndarray of shape (w, h, 1) or (w, h, 3), image to be processed
        :return: ndarray, processed image
        """
        # Timed operation
        start = time.time()
        self._logger.info("Start individual processing.")
        images = image.reshape(1, )
        x, y, meta = self._target_processing(images)
        # batch_x can be larger than the minibatch_size when is_random is set to False
        batch_x, batch_y, meta = self._batch_processing(x, y, meta, self.input_shape, -1, is_random=False)
        post_x = self._model.model.predict(batch_x)
        output = self._post_processing(post_x, meta)
        end = time.time()
        elapsed = end - start
        self._logger.info("Processing completed in %.3f seconds." % elapsed)
        return output


class Model(BaseModel):
    def __init__(self, model):
        """
        Keras model wrapper

        :param model: Model (from keras.models), already compiled, can take multiple inputs/outputs
        """
        super(Model, self).__init__(model, backend=BACKEND)

    def save(self, path):
        """
        Save the model to the file system

        :param path: string, root folder for the model file
        :return:
        """
        self._model.save(os.path.join(path, MODEL_FILENAME))


def load(path):
    """
    Load a pre-trained model from the file system

    :param path: string, path to the folder containing the model file
    :return:
    """
    from keras.models import load_model
    return load_model(os.path.join(path, MODEL_FILENAME))
