"""
Keras backend 

The backend should exposes:
    - Class: Trainer
    - Class: Model
    - Function: load
"""
import os
import time

from keras.callbacks import TensorBoard

from nnimgproc.model import BaseModel
from nnimgproc.trainer import BaseTrainer

# Name of the backend depends on the file name
BACKEND = os.path.basename(__file__)
# Name of the saved model file
MODEL_FILENAME = 'model.h5'


class Trainer(BaseTrainer):
    def __init__(self, model, params, dataset, target_processor,
                 batch_processor, model_processor):
        """
        Initialize a neural network trainer/optimizer

        :param model: Model (from nnimgproc.backend.keras)
        :param params: Parameters (from nnimgproc.util.parameters),
                       training parameter set such as learning rate
        :param dataset: Dataset (from nnimgproc.dataset), image provider
        :param target_processor: TargetProcessor (from nnimgproc.processor)
        :param batch_processor: BatchProcessor (from nnimgproc.processor)
        :param model_processor: ModelProcessor (from nnimgproc.processor)
        """
        super(Trainer, self).__init__(model, params, dataset,
                                      target_processor, batch_processor,
                                      model_processor)
        assert model.backend == BACKEND, 'The model backend is not keras: %s' \
                                         % model.backend

        # Read out data-related training parameters
        self._raw_minibatch = self._params.get('image_minibatch')
        self._minibatch = self._params.get('training_minibatch')
        self._epochs = self._params.get('epochs')
        self._train_batches = self._params.get('training_batches')
        self._val_batches = self._params.get('validation_batches')
        self._workers = self._params.get('workers')
        self._input_shape = self._params.get('input_shape')

        self._logger.info('Trainer (%s) created.' % self._model.backend)

    def train(self):
        """
        Training loop
        
        :return: 
        """
        tensorboard = TensorBoard(log_dir=self._output_dir,
                                  batch_size=self._minibatch)

        def minibatch_generator(is_validation):
            while True:
                # Get some images from dataset
                images = self._dataset.get_minibatch(self._raw_minibatch,
                                                     is_validation)
                # Do target processing on those images
                x, y, meta = self._target_processor(images)
                # Convert pairs of (x, y) to some training points (usually
                # sub-sampling/patch)
                batch_x, batch_y = self._batch_processor(x, y, meta)
                yield batch_x, batch_y

        # Start training and timer
        start = time.time()
        self._logger.info("Start training the keras model.")
        self._model.model.fit_generator(self, minibatch_generator(False),
                                        self._train_batches,
                                        epochs=self._epochs,
                                        validation_data=minibatch_generator(
                                            is_validation=False
                                        ),
                                        validation_steps=self._val_batches,
                                        workers=self._workers,
                                        callbacks=[tensorboard])
        end = time.time()
        elapsed = end - start
        self._logger.info("End of training after %.3f seconds." % elapsed)


class Model(BaseModel):
    def __init__(self, model):
        """
        Keras model wrapper

        :param model: Model (from keras.models), already compiled, can take
                      multiple inputs/outputs
        """
        super(Model, self).__init__(model, backend=BACKEND)
        self._logger.info('Model (%s) created.' % self._backend)

    def save(self, path: str):
        """
        Save the model to the file system

        :param path: string, root folder for the model file
        :return:
        """
        self._model.save(os.path.join(path, MODEL_FILENAME))
        self._logger.info('Model saved under: %s' % path)


def load(path: str):
    """
    Load a pre-trained model from the file system

    :param path: string, path to the folder containing the model file
    :return:
    """
    from keras.models import load_model
    return load_model(os.path.join(path, MODEL_FILENAME))
