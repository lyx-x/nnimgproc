"""
Chainer backend

The backend should exposes:
    - Class: Trainer
    - Class: Model
    - Function: load
"""
import numpy as np
import os
import time
from typing import Any

from chainer import Chain, Variable
from chainer.cuda import to_cpu, to_gpu
from chainer.functions import absolute_error, sum
from chainer.optimizers import Adam
from chainer.serializers.hdf5 import save_hdf5, load_hdf5

from nnimgproc.dataset import Dataset
from nnimgproc.model import BaseModel
from nnimgproc.processor import TargetProcessor, BatchProcessor
from nnimgproc.trainer import BaseTrainer
from nnimgproc.util.parameters import Parameters

# Name of the backend depends on the file name
BACKEND = str(os.path.basename(__file__).split('.')[0])
# Name of the saved model file
MODEL_FILENAME = 'model.h5'


# Helper function to move a container of arrays to GPU or CPU
def to_device(obj: Any, device: int) -> Any:
    if device >= 0:
        def move(x):
            return Variable(to_gpu(x, device=device))
    else:
        def move(x):
            return Variable(to_cpu(x))
    if isinstance(obj, dict):
        return dict(map(lambda a: (a[0], move(a[1])), obj.items()))
    if isinstance(obj, list):
        return list(map(move, obj))
    if isinstance(obj, tuple):
        return tuple(map(move, obj))
    else:
        return move(obj)


class Model(BaseModel):
    def __init__(self, model: Chain):
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
        save_hdf5(os.path.join(path, MODEL_FILENAME), obj=self._model)
        self._logger.info('Model saved under: %s' % path)


def load(model: Chain, path: str) -> Model:
    """
    Load a pre-trained model from the file system

    :param model: None, keras doesn't need any object to help loading weights
    :param path: string, path to the folder containing the model file
    :return: Model
    """
    load_hdf5(os.path.join(path, MODEL_FILENAME), obj=model)
    return Model(model=model)


class Trainer(BaseTrainer):
    def __init__(self, model: Model, params: Parameters, dataset: Dataset,
                 target_processor: TargetProcessor,
                 batch_processor: BatchProcessor):
        """
        Initialize a neural network trainer/optimizer

        :param model: Model (from nnimgproc.backend.keras)
        :param params: Parameters (from nnimgproc.util.parameters),
                       training parameter set such as learning rate
        :param dataset: Dataset (from nnimgproc.dataset), image provider
        :param target_processor: TargetProcessor (from nnimgproc.processor)
        """
        super(Trainer, self).__init__(model, params, dataset,
                                      target_processor, batch_processor)
        assert model.backend == BACKEND, 'The model backend is not keras: %s' \
                                         % model.backend

        # Read out data-related training parameters
        self._learning_rate = self._params.get('learning_rate')
        self._raw_minibatch = self._params.get('image_minibatch')
        self._minibatch = self._params.get('training_minibatch')
        self._epochs = self._params.get('epochs')
        self._train_batches = self._params.get('training_batches')
        self._val_batches = self._params.get('validation_batches')
        self._device = self._params.get('device')

        # Move the model to GPU if needed and set a move function for arrays
        if self._device >= 0:
            self._model.model.to_gpu(device=self._device)
        self._move = lambda x: to_device(obj=x, device=self._device)

        self._logger.info('Trainer (%s) created.' % self._model.backend)

    def train(self):
        """
        Training loop
        
        :return: 
        """
        # Choose an optimizer algorithm
        optimizer = Adam(alpha=self._learning_rate)
        optimizer.setup(self._model.model)

        # This generator seems to be thread unsafe
        def minibatch_generator(is_validation: bool):
            while True:
                # Get some images from dataset
                images = self._dataset.get_minibatch(self._raw_minibatch,
                                                     is_validation)
                # Do target processing on all images
                xs = []
                ys = []
                metas = []
                for i in range(self._raw_minibatch):
                    x, y, meta = self._target_processor(images[i])
                    xs.append(x)
                    ys.append(y)
                    metas.append(meta)

                # Convert pairs of (x, y) to some training points (usually
                # sub-sampling/patch).
                batch_x, batch_y = self._batch_processor(xs, ys, metas)
                yield batch_x, batch_y

        # Start training and timer
        start = time.time()
        self._logger.info("Start training the keras model.")

        zero = np.zeros(1, np.float32)
        if self._device >= 0:
            zero = self._move(zero)
        zero = sum(zero)

        epoch = 0
        step = 0

        for batch_x, batch_y in minibatch_generator(False):
            d_batch_x = self._move(batch_x)
            d_batch_y = self._move(batch_y)

            prediction_train = self._model.model(d_batch_x, d_batch_y)
            loss = absolute_error(prediction_train, zero)
            self._model.model.cleargrads()
            loss.backward()
            optimizer.update()

            step += 1
            if step >= self._train_batches:
                step = 0
                epoch += 1

                step_val = 0
                loss_val = 0
                for batch_x_val, batch_y_val in minibatch_generator(True):
                    d_batch_x_val = self._move(batch_x_val)
                    d_batch_y_val = self._move(batch_y_val)
                    prediction_val = self._model.model(d_batch_x_val,
                                                       d_batch_y_val)
                    loss_val += absolute_error(prediction_val, zero)
                    step_val += 1
                    if step_val >= self._val_batches:
                        break
                loss_val = loss_val / self._val_batches

                msg = 'epoch:{:02d} train_mse_loss:{:.04f} ' \
                      'val_mse_loss:{:.04f}'.format(epoch,
                                                    float(to_cpu(loss.data)),
                                                    float(to_cpu(
                                                        loss_val.data)))
                self._logger.info(msg)

            if epoch >= self._epochs:
                break

        end = time.time()
        elapsed = end - start
        self._logger.info("End of training after %.3f seconds." % elapsed)
