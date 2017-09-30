import logging
from os import path

from nnimgproc.model import BaseModel
from nnimgproc.trainer import BaseTrainer

# name of the backend depends on the file name
BACKEND = path.basename(__file__)


class Trainer(BaseTrainer):
    def __init__(self, model, learning_parameters, dataset, target_processing, reshaper):
        super(Trainer, self).__init__(model, learning_parameters, dataset, target_processing, reshaper)
        self._logger = logging.getLogger(__name__)
        assert model.backend == BACKEND, 'The model backend is not keras: %s' % model.backend


class Model(BaseModel):
    def __init__(self, model):
        super(Model, self).__init__(model, backend=BACKEND)
        self._logger = logging.getLogger(__name__)
