import importlib
import logging
from typing import Any

from nnimgproc.dataset import Dataset
from nnimgproc.model import BaseModel
from nnimgproc.processor import TargetProcessor, BatchProcessor, ModelProcessor
from nnimgproc.tester import Tester
from nnimgproc.trainer import BaseTrainer
from nnimgproc.util.parameters import Parameters

logging.basicConfig(level=logging.INFO)


def build_model(model: Any, backend: str) -> BaseModel:
    """
    Build a correct model wrapper given the backend
    :param model: any neural network models, see backend folder for more
                  details
    :param backend: string, name of the backend such as 'keras'
    :return: Model (from nnimgproc.model)
    """
    lib = importlib.import_module('nnimgproc.backend.%s' % backend)
    return lib.Model(model)


def load_model(model: Any, path: str, backend: str) -> BaseModel:
    """
    Load a pre-trained model given the backend
    :param model: any neural network models, may be None
    :param path: string, folder under which the old model is saved
    :param backend: string, name of the backend
    :return: Model (from nnimgproc.model)
    """
    lib = importlib.import_module('nnimgproc.backend.%s' % backend)
    return lib.load(model, path)


def build_trainer(model: BaseModel,
                  params: Parameters,
                  dataset: Dataset,
                  target_processor: TargetProcessor,
                  batch_processor: BatchProcessor) \
        -> BaseTrainer:
    """
    Build a neural network trainer/optimizer based on different backend

    :param model: Model (inherited from nnimgproc.model.BaseModel)
    :param params: Parameters (from nnimgproc.util.parameters),
                   training parameter set such as learning rate
    :param dataset: Dataset (from nnimgproc.dataset), image provider
    :param target_processor: TargetProcessor (from nnimgproc.processor)
    :param batch_processor: BatchProcessor (from nnimgproc.processor)
    :return: Trainer (from nnimgproc.trainer)
    """
    lib = importlib.import_module('nnimgproc.backend.%s' % model.backend)
    return lib.Trainer(model, params, dataset,
                       target_processor, batch_processor)


def build_tester(model: BaseModel, model_processor: ModelProcessor) \
        -> Tester:
    """
    Build a tester object

    :param model: Model (inherited from nnimgproc.model.BaseModel)
    :param model_processor: ModelProcessor (from nnimgproc.processor)
    :return: Tester (from nnimgproc.tester)
    """
    return Tester(model, model_processor)
