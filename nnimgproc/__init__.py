import importlib

from nnimgproc.util.parameters import default_parameters


def build_model(model, backend):
    lib = importlib.import_module('nnimgproc.backend.%s' % backend)
    return lib.Model(model)


def build_trainer(model, learning_parameters=default_parameters, dataset=None,
                  target_processing=None, reshaper=(None, None)):
    """
    Build a neural network trainer/optimizer based on different backends

    :param model: neural network model
    :param learning_parameters: training parameters in form of a dictionary
    :param dataset: image minibatch provider
    :param target_processing: image processing pipeline to imitate
    :param reshaper: a tuple of pre/post-processing methods that are used to create minibatches
    :return: trainer object
    """
    lib = importlib.import_module('nnimgproc.backend.%s' % model.backend)
    return lib.Trainer(model, learning_parameters, dataset, target_processing, reshaper)
