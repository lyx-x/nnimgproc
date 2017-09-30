import importlib

from nnimgproc.util.parameters import default_parameters


def build_model(model, backend):
    """
    Build a correct model wrapper given the backend
    :param model: Model (from backend), see backend folder for more details
    :param backend: string, name of the backend such as 'keras'
    :return: Model (from nnimgproc.model)
    """
    lib = importlib.import_module('nnimgproc.backend.%s' % backend)
    return lib.Model(model)


def load_model(path, backend):
    """
    Load a pre-trained model given the backend
    :param path: string, folder under which the old model is saved
    :param backend: string, name of the backend
    :return: Model (from nnimgproc.model)
    """
    lib = importlib.import_module('nnimgproc.backend.%s' % backend)
    return lib.Model(lib.load(path))


def build_trainer(model, training_parameters=default_parameters, dataset=None,
                  target_processing=None, batch_processing=None, post_processing=None):
    """
    Build a neural network trainer/optimizer based on different backend

    :param model: Model (from nnimgproc.backend.keras)
    :param training_parameters: Parameters (from nnimgproc.util.parameters), training parameter set
    :param dataset: Dataset (from nnimgproc.dataset), image minibatch provider
    :param target_processing: lambda img -> (input, output, meta), image processing pipeline to imitate.
                             The meta contains some parameters used in the processing pipeline which
                             can then be used to help learning or evaluating the result
    :param batch_processing: lambda function (x, y, meta, shape, batch_size, is_random) -> (x, y, meta),
                             convert images to patches. If is_random is False, the processor will act in a
                             deterministic way which (together with meta) will be used to rebuild the image
    :param post_processing: (batch_x, meta) -> img, rebuild one image from batches which may be patches of it
    :return: Trainer (from nnimgproc.trainer)
    """
    lib = importlib.import_module('nnimgproc.backend.%s' % model.backend)
    return lib.Trainer(model, training_parameters, dataset, target_processing, batch_processing, post_processing)
