import logging


class Parameters(object):
    """
    A wrapper around dictionary to help write shorter function signature
    """
    def __init__(self):
        """
        Constructor
        """
        self._dict = {}
        self._logger = logging.getLogger(__name__)

    def add(self, key, value):
        """
        Insert a new key-value pair

        :param key: string
        :param value: value (any type)
        :return:
        """
        self._dict[key] = value

    def get(self, key, default):
        """
        Query the dictionary with a default value

        :param key: string
        :param default: default value is the key is not present
        :return: value (any type)
        """
        if key in self._dict:
            value = self._dict[key]
        else:
            value = default
        self._logger.info("Parameters retrieved: {}, {}.".format(key, value))


# Construct a set of default values
default_parameters = Parameters()
default_parameters.add("learning_rate", 1e-4)
