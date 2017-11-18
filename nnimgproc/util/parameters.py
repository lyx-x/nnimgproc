import logging
from pickle import dump, load
from typing import Any


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

    def set(self, key: str, value: Any):
        """
        Insert a new key-value pair. If value is mutable, be careful when
        changing its value in another scope.

        :param key: string
        :param value: value (any type)
        :return:
        """
        self._dict[key] = value

    def get(self, key: str, default: Any=None) -> Any:
        """
        Query the dictionary with a default value

        :param key: string
        :param default: default value is the key is not present
        :return: value (any type)
        """
        if key in self._dict:
            value = self._dict[key]
        elif default is not None:
            value = default
        else:
            raise ValueError('Parameter %s doesn\'t exist.' % key)
        return value

    def save(self, path: str):
        """
        Save the object to local file system

        :param path: string, path to the file
        :return:
        """
        file = open(path, 'wb')
        dump(self._dict, file)
        file.close()

    def load(self, path: str):
        """
        Load the object from local file system

        :param path: string, path to the file
        :return:
        """
        file = open(path, 'rb')
        self._dict = load(file)
        file.close()
