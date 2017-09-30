import logging
import numpy as np
from os import listdir
from os.path import isfile, join

from nnimgproc.util.image import read


class Dataset(object):
    """
    Abstraction of dataset/data pool which is the only interface between model and local file system
    """
    def __init__(self, shape, max_size, as_grey):
        """
        Dataset constructor

        :param shape: tuple of 2 integers, same across all images
        :param max_size: integer, maximum size of the dataset (due to memory limits)
        :param as_grey: bool, whether or not images are in greyscale
        """
        self._logger = logging.getLogger(__name__)
        self._shape = shape
        self._max_size = max_size
        self._as_grey = as_grey
        self._images = np.ndarray((self._max_size, self._shape[0], self._shape[1], 1 if self._as_grey else 3),
                                  dtype=np.float32)
        self._size = 0

    def add(self, image):
        """
        Add an image to the dataset

        :param image: ndarray of shape (w, h, channels)
        :return: bool, true if the dataset is not full, false otherwise
        """
        if self._size < self._max_size:
            self._images[self._size] = image
            self._size += 1
            return True
        else:
            return False

    def get(self):
        """
        Retrieve a random image from the dataset

        :return: ndarray of shape (w, h, 1) or (w, h, 3)
        """
        index = np.random.randint(0, self._size)
        return self._images[index]

    def get_minibatch(self, size):
        """
        Retrieve randomly a set of images

        :param size: integer, number of images to be retrieved
        :return: ndarray of shape (size, w, h, 1 or 3)
        """
        indices = np.random.randint(0, self._size, size)
        return self._images[indices]


class ImageFolder(Dataset):
    """
    A folder full of images
    """
    def __init__(self, folder, shape=(128, 128), max_size=2000, as_grey=True):
        """
        ImageFolder dataset constructor: retrieve all images directly under the root folder

        :param folder: string, path to the root folder
        :param shape: tuple of 2 integers, same across all images
        :param max_size: integer, maximum size of the dataset (due to memory limits)
        :param as_grey: bool, whether or not images are in greyscale
        """
        super(ImageFolder, self).__init__(shape, max_size, as_grey)
        self._folder = folder
        # Read all image files under the folder until the dataset is full
        for filename in listdir(folder):
            path = join(folder, filename)
            if isfile(path):
                ok = self.add(read(path, self._shape, self._as_grey))
                if not ok:
                    self._logger.info("Maximum dataset size reached: %d" % self._size)
                    break


class ImageSingleton(Dataset):
    """
    Dataset with one single image, used when we need to test on one single image
    """
    def __init__(self, path, shape=(128, 128), as_grey=True):
        """
        ImageSingleton dataset constructor: read only one image

        :param path: string, path to the image file
        :param shape: tuple of 2 integers, same across all images
        :param as_grey: bool, whether or not images are in greyscale
        """
        super(ImageSingleton, self).__init__(shape, 1, as_grey)
        if isfile(path):
            ok = self.add(read(path, self._shape, self._as_grey))
            if not ok:
                self._logger.error("Not enough space for just one image, impossible.")
        else:
            self._logger.error("Not a path: %s" % path)