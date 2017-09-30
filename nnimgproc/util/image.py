from skimage.color import gray2rgb
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.viewer import ImageViewer


def read(path, shape, as_grey=True):
    """
    Read an image

    :param path: string
    :param shape: tuple of 2 integers
    :param as_grey: bool
    :return: ndarray of shape (w, h, 1) for grey image or (w, h, 3) for RBG images
    """
    image = imread(path, as_grey=as_grey)
    if as_grey:
        image = resize(image, (shape[0], shape[1], 1))
    else:
        if image.ndim < 3:
            # If greyscale image, duplicate the channels
            image = resize(gray2rgb(image), (shape[0], shape[1], 3))
        else:
            # No need for the alpha channel
            image = resize(image[:, :, :3], (shape[0], shape[1], 3))

    assert image.ndim == 3, "The image should be read in shape (width, height, channels)."
    return image


def write(image, path):
    """
    Save the image to local disk

    :param image: ndarray of shape (w, h, 1) or (w, h, 3)
    :param path: string
    :return:
    """
    if image.ndim > 2 and image.shape[2] == 1:
        imsave(image[:, :, 0], path)
    else:
        imsave(image, path)


def show(image):
    """
    Display the image

    :param image: ndarray of shape (w, h, 1) or (w, h, 3)
    :return:
    """
    if image.ndim > 2 and image.shape[2] == 1:
        viewer = ImageViewer(image[:, :, 0])
    else:
        viewer = ImageViewer(image)
    viewer.show()
