"""
What is desteaking?

When computing inverse Radon transform using Filter back projection,
streaks (line artifacts) would appear if information from some angles are
missing. A popular way to remove them is to optimize some loss function
in the image and Radon transform domain, such loss functions are exquisitely
studied in compressive sensing.

Mode (streak_type and streak_params):
- "periodic", [number of angles to keep]: angles are periodic
- "uniform", [number of angles to keep]: angles are uniformly distributed
- "fix", [list of angles valued from 0 to 180 (exclusive)]

"""

import numpy as np
from typing import Tuple, Optional

from skimage.color import rgb2grey
from skimage.transform import radon, iradon

from nnimgproc.processor import TargetProcessor
from nnimgproc.util.parameters import Parameters


# TargetProcessor for destreaking
class DestreakingTargetProcessor(TargetProcessor):
    def __init__(self, streak_type: str, streak_params: list):
        super(DestreakingTargetProcessor, self).__init__()
        self._streak_type = streak_type
        self._streak_params = streak_params
        self._params = Parameters()

    # Noise definitions
    def __call__(self, img: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, Optional[Parameters]]:
        """
        Compute Radon reconstruction using Filtered Back Projection method.
        The input can be color image, but it will be converted to black and
        white. The output will always be black and white.

        :param img: ndarray of shape (w, h, 1) for grey image or (w, h, 3)
        :return: ndarray of shape (w, h, 1)
        """

        # Convert img to 2D array
        if img.shape[2] == 3:
            img = rgb2grey(img)
        else:
            img = img[:, :, 0]

        # Compute the right angles according to the parameters
        if self._streak_type == 'periodic':
            theta = np.linspace(0., 180., int(self._streak_params[0]),
                                  endpoint=False)
        elif self._streak_type == "uniform":
            theta = np.random.uniform(0., 180., int(self._streak_params[0]))
        elif self._streak_type == "fix":
            theta = map(float, self._streak_params)
        else:
            raise NotImplementedError('%s streaking is not implemented' %
                                      self._streak_type)
        self._params.set('angles', theta)
        sinogram = radon(img, theta=theta, circle=False)
        x = np.expand_dims(iradon(sinogram, theta=theta, circle=False),
                           axis=2)

        y = np.expand_dims(img, axis=2)

        return x.clip(0, 1), y, self._params
