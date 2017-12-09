"""
What is denoising?

Noise is modeled as samples from a certain distribution, it can be referred
with the name of distribution it follows, for example, a Gaussian noise.

Mode (noise_type and noise_params):
- "gaussian", [standard deviation of the distribution]: Gaussian noise
- "poisson", [peak value]: Poisson noise

"""

import numpy as np
from typing import Tuple, Optional

from nnimgproc.processor import TargetProcessor
from nnimgproc.util.parameters import Parameters


# TargetProcessor for denoising
class DenoisingTargetProcessor(TargetProcessor):
    def __init__(self, noise_type: str, noise_params: list):
        super(DenoisingTargetProcessor, self).__init__()
        self._noise_type = noise_type
        self._noise_params = noise_params
        self._params = Parameters()

    # Noise definitions
    def __call__(self, img: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, Optional[Parameters]]:
        y = img
        if self._noise_type == 'gaussian':
            sigma = float(self._noise_params[0])
            if sigma == 0:
                x = np.copy(img)
            else:
                x = (img + np.random.normal(0, 1, img.shape) * sigma)
            self._params.set('noise_level', sigma)
        elif self._noise_type == "poisson":
            peak = float(self._noise_params[0])
            x = (np.random.poisson(img * peak) / peak)
            self._params.set('noise_level', peak)
        else:
            raise NotImplementedError('%s noise is not implemented' %
                                      self._noise_type)
        return x.clip(0, 1), y, self._params
